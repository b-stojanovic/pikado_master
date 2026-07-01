import streamlit as st
import json
from math import comb, exp
from collections import defaultdict, Counter
from datetime import date, datetime
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from collections import defaultdict, Counter, deque

# ---------- path to main JSON file ----------
JSON_FILE = "master.json"  # <-- prilagodi ime datoteke

# ---------- load data into session_state ----------
def init_session_state():
    if 'players' not in st.session_state:
        with open(JSON_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
        st.session_state.players = {p['id']: p['name'] for p in data['players']}
        st.session_state.matches = data['matches']
    if 'new_matches' not in st.session_state:
        st.session_state.new_matches = []

# ---------- leg win rate ----------
def compute_leg_winrates(matches):
    legs_won = defaultdict(int)
    legs_total = defaultdict(int)
    for m in matches:
        p1, p2 = m['p1id'], m['p2id']
        a, b = m['p1legs'], m['p2legs']
        legs_won[p1] += a
        legs_won[p2] += b
        legs_total[p1] += a + b
        legs_total[p2] += a + b
    return {pid: legs_won[pid] / legs_total[pid] for pid in legs_won}

# ---------- Elo ratings ----------
def compute_elo_ratings(matches):
    elo = defaultdict(lambda: 1500)
    K = 40
    sorted_matches = sorted(matches, key=lambda m: (m['date'], m.get('_meta', {}).get('time', '')))
    for m in sorted_matches:
        p1, p2 = m['p1id'], m['p2id']
        if m['p1legs'] > m['p2legs']:
            winner, loser = p1, p2
        elif m['p2legs'] > m['p1legs']:
            winner, loser = p2, p1
        else:
            continue
        expected_winner = 1 / (1 + exp((elo[loser] - elo[winner]) / 400))
        expected_loser = 1 - expected_winner
        elo[winner] += K * (1 - expected_winner)
        elo[loser] += K * (0 - expected_loser)
    return elo

def elo_leg_prob(player_elo, opponent_elo):
    return 1 / (1 + exp((opponent_elo - player_elo) / 400))

def leg_prob(p1_id, p2_id, elo_ratings):
    e1 = elo_ratings.get(p1_id, 1500)
    e2 = elo_ratings.get(p2_id, 1500)
    return elo_leg_prob(e1, e2)

# ---------- match distribution ----------
def match_distribution(p, N=4):
    dist = {}
    for b in range(N):
        a = N
        n = a + b - 1
        dist[(a, b)] = comb(n, a-1) * (p ** a) * ((1-p) ** b)
    for a in range(N):
        b = N
        n = a + b - 1
        dist[(a, b)] = comb(n, b-1) * (p ** a) * ((1-p) ** b)
    return dist

def handicap_prob(p, H, N=4):
    dist = match_distribution(p, N)
    prob = 0.0
    for (a, b), vj in dist.items():
        if a + H > b:
            prob += vj
    return prob

AVAILABLE_HANDICAPS = [-2.5, -1.5, 1.5, 2.5]

def gold_tip(p_leg):
    best_prob = 0
    best_side = None
    best_h_val = None
    for h in AVAILABLE_HANDICAPS:
        prob_home = handicap_prob(p_leg, h)
        prob_away = 1 - prob_home
        if prob_home > best_prob:
            best_prob = prob_home
            best_side = "1 (domaćin)"
            best_h_val = h
        if prob_away > best_prob:
            best_prob = prob_away
            best_side = "2 (gost)"
            best_h_val = h
    return best_side, best_h_val, best_prob

def gold_tip_player(p_leg):
    best_prob = 0
    best_h = None
    for h in AVAILABLE_HANDICAPS:
        ph = handicap_prob(p_leg, h)
        if ph > best_prob:
            best_prob = ph
            best_h = h
    return best_h, best_prob

def h2h_history(p1_id, p2_id, matches):
    return [m for m in matches if (m['p1id'] == p1_id and m['p2id'] == p2_id)
            or (m['p1id'] == p2_id and m['p2id'] == p1_id)]

def is_duplicate(new_match, existing_matches):
    if 'id' in new_match:
        for m in existing_matches:
            if m.get('id') == new_match['id']:
                return True, m
    for m in existing_matches:
        if (m['p1id'] == new_match['p1id'] and 
            m['p2id'] == new_match['p2id'] and
            m['date'] == new_match['date'] and
            m['p1legs'] == new_match['p1legs'] and
            m['p2legs'] == new_match['p2legs'] and
            m['_meta'] == new_match['_meta']):
            return True, m
    return False, None

def similar_match_exists(new_match, existing_matches):
    for m in existing_matches:
        if (m['p1id'] == new_match['p1id'] and 
            m['p2id'] == new_match['p2id'] and
            m['date'] == new_match['date']):
            if (m['p1legs'] != new_match['p1legs'] or 
                m['p2legs'] != new_match['p2legs'] or 
                m['_meta'] != new_match['_meta']):
                return True, m
    return False, None

def save_to_json():
    data = {
        "exported_at": date.today().isoformat(),
        "source": "Modus Super Series",
        "players": [{"id": pid, "name": name} for pid, name in st.session_state.players.items()],
        "matches": st.session_state.matches
    }
    with open(JSON_FILE, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    st.session_state.new_matches = []

def handicap_winner(legs_selected, legs_opponent, handicap):
    return (legs_selected + handicap) > legs_opponent

def merge_daily_json(daily_data, existing_players, existing_matches):
    updated_players = existing_players.copy()
    for p in daily_data.get('players', []):
        pid = p['id']
        if pid not in updated_players:
            updated_players[pid] = p['name']
    updated_matches = existing_matches.copy()
    added, skipped = 0, 0
    for m in daily_data.get('matches', []):
        if is_duplicate(m, updated_matches)[0]:
            skipped += 1
            continue
        updated_matches.append(m)
        added += 1
    return updated_players, updated_matches, added, skipped

def compute_player_180_stats(matches):
    stats = defaultdict(list)
    for m in matches:
        stats[m['p1id']].append(m.get('p1_180s', 0))
        stats[m['p2id']].append(m.get('p2_180s', 0))
    return stats

def get_180_distribution(player_id, stats_180):
    counts = stats_180.get(player_id, [0])
    if not counts:
        return {0: 1.0}
    freq = Counter(counts)
    total = len(counts)
    return {k: v/total for k, v in freq.items()}

def fer_koef(prob):
    return round(1 / prob, 2) if prob > 0 else None

# ---------- find all duplicates ----------
def find_all_duplicates(matches):
    groups = defaultdict(list)
    for m in matches:
        key = (
            m['p1id'], m['p2id'], m['date'],
            m['p1legs'], m['p2legs'],
            tuple(sorted(m.get('_meta', {}).items()))
        )
        groups[key].append(m)
    return {k: v for k, v in groups.items() if len(v) > 1}

# ---------- ADVANCED MODEL (Random Forest with volatility) ----------
def compute_player_180_avg(matches):
    player_180 = defaultdict(list)
    for m in matches:
        player_180[m['p1id']].append(m.get('p1_180s', 0))
        player_180[m['p2id']].append(m.get('p2_180s', 0))
    return {pid: sum(v)/len(v) for pid, v in player_180.items()}

@st.cache_resource
def train_advanced_models(matches, elo_ratings):
    player_matches = defaultdict(list)
    for m in matches:
        player_matches[m['p1id']].append(m)
        player_matches[m['p2id']].append(m)

    player_stats = {}
    for pid, pms in player_matches.items():
        total_won = sum(m['p1legs'] if m['p1id'] == pid else m['p2legs'] for m in pms)
        total_lost = sum(m['p2legs'] if m['p1id'] == pid else m['p1legs'] for m in pms)
        avg_margin = (total_won - total_lost) / len(pms)
        player_stats[pid] = {'avg_margin': avg_margin, 'total': len(pms)}

    avg_180 = compute_player_180_avg(matches)

    sorted_matches = sorted(matches, key=lambda x: (x['date'], x.get('_meta', {}).get('time', '')))

    day_counter = defaultdict(int)
    recent_results = defaultdict(list)   # lista dictova {'won': bool, 'margin': int}
    recent_180 = defaultdict(list)       # lista brojeva 180s

    feature_cols = ['elo_diff', 'hist_pct', 'h2h_pct', 'avg_margin',
                    'recent_margin', 'recent_winrate',
                    'matches_today', 'avg_180_diff', 'recent_180s_avg',
                    'volatility']  # dodana volatilnost

    handicap_data = {h: [] for h in AVAILABLE_HANDICAPS}

    for m in sorted_matches:
        for player_id, opponent_id in [(m['p1id'], m['p2id']), (m['p2id'], m['p1id'])]:
            if player_id == m['p1id']:
                player_legs = m['p1legs']
                opponent_legs = m['p2legs']
                player_180s = m.get('p1_180s', 0)
                opponent_180s = m.get('p2_180s', 0)
            else:
                player_legs = m['p2legs']
                opponent_legs = m['p1legs']
                player_180s = m.get('p2_180s', 0)
                opponent_180s = m.get('p1_180s', 0)

            elo_diff = elo_ratings[player_id] - elo_ratings[opponent_id]
            avg_margin = player_stats[player_id]['avg_margin']

            day_key = (player_id, m['date'])
            matches_today = day_counter[day_key]
            day_counter[day_key] += 1

            recent = recent_results[player_id][-5:]
            if recent:
                margins = [r['margin'] for r in recent]
                recent_wins = sum(1 for r in recent if r['won']) / len(recent)
                recent_margin = sum(margins) / len(margins)
                volatility = (sum((mg - recent_margin) ** 2 for mg in margins) / len(margins)) ** 0.5
            else:
                recent_wins = 0.5
                recent_margin = 0.0
                volatility = 2.0

            recent180_list = recent_180[player_id][-5:]
            if recent180_list:
                recent_180s_avg = sum(recent180_list) / len(recent180_list)
            else:
                recent_180s_avg = avg_180.get(player_id, 0.0)

            avg_180_diff = avg_180.get(player_id, 0.0) - avg_180.get(opponent_id, 0.0)

            player_past = [pm for pm in player_matches[player_id] if pm is not m]
            total_past = len(player_past) or 1

            h2h_all = h2h_history(player_id, opponent_id, matches)
            h2h_past = [hm for hm in h2h_all if hm is not m]
            h2h_total_past = len(h2h_past)

            for h in AVAILABLE_HANDICAPS:
                covered = 1 if (player_legs + h > opponent_legs) else 0

                hist_cnt = sum(1 for pm in player_past if (
                    (pm['p1legs'] if pm['p1id'] == player_id else pm['p2legs']) + h >
                    (pm['p2legs'] if pm['p1id'] == player_id else pm['p1legs'])
                ))
                hist_pct = hist_cnt / total_past

                if h2h_total_past >= 3:
                    h2h_cnt = sum(1 for hm in h2h_past if (
                        (hm['p1legs'] if hm['p1id'] == player_id else hm['p2legs']) + h >
                        (hm['p2legs'] if hm['p1id'] == player_id else hm['p1legs'])
                    ))
                    h2h_pct = h2h_cnt / h2h_total_past
                else:
                    h2h_pct = 0.5

                handicap_data[h].append([
                    elo_diff, hist_pct, h2h_pct, avg_margin,
                    recent_margin, recent_wins,
                    matches_today, avg_180_diff, recent_180s_avg,
                    volatility, covered
                ])

            won = 1 if player_legs > opponent_legs else 0
            margin = player_legs - opponent_legs
            recent_results[player_id].append({'won': won, 'margin': margin})
            recent_180[player_id].append(player_180s)

    models = {}
    for h in AVAILABLE_HANDICAPS:
        rows = handicap_data[h]
        if not rows:
            continue
        df = pd.DataFrame(rows, columns=feature_cols + ['target'])
        X = df[feature_cols]
        y = df['target']
        scaler_h = StandardScaler()
        X_scaled = scaler_h.fit_transform(X)
        model = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
        model.fit(X_scaled, y)
        models[h] = (model, scaler_h)

    return models

def predict_handicap_advanced(models, player_id, opponent_id, handicap, elo_ratings, all_matches):
    if handicap not in models:
        return 0.5, 0.0
    model, scaler_h = models[handicap]

    player_matches = [m for m in all_matches if m['p1id'] == player_id or m['p2id'] == player_id]
    total = len(player_matches)
    if total == 0:
        return 0.5, 2.0

    total_won = sum(m['p1legs'] if m['p1id'] == player_id else m['p2legs'] for m in player_matches)
    total_lost = sum(m['p2legs'] if m['p1id'] == player_id else m['p1legs'] for m in player_matches)
    avg_margin = (total_won - total_lost) / total

    hist_cnt = sum(1 for m in player_matches if (
        (m['p1legs'] if m['p1id'] == player_id else m['p2legs']) + handicap >
        (m['p2legs'] if m['p1id'] == player_id else m['p1legs'])
    ))
    hist_pct = hist_cnt / total

    h2h_matches = h2h_history(player_id, opponent_id, all_matches)
    h2h_total = len(h2h_matches)
    if h2h_total >= 3:
        h2h_cnt = sum(1 for m in h2h_matches if (
            (m['p1legs'] if m['p1id'] == player_id else m['p2legs']) + handicap >
            (m['p2legs'] if m['p1id'] == player_id else m['p1legs'])
        ))
        h2h_pct = h2h_cnt / h2h_total
    else:
        h2h_pct = 0.5

    elo_diff = elo_ratings.get(player_id, 1500) - elo_ratings.get(opponent_id, 1500)

    sorted_player_matches = sorted(player_matches, key=lambda x: (x['date'], x.get('_meta', {}).get('time', '')), reverse=True)
    last5 = sorted_player_matches[:5]
    if last5:
        margins = []
        wins = []
        for m in last5:
            if m['p1id'] == player_id:
                mg = m['p1legs'] - m['p2legs']
                w = 1 if m['p1legs'] > m['p2legs'] else 0
            else:
                mg = m['p2legs'] - m['p1legs']
                w = 1 if m['p2legs'] > m['p1legs'] else 0
            margins.append(mg)
            wins.append(w)
        recent_wins = sum(wins) / len(wins)
        recent_margin = sum(margins) / len(margins)
        volatility = (sum((mg - recent_margin) ** 2 for mg in margins) / len(margins)) ** 0.5
    else:
        recent_wins = 0.5
        recent_margin = 0.0
        volatility = 2.0

    matches_today = sum(1 for m in all_matches if m['date'] == date.today().isoformat() and (m['p1id'] == player_id or m['p2id'] == player_id))

    avg_180_all = compute_player_180_avg(all_matches)
    avg_180_diff = avg_180_all.get(player_id, 0.0) - avg_180_all.get(opponent_id, 0.0)

    recent_180_list = [m.get('p1_180s', 0) if m['p1id'] == player_id else m.get('p2_180s', 0) for m in last5]
    recent_180s_avg = sum(recent_180_list) / len(recent_180_list) if recent_180_list else avg_180_all.get(player_id, 0.0)

    X_new = pd.DataFrame([[elo_diff, hist_pct, h2h_pct, avg_margin,
                           recent_margin, recent_wins,
                           matches_today, avg_180_diff, recent_180s_avg,
                           volatility]],
                         columns=['elo_diff', 'hist_pct', 'h2h_pct', 'avg_margin',
                                  'recent_margin', 'recent_winrate',
                                  'matches_today', 'avg_180_diff', 'recent_180s_avg',
                                  'volatility'])
    X_scaled = scaler_h.transform(X_new)
    prob = model.predict_proba(X_scaled)[0, 1]
    return prob, volatility 

@st.cache_resource
def train_over_under_model(matches, elo_ratings):
    """Trenira Random Forest za predikciju over/under 5.5 (≥6 legova)."""
    # Priprema povijesnih podataka po igraču (bez curenja informacija)
    player_matches = defaultdict(list)
    for m in matches:
        player_matches[m['p1id']].append(m)
        player_matches[m['p2id']].append(m)

    player_stats = {}
    for pid, pms in player_matches.items():
        total_won = sum(m['p1legs'] if m['p1id'] == pid else m['p2legs'] for m in pms)
        total_lost = sum(m['p2legs'] if m['p1id'] == pid else m['p1legs'] for m in pms)
        avg_margin = (total_won - total_lost) / len(pms)
        player_stats[pid] = {'avg_margin': avg_margin, 'total': len(pms)}

    avg_180 = compute_player_180_avg(matches)

    sorted_matches = sorted(matches, key=lambda m: (m['date'], m.get('_meta', {}).get('time', '')))

    # Stanja za svakog igrača
    day_counter = defaultdict(int)
    recent_results = defaultdict(list)
    recent_180 = defaultdict(list)
    # Povijest over/under za svakog igrača (prije meča)
    player_over_count = defaultdict(int)
    player_total_count = defaultdict(int)

    feature_cols = [
        'elo_diff', 'avg_margin_diff', 'recent_margin_diff', 'recent_winrate_diff',
        'volatility_diff', 'matches_today_avg', 'hist_over_diff', 'h2h_over_pct',
        'avg_180_diff', 'recent_180s_diff'
    ]
    rows = []

    for m in sorted_matches:
        p1, p2 = m['p1id'], m['p2id']
        legs1, legs2 = m['p1legs'], m['p2legs']
        total_legs = legs1 + legs2
        target = 1 if total_legs >= 6 else 0

        # Značajke za oba igrača prije ovog meča
        # Elo razlika
        elo_diff = elo_ratings[p1] - elo_ratings[p2]

        # Ukupni prosjeci (već izračunati)
        avg_margin_p1 = player_stats[p1]['avg_margin']
        avg_margin_p2 = player_stats[p2]['avg_margin']

        # Nedavna forma (posljednjih 5 mečeva prije ovog)
        recent_p1 = recent_results[p1][-5:]
        recent_p2 = recent_results[p2][-5:]
        if recent_p1:
            margins_p1 = [r['margin'] for r in recent_p1]
            recent_margin_p1 = sum(margins_p1) / len(margins_p1)
            recent_winrate_p1 = sum(1 for r in recent_p1 if r['won']) / len(recent_p1)
            volatility_p1 = (sum((mg - recent_margin_p1) ** 2 for mg in margins_p1) / len(margins_p1)) ** 0.5
        else:
            recent_margin_p1 = 0.0
            recent_winrate_p1 = 0.5
            volatility_p1 = 2.0
        if recent_p2:
            margins_p2 = [r['margin'] for r in recent_p2]
            recent_margin_p2 = sum(margins_p2) / len(margins_p2)
            recent_winrate_p2 = sum(1 for r in recent_p2 if r['won']) / len(recent_p2)
            volatility_p2 = (sum((mg - recent_margin_p2) ** 2 for mg in margins_p2) / len(margins_p2)) ** 0.5
        else:
            recent_margin_p2 = 0.0
            recent_winrate_p2 = 0.5
            volatility_p2 = 2.0

        # Razlike u nedavnim marginama i winrate‑u
        recent_margin_diff = recent_margin_p1 - recent_margin_p2
        recent_winrate_diff = recent_winrate_p1 - recent_winrate_p2
        volatility_diff = volatility_p1 - volatility_p2

        # Broj mečeva danas (do sada)
        matches_today_p1 = day_counter[(p1, m['date'])]
        matches_today_p2 = day_counter[(p2, m['date'])]
        matches_today_avg = (matches_today_p1 + matches_today_p2) / 2.0

        # Povijesni over 5.5 postotak (prije ovog meča)
        total_p1 = player_total_count[p1]
        total_p2 = player_total_count[p2]
        hist_over_p1 = (player_over_count[p1] / total_p1) if total_p1 > 0 else 0.5
        hist_over_p2 = (player_over_count[p2] / total_p2) if total_p2 > 0 else 0.5
        hist_over_diff = hist_over_p1 - hist_over_p2

        # H2H over 5.5 postotak (međusobni susreti prije ovog)
        h2h_matches = [x for x in h2h_history(p1, p2, matches) if x['date'] < m['date'] or (x['date'] == m['date'] and x.get('_meta', {}).get('time', '') < m.get('_meta', {}).get('time', ''))]
        h2h_total = len(h2h_matches)
        if h2h_total >= 3:
            h2h_over = sum(1 for x in h2h_matches if x['p1legs'] + x['p2legs'] >= 6)
            h2h_over_pct = h2h_over / h2h_total
        else:
            h2h_over_pct = 0.5

        # Razlika u prosječnim 180‑icama
        avg_180_diff = avg_180.get(p1, 0.0) - avg_180.get(p2, 0.0)

        # Nedavne 180‑ice
        recent180_p1 = recent_180[p1][-5:]
        recent180_p2 = recent_180[p2][-5:]
        recent_180s_p1 = sum(recent180_p1) / len(recent180_p1) if recent180_p1 else avg_180.get(p1, 0.0)
        recent_180s_p2 = sum(recent180_p2) / len(recent180_p2) if recent180_p2 else avg_180.get(p2, 0.0)
        recent_180s_diff = recent_180s_p1 - recent_180s_p2

        # Spremi redak
        rows.append([
            elo_diff, avg_margin_p1 - avg_margin_p2, recent_margin_diff, recent_winrate_diff,
            volatility_diff, matches_today_avg, hist_over_diff, h2h_over_pct,
            avg_180_diff, recent_180s_diff, target
        ])

        # Ažuriraj stanja nakon meča
        day_counter[(p1, m['date'])] += 1
        day_counter[(p2, m['date'])] += 1
        # Nedavni rezultati i 180‑ice
        for pid, legs_own, legs_opp, s180 in [(p1, legs1, legs2, m.get('p1_180s', 0)),
                                               (p2, legs2, legs1, m.get('p2_180s', 0))]:
            won = 1 if legs_own > legs_opp else 0
            margin = legs_own - legs_opp
            recent_results[pid].append({'won': won, 'margin': margin})
            recent_180[pid].append(s180)
            # Ažuriraj over/under brojače
            player_total_count[pid] += 1
            if total_legs >= 6:
                player_over_count[pid] += 1

    df = pd.DataFrame(rows, columns=feature_cols + ['target'])
    X = df[feature_cols]
    y = df['target']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
    model.fit(X_scaled, y)
    return model, scaler 

def predict_over_under(model, scaler, p1_id, p2_id, elo_ratings, all_matches):
    """Vraća ML vjerojatnost over 5.5 za dati par."""
    # Izračunaj značajke koristeći sve dostupne mečeve (bez curenja, jer su svi prošli)
    player_matches = defaultdict(list)
    for m in all_matches:
        player_matches[m['p1id']].append(m)
        player_matches[m['p2id']].append(m)

    avg_180 = compute_player_180_avg(all_matches)

    # Prosječne margine
    def avg_margin(pid):
        pms = player_matches[pid]
        if not pms:
            return 0.0
        won = sum(x['p1legs'] if x['p1id'] == pid else x['p2legs'] for x in pms)
        lost = sum(x['p2legs'] if x['p1id'] == pid else x['p1legs'] for x in pms)
        return (won - lost) / len(pms)

    avg_margin_p1 = avg_margin(p1_id)
    avg_margin_p2 = avg_margin(p2_id)

    # Nedavna forma (posljednjih 5 mečeva)
    def recent_stats(pid):
        pms = sorted(player_matches[pid], key=lambda x: (x['date'], x.get('_meta', {}).get('time', '')), reverse=True)
        last5 = pms[:5]
        margins = []
        wins = []
        vols = []
        s180s = []
        for m in last5:
            if m['p1id'] == pid:
                mg = m['p1legs'] - m['p2legs']
                w = 1 if m['p1legs'] > m['p2legs'] else 0
                s180 = m.get('p1_180s', 0)
            else:
                mg = m['p2legs'] - m['p1legs']
                w = 1 if m['p2legs'] > m['p1legs'] else 0
                s180 = m.get('p2_180s', 0)
            margins.append(mg)
            wins.append(w)
            s180s.append(s180)
        if margins:
            avg_mg = sum(margins) / len(margins)
            winrate = sum(wins) / len(wins)
            vol = (sum((m - avg_mg) ** 2 for m in margins) / len(margins)) ** 0.5
            recent_180 = sum(s180s) / len(s180s) if s180s else avg_180.get(pid, 0.0)
        else:
            avg_mg = 0.0
            winrate = 0.5
            vol = 2.0
            recent_180 = avg_180.get(pid, 0.0)
        return avg_mg, winrate, vol, recent_180

    recent_margin_p1, recent_winrate_p1, vol_p1, recent_180_p1 = recent_stats(p1_id)
    recent_margin_p2, recent_winrate_p2, vol_p2, recent_180_p2 = recent_stats(p2_id)

    # Broj mečeva danas (pretpostavljamo da nema budućih – gledamo današnji datum)
    today_str = date.today().isoformat()
    matches_today_p1 = sum(1 for m in all_matches if m['date'] == today_str and (m['p1id'] == p1_id or m['p2id'] == p1_id))
    matches_today_p2 = sum(1 for m in all_matches if m['date'] == today_str and (m['p1id'] == p2_id or m['p2id'] == p2_id))
    matches_today_avg = (matches_today_p1 + matches_today_p2) / 2.0

    # Povijesni over/under postotak
    def hist_over(pid):
        pms = player_matches[pid]
        if not pms:
            return 0.5
        return sum(1 for m in pms if m['p1legs'] + m['p2legs'] >= 6) / len(pms)
    hist_over_p1 = hist_over(p1_id)
    hist_over_p2 = hist_over(p2_id)

    # H2H over postotak
    h2h_matches = h2h_history(p1_id, p2_id, all_matches)
    h2h_total = len(h2h_matches)
    if h2h_total >= 3:
        h2h_over = sum(1 for m in h2h_matches if m['p1legs'] + m['p2legs'] >= 6)
        h2h_over_pct = h2h_over / h2h_total
    else:
        h2h_over_pct = 0.5

    # Sastavi feature vektor
    features = [
        elo_ratings[p1_id] - elo_ratings[p2_id],
        avg_margin_p1 - avg_margin_p2,
        recent_margin_p1 - recent_margin_p2,
        recent_winrate_p1 - recent_winrate_p2,
        vol_p1 - vol_p2,
        matches_today_avg,
        hist_over_p1 - hist_over_p2,
        h2h_over_pct,
        avg_180.get(p1_id, 0.0) - avg_180.get(p2_id, 0.0),
        recent_180_p1 - recent_180_p2
    ]
    X = pd.DataFrame([features], columns=[
        'elo_diff', 'avg_margin_diff', 'recent_margin_diff', 'recent_winrate_diff',
        'volatility_diff', 'matches_today_avg', 'hist_over_diff', 'h2h_over_pct',
        'avg_180_diff', 'recent_180s_diff'
    ])
    X_scaled = scaler.transform(X)
    prob = model.predict_proba(X_scaled)[0, 1]
    return prob

# ---------- Expert Tips ----------
def generate_expert_tip_supersport_v2(p1_id, p2_id, p1_name, p2_name, matches, handicaps, p_leg):
    if p_leg > 0.5:
        favorite_id, underdog_id = p1_id, p2_id
        fav_name, dog_name = p1_name, p2_name
        underdog_is_home = False
    elif p_leg < 0.5:
        favorite_id, underdog_id = p2_id, p1_id
        fav_name, dog_name = p2_name, p1_name
        underdog_is_home = True
    else:
        return "Match is perfectly balanced. No clear underdog for an expert tip."
    underdog_matches = [m for m in matches if m['p1id'] == underdog_id or m['p2id'] == underdog_id]
    total = len(underdog_matches)
    if total < 3:
        return "Not enough matches for the underdog in the database."
    positive_handicaps = [1.5, 2.5]
    overall_cover = {}
    for ph in positive_handicaps:
        cnt = 0
        for m in underdog_matches:
            if m['p1id'] == underdog_id:
                legs_dog = m['p1legs']
                legs_opp = m['p2legs']
            else:
                legs_dog = m['p2legs']
                legs_opp = m['p1legs']
            if legs_dog + ph > legs_opp:
                cnt += 1
        overall_cover[ph] = (cnt / total * 100, cnt, total)
    h2h_matches = h2h_history(underdog_id, favorite_id, matches)
    h2h_total = len(h2h_matches)
    h2h_cover = {}
    if h2h_total >= 3:
        for ph in positive_handicaps:
            cnt = 0
            for m in h2h_matches:
                if m['p1id'] == underdog_id:
                    legs_dog = m['p1legs']
                    legs_opp = m['p2legs']
                else:
                    legs_dog = m['p2legs']
                    legs_opp = m['p1legs']
                if legs_dog + ph > legs_opp:
                    cnt += 1
            h2h_cover[ph] = (cnt / h2h_total * 100, cnt, h2h_total)
    best_h = None
    best_score = -1
    best_details = ""
    for ph in positive_handicaps:
        overall_pct, cnt_overall, total_overall = overall_cover[ph]
        if overall_pct >= 85:
            continue
        score = overall_pct * 0.7
        h2h_str = ""
        if h2h_cover:
            h2h_pct, cnt_h2h, total_h2h = h2h_cover[ph]
            score += h2h_pct * 0.3
            h2h_str = f"\n- H2H pokriće: {h2h_pct:.1f}% ({cnt_h2h}/{total_h2h})"
        else:
            score += overall_pct * 0.3
        if score > best_score:
            best_score = score
            best_h = ph
            best_details = (
                f"**{dog_name} +{ph}**\n"
                f"- Ukupno pokriće: {overall_pct:.1f}% ({cnt_overall}/{total_overall})"
                f"{h2h_str}\n"
                f"- Kombinirana ocjena: {best_score:.1f}%"
            )
    if best_h is None:
        return f"Svi pozitivni hendikepi za autsajdera **{dog_name}** imaju izrazito visoko pokriće (>85%). Nema value opcije s većim koeficijentom."
    if underdog_is_home:
        market = f"Tip 1 (domaćin), domaćin +{best_h}"
        tip_side = "1"
    else:
        market = f"Tip 2 (gost), domaćin -{best_h}"
        tip_side = "2"
    return f"📈 {best_details}\n💡 Preporuka: **{market}** – kladite se na {tip_side}."

def generate_expert_tip_rainbet_combined(selected_id, opponent_id, selected_name, opponent_name, matches, handicaps):
    w_overall = 0.7
    w_h2h = 0.3
    h2h_matches = h2h_history(selected_id, opponent_id, matches)
    h2h_total = len(h2h_matches)
    players = [(selected_id, selected_name), (opponent_id, opponent_name)]
    results = []
    for pid, pname in players:
        player_matches = [m for m in matches if m['p1id'] == pid or m['p2id'] == pid]
        total = len(player_matches)
        if total == 0:
            continue
        overall_cover = {}
        for h in handicaps:
            cnt = 0
            for m in player_matches:
                if m['p1id'] == pid:
                    legs_sel = m['p1legs']
                    legs_opp = m['p2legs']
                else:
                    legs_sel = m['p2legs']
                    legs_opp = m['p1legs']
                if legs_sel + h > legs_opp:
                    cnt += 1
            overall_cover[h] = (cnt / total * 100, cnt, total)
        h2h_cover = {}
        if h2h_total >= 3:
            for h in handicaps:
                cnt = 0
                for m in h2h_matches:
                    if m['p1id'] == pid:
                        legs_sel = m['p1legs']
                        legs_opp = m['p2legs']
                    else:
                        legs_sel = m['p2legs']
                        legs_opp = m['p1legs']
                    if legs_sel + h > legs_opp:
                        cnt += 1
                h2h_cover[h] = (cnt / h2h_total * 100, cnt, h2h_total)
        best_h = None
        best_score = -1
        for h in handicaps:
            overall_pct, _, _ = overall_cover[h]
            if overall_pct >= 85:
                continue
            score = overall_pct * w_overall
            if h2h_cover:
                h2h_pct, _, _ = h2h_cover[h]
                score += h2h_pct * w_h2h
            else:
                score += overall_pct * w_h2h
            if score > best_score:
                best_score = score
                best_h = h
        if best_h is None:
            best_h = max(overall_cover, key=lambda h: overall_cover[h][0])
            overall_pct, cnt_overall, total_overall = overall_cover[best_h]
            detail = (
                f"**{pname} {best_h:+.1f}**\n"
                f"- Ukupno pokriće: {overall_pct:.1f}% ({cnt_overall}/{total_overall})\n"
                f"- ⚠️ Siguran tip, nizak koeficijent.\n"
                f"- Kombinirana ocjena: {overall_pct*0.7:.1f}%"
            )
            results.append((pname, best_h, overall_pct, None, detail, overall_pct*0.7))
            continue
        overall_pct, cnt_overall, total_overall = overall_cover[best_h]
        detail = f"**{pname} {best_h:+.1f}**\n"
        detail += f"- Ukupno pokriće: {overall_pct:.1f}% ({cnt_overall}/{total_overall})\n"
        if h2h_cover:
            h2h_pct, cnt_h2h, total_h2h = h2h_cover[best_h]
            detail += f"- H2H pokriće: {h2h_pct:.1f}% ({cnt_h2h}/{total_h2h})\n"
        detail += f"- Kombinirana ocjena: {best_score:.1f}%\n"
        results.append((pname, best_h, overall_pct, h2h_cover.get(best_h, None), detail, best_score))
    if not results:
        return "Nema dovoljno podataka za bilo kojeg igrača."
    results.sort(key=lambda x: x[5], reverse=True)
    best = results[0]
    second = results[1] if len(results) > 1 and results[1][0] != best[0] else None
    tip = f"📈 **Najbolja vrijednost:** {best[4]}"
    if second:
        tip += f"\n🔄 **Alternativa (drugi igrač):**\n{second[4]}"
    return tip

# ---------- 180s Bounce ----------
def analyze_180_bounce(matches, players_dict):
    player_matches = defaultdict(list)
    for m in matches:
        player_matches[m['p1id']].append((m['date'], m.get('time', ''), m['p1_180s']))
        player_matches[m['p2id']].append((m['date'], m.get('time', ''), m['p2_180s']))
    eligible = {}
    for pid, lst in player_matches.items():
        total_180 = sum(x[2] for x in lst)
        avg = total_180 / len(lst)
        if avg > 1.0:
            eligible[pid] = avg
    results, total, success = [], 0, 0
    for pid in eligible:
        name = players_dict.get(pid, str(pid))
        sorted_m = sorted(player_matches[pid], key=lambda x: (x[0], x[1]))
        day_groups = defaultdict(list)
        for idx, (d, t, p180) in enumerate(sorted_m):
            day_groups[d].append((idx, t, p180))
        for day, dm in day_groups.items():
            if len(dm) < 2:
                continue
            for i in range(len(dm)-1):
                if dm[i][2] == 0:
                    total += 1
                    passed = dm[i+1][2] >= 1
                    if passed:
                        success += 1
                    results.append((name, day, i+1, dm[i+1][2], passed))
    rate = (success / total * 100) if total > 0 else 0
    return results, total, success, rate

def player_180_bounce(player_id, matches):
    player_data = []
    for m in matches:
        if m['p1id'] == player_id:
            player_data.append((m['date'], m.get('time', ''), m['p1_180s']))
        elif m['p2id'] == player_id:
            player_data.append((m['date'], m.get('time', ''), m['p2_180s']))
    if not player_data:
        return 0.0, 0.0, 0, 0
    total_180 = sum(x[2] for x in player_data)
    avg = total_180 / len(player_data)
    sorted_data = sorted(player_data, key=lambda x: (x[0], x[1]))
    day_groups = defaultdict(list)
    for idx, (d, t, p180) in enumerate(sorted_data):
        day_groups[d].append((idx, t, p180))
    total_cases, success_cases = 0, 0
    for day, dm in day_groups.items():
        if len(dm) < 2:
            continue
        for i in range(len(dm)-1):
            if dm[i][2] == 0:
                total_cases += 1
                if dm[i+1][2] >= 1:
                    success_cases += 1
    rate = (success_cases / total_cases * 100) if total_cases > 0 else 0.0
    return avg, rate, total_cases, success_cases

# ---------- Bounce nakon 4:0 (proširena) ----------
def analyze_bounce_after_4_0(matches):
    player_matches = defaultdict(list)
    for m in matches:
        time_str = m.get('_meta', {}).get('time', '')
        player_matches[m['p1id']].append((m['date'], time_str, m, 'p1'))
        player_matches[m['p2id']].append((m['date'], time_str, m, 'p2'))

    overall_total = 0
    overall_under = 0
    per_player = defaultdict(lambda: {'total': 0, 'under': 0})
    per_player_opponent = defaultdict(lambda: defaultdict(lambda: {'total': 0, 'under': 0}))

    for pid, lst in player_matches.items():
        sorted_matches = sorted(lst, key=lambda x: (x[0], x[1]))
        day_groups = defaultdict(list)
        for date, time, m, role in sorted_matches:
            day_groups[date].append((time, m, role))

        for day, day_list in day_groups.items():
            day_list.sort(key=lambda x: x[0])
            n = len(day_list)
            for i in range(n - 1):
                time, m, role = day_list[i]
                # Provjeri poraz 0:4
                if role == 'p1' and m['p1legs'] == 0 and m['p2legs'] == 4:
                    next_m = day_list[i+1][1]
                    total_legs = next_m['p1legs'] + next_m['p2legs']
                    under = 1 if total_legs < 6 else 0
                    overall_total += 1
                    per_player[pid]['total'] += 1
                    if under:
                        overall_under += 1
                        per_player[pid]['under'] += 1
                    # Protivnik u sljedećem meču
                    if next_m['p1id'] == pid:
                        opp_id = next_m['p2id']
                    else:
                        opp_id = next_m['p1id']
                    per_player_opponent[pid][opp_id]['total'] += 1
                    if under:
                        per_player_opponent[pid][opp_id]['under'] += 1
                elif role == 'p2' and m['p2legs'] == 0 and m['p1legs'] == 4:
                    next_m = day_list[i+1][1]
                    total_legs = next_m['p1legs'] + next_m['p2legs']
                    under = 1 if total_legs < 6 else 0
                    overall_total += 1
                    per_player[pid]['total'] += 1
                    if under:
                        overall_under += 1
                        per_player[pid]['under'] += 1
                    if next_m['p1id'] == pid:
                        opp_id = next_m['p2id']
                    else:
                        opp_id = next_m['p1id']
                    per_player_opponent[pid][opp_id]['total'] += 1
                    if under:
                        per_player_opponent[pid][opp_id]['under'] += 1

    overall_rate = (overall_under / overall_total * 100) if overall_total > 0 else 0.0
    return overall_total, overall_under, overall_rate, per_player, per_player_opponent

# ---------- Gap Bucket Over 5.5 ----------
def compute_leg_winrate_gap(matches):
    winrates = compute_leg_winrates(matches)
    return winrates

def build_gap_bucket_over_table(matches):
    winrates = compute_leg_winrates(matches)
    buckets = defaultdict(lambda: {'total': 0, 'over': 0})
    for m in matches:
        p1, p2 = m['p1id'], m['p2id']
        if p1 not in winrates or p2 not in winrates:
            continue
        gap = abs(winrates[p1] - winrates[p2]) * 100
        if gap < 5:
            bucket = "0–5%"
        elif gap < 10:
            bucket = "5–10%"
        elif gap < 15:
            bucket = "10–15%"
        elif gap < 20:
            bucket = "15–20%"
        else:
            bucket = "20%+"
        total_legs = m['p1legs'] + m['p2legs']
        buckets[bucket]['total'] += 1
        if total_legs >= 6:
            buckets[bucket]['over'] += 1
    table = {}
    for b, v in buckets.items():
        if v['total'] > 0:
            table[b] = {
                'total': v['total'],
                'over': v['over'],
                'pct': v['over'] / v['total'] * 100
            }
    return table

GAP_OVER_TABLE = None

# ---------- Under 5.5: Gap + hladnoća autsajdera ----------
@st.cache_resource
def build_under55_regime_table(matches):
    """Leak-free backtest: under 5.5 stopa po režimu A/B. Min 20 prethodnih mečeva."""
    MIN = 20
    legw = defaultdict(lambda: [0, 0])      # [legs_won, legs_played]
    nmatch = defaultdict(int)
    roll = defaultdict(lambda: deque(maxlen=3))
    sorted_m = sorted(matches, key=lambda m: (m['date'], m.get('_meta', {}).get('time', '')))
    reg = {'A': [0, 0], 'B': [0, 0], 'BASE15': [0, 0]}   # [n, under]
    avg = lambda dq: sum(dq) / len(dq) if dq else None
    for m in sorted_m:
        p1, p2 = m['p1id'], m['p2id']
        lw1, lw2 = legw[p1], legw[p2]
        if (nmatch[p1] >= MIN and nmatch[p2] >= MIN and lw1[1] > 0 and lw2[1] > 0
                and len(roll[p1]) == 3 and len(roll[p2]) == 3):
            e1, e2 = lw1[0] / lw1[1], lw2[0] / lw2[1]
            gap = abs(e1 - e2)
            und = p2 if e1 >= e2 else p1
            ur = avg(roll[und])
            u = 1 if (m['p1legs'] + m['p2legs']) <= 5 else 0
            if gap >= 0.15:
                reg['BASE15'][0] += 1; reg['BASE15'][1] += u
                if ur is not None and ur <= 0.35:
                    reg['A'][0] += 1; reg['A'][1] += u
            if gap >= 0.20 and ur is not None and ur <= 0.25:
                reg['B'][0] += 1; reg['B'][1] += u
        t = m['p1legs'] + m['p2legs']
        legw[p1] = [lw1[0] + m['p1legs'], lw1[1] + t]
        legw[p2] = [lw2[0] + m['p2legs'], lw2[1] + t]
        nmatch[p1] += 1; nmatch[p2] += 1
        if t > 0:
            roll[p1].append(m['p1legs'] / t); roll[p2].append(m['p2legs'] / t)
    out = {}
    for k, (n, u) in reg.items():
        out[k] = {'n': n, 'under': u, 'pct': (u / n * 100 if n else 0.0),
                  'be': (round(n / u, 3) if u else None)}
    return out

def compute_roll3(matches):
    """Trenutni roll3 svakog igrača = udio osvojenih legova u zadnja 3 meča."""
    sorted_m = sorted(matches, key=lambda m: (m['date'], m.get('_meta', {}).get('time', '')))
    last3 = defaultdict(lambda: deque(maxlen=3))
    for m in sorted_m:
        t = m['p1legs'] + m['p2legs']
        if t > 0:
            last3[m['p1id']].append(m['p1legs'] / t)
            last3[m['p2id']].append(m['p2legs'] / t)
    return {pid: (sum(dq) / len(dq) if dq else None) for pid, dq in last3.items()}

def classify_under55(p1_id, p2_id, winrates, roll3, regime_table):
    """Klasifikuje par u režim A/B/None i vraća empirijsku under stopu + break-even."""
    e1, e2 = winrates.get(p1_id), winrates.get(p2_id)
    if e1 is None or e2 is None:
        return None
    gap = abs(e1 - e2) * 100.0
    fav_id, und_id = (p1_id, p2_id) if e1 >= e2 else (p2_id, p1_id)
    ur = roll3.get(und_id)
    regime = None
    if gap >= 20 and ur is not None and ur <= 0.25:
        regime = 'B'                       # jača ivica ima prednost
    elif gap >= 15 and ur is not None and ur <= 0.35:
        regime = 'A'
    info = regime_table.get(regime) if regime else None
    return {'gap': gap, 'fav_id': fav_id, 'und_id': und_id, 'und_roll3': ur,
            'regime': regime,
            'under_pct': (info['pct'] if info else None),
            'be_odds': (info['be'] if info else None),
            'n': (info['n'] if info else None)} 

# ---------- Profil rezultata igrača (kratki/dugi mečevi) ----------
def gap_bucket_name(gap_pct):
    if gap_pct < 5: return '0-5%'
    if gap_pct < 10: return '5-10%'
    if gap_pct < 15: return '10-15%'
    if gap_pct < 20: return '15-20%'
    if gap_pct < 25: return '20-25%'
    return '25%+'

def player_score_profile(player_id, matches, leg_wr, bucket=None, last_n=None):
    """Profil po margini (4:0/4:1/4:2/4:3), samo bo7. Vraća (Counter po legovima gubitnika, n)."""
    pm = sorted([m for m in matches if m['p1id'] == player_id or m['p2id'] == player_id],
                key=lambda x: (x['date'], x.get('_meta', {}).get('time', '')))
    if last_n:
        pm = pm[-last_n:]
    cnt = Counter(); n = 0
    for m in pm:
        if max(m['p1legs'], m['p2legs']) != 4:
            continue
        if m['p1id'] == player_id:
            my, opp, oid = m['p1legs'], m['p2legs'], m['p2id']
        else:
            my, opp, oid = m['p2legs'], m['p1legs'], m['p1id']
        if bucket is not None:
            g = abs(leg_wr.get(player_id, 0.5) - leg_wr.get(oid, 0.5)) * 100
            if gap_bucket_name(g) != bucket:
                continue
        cnt[min(my, opp)] += 1; n += 1
    return cnt, n

# ===================== APP =====================
st.set_page_config(page_title="Darts Handicap Tool", layout="wide")
init_session_state()

all_matches = st.session_state.matches + st.session_state.new_matches
if GAP_OVER_TABLE is None:
    GAP_OVER_TABLE = build_gap_bucket_over_table(all_matches)
elo_ratings = compute_elo_ratings(all_matches)
stats_180 = compute_player_180_stats(all_matches)

advanced_models = train_advanced_models(all_matches, elo_ratings)
over_under_model = train_over_under_model(all_matches, elo_ratings)

UNDER55_REGIME = build_under55_regime_table(all_matches)
ROLL3 = compute_roll3(all_matches)
LEG_WR = compute_leg_winrates(all_matches)

all_vols = []
for pid in st.session_state.players:
    p_matches = [m for m in all_matches if m['p1id'] == pid or m['p2id'] == pid]
    sorted_pm = sorted(p_matches, key=lambda x: (x['date'], x.get('_meta', {}).get('time', '')), reverse=True)
    last5 = sorted_pm[:5]
    if len(last5) >= 2:
        margins = [(m['p1legs'] if m['p1id'] == pid else m['p2legs']) - 
                   (m['p2legs'] if m['p1id'] == pid else m['p1legs']) for m in last5]
        avg_m = sum(margins) / len(margins)
        vol = (sum((m - avg_m) ** 2 for m in margins) / len(margins)) ** 0.5
        all_vols.append(vol)
VOLATILITY_THRESHOLD = pd.Series(all_vols).quantile(0.9) if all_vols else 3.0

VALUE_THRESHOLD = 0.02

st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", [
    "🏠 Home (handicap on home)",
    "👤 Handicap on player (Rainbet)",
    "🎯 180s Prediction",
    "➕ Add match",
    "📤 Upload dnevnog JSON-a",
    "🔍 Check ticket (SuperSport)",
    "🔍 Check ticket (Rainbet)",
    "🔄 180s Bounce Back (opći)",
    "📊 180s Player Bounce",
    "🔢 Over 5.5 Legs",
    "🎯 Under 5.5 (Gap + Forma)",
    "🔄 Bounce nakon 4:0",
    "🧹 Očisti duplikate"
])

# ========== HOME (home handicap) ==========
if page == "🏠 Home (handicap on home)":
    st.title("🎯 Darts Handicap Predictor – SuperSport rules")
    st.markdown("Handicap is **always added to the home player (Player 1)**. You bet on Tip 1 or Tip 2.")

    player_names = list(st.session_state.players.values())
    player_ids = list(st.session_state.players.keys())

    col1, col2 = st.columns(2)
    with col1:
        p1_name = st.selectbox("👤 Home (Player 1)", player_names, index=0)
    with col2:
        p2_name = st.selectbox("👤 Away (Player 2)", player_names, index=1)

    p1_id = player_ids[player_names.index(p1_name)]
    p2_id = player_ids[player_names.index(p2_name)]

    if p1_id == p2_id:
        st.warning("Choose two different players.")
        st.stop()

    p_leg = leg_prob(p1_id, p2_id, elo_ratings)
    dist = match_distribution(p_leg)
    best_side, best_h_val, best_prob = gold_tip(p_leg)

    st.header("💡 Gold Tip – best recommendation")
    st.success(f"**We recommend betting Tip {best_side}** with handicap **{best_h_val:+.1f}** on home.")
    st.metric("Pass probability", f"{best_prob:.2%}")

    with st.expander("📈 Expert Tips"):
        tip = generate_expert_tip_supersport_v2(p1_id, p2_id, p1_name, p2_name, all_matches, AVAILABLE_HANDICAPS, p_leg)
        st.markdown(tip)

    st.header("📊 Detailed statistics")
    e1 = elo_ratings.get(p1_id, 1500)
    e2 = elo_ratings.get(p2_id, 1500)
    st.metric(f"{p1_name} Elo", f"{e1:.0f}")
    st.metric(f"{p2_name} Elo", f"{e2:.0f}")
    st.metric("Probability home wins a leg", f"{p_leg:.3f}")

    st.subheader("Final score distribution")
    dist_sorted = sorted(dist.items(), key=lambda x: (-x[0][0], x[0][1]))
    dist_table = [{"Score": f"{a}:{b}", "Probability": f"{prob:.2%}"} for (a,b), prob in dist_sorted]
    st.dataframe(dist_table, use_container_width=True)

    st.subheader("✨ Probability for selected handicap")
    selected_h_str = st.selectbox("Select handicap", ["+1.5", "-1.5", "+2.5", "-2.5"], index=0)
    selected_h_val = float(selected_h_str.replace("+", ""))
    tip1_prob = handicap_prob(p_leg, selected_h_val)
    col_p1, col_p2 = st.columns(2)
    col_p1.metric("Tip 1 (home wins)", f"{tip1_prob:.2%}")
    col_p2.metric("Tip 2 (away wins)", f"{1 - tip1_prob:.2%}")

    with st.expander("💎 Value kalkulator & točan rezultat"):
        st.markdown("### Usporedba s kladioničarskim koeficijentom")
        home_matches = [m for m in all_matches if m['p1id'] == p1_id or m['p2id'] == p1_id]
        total_home = len(home_matches)
        hist_cover_home = {}
        for h in AVAILABLE_HANDICAPS:
            cnt = 0
            for m in home_matches:
                if m['p1id'] == p1_id:
                    legs_sel = m['p1legs']
                    legs_opp = m['p2legs']
                else:
                    legs_sel = m['p2legs']
                    legs_opp = m['p1legs']
                if legs_sel + h > legs_opp:
                    cnt += 1
            hist_cover_home[h] = cnt / total_home * 100 if total_home > 0 else 0.0
        away_matches = [m for m in all_matches if m['p1id'] == p2_id or m['p2id'] == p2_id]
        total_away = len(away_matches)
        hist_cover_away = {}
        for h in AVAILABLE_HANDICAPS:
            cnt = 0
            for m in away_matches:
                if m['p1id'] == p2_id:
                    legs_sel = m['p1legs']
                    legs_opp = m['p2legs']
                else:
                    legs_sel = m['p2legs']
                    legs_opp = m['p1legs']
                if legs_sel + h > legs_opp:
                    cnt += 1
            hist_cover_away[h] = cnt / total_away * 100 if total_away > 0 else 0.0
        h2h_matches = h2h_history(p1_id, p2_id, all_matches)
        h2h_total = len(h2h_matches)
        h2h_cover_home = {}
        h2h_cover_away = {}
        if h2h_total >= 3:
            for h in AVAILABLE_HANDICAPS:
                cnt_h = 0
                cnt_a = 0
                for m in h2h_matches:
                    if m['p1id'] == p1_id:
                        home_legs = m['p1legs']
                        away_legs = m['p2legs']
                    else:
                        home_legs = m['p2legs']
                        away_legs = m['p1legs']
                    if home_legs + h > away_legs:
                        cnt_h += 1
                    if away_legs - h > home_legs:
                        cnt_a += 1
                h2h_cover_home[h] = cnt_h / h2h_total * 100
                h2h_cover_away[h] = cnt_a / h2h_total * 100
        else:
            for h in AVAILABLE_HANDICAPS:
                h2h_cover_home[h] = None
                h2h_cover_away[h] = None

        col_odds1, col_odds2 = st.columns(2)
        with col_odds1:
            odds_tip1 = st.number_input("Koeficijent za Tip 1", 1.01, 20.0, 2.0, 0.05, key="odds_tip1")
        with col_odds2:
            odds_tip2 = st.number_input("Koeficijent za Tip 2", 1.01, 20.0, 2.0, 0.05, key="odds_tip2")

        if st.button("Izračunaj value za odabrani hendikep", key="calc_ss"):
            impl_tip1 = 1 / odds_tip1
            impl_tip2 = 1 / odds_tip2
            edge_model1 = tip1_prob - impl_tip1
            edge_model2 = (1 - tip1_prob) - impl_tip2

            hist_pct_home = hist_cover_home[selected_h_val]
            edge_hist_home = (hist_pct_home / 100) - impl_tip1

            ml_prob_home, vol_home = predict_handicap_advanced(advanced_models, p1_id, p2_id, selected_h_val, elo_ratings, all_matches)
            edge_ml_home = ml_prob_home - impl_tip1

            ml_prob_away, vol_away = predict_handicap_advanced(advanced_models, p2_id, p1_id, selected_h_val, elo_ratings, all_matches)
            edge_ml_away = ml_prob_away - impl_tip2

            st.write(f"**Tip 1 (domaćin)** – model: {tip1_prob:.2%}, implicirana: {impl_tip1:.2%}, edge model: {edge_model1:+.2%}")
            if total_home > 0:
                st.write(f"   Povijesno pokriće ({total_home} mečeva): {hist_pct_home:.1f}%, edge povijesni: {edge_hist_home:+.2%}")
            if h2h_total >= 3 and h2h_cover_home[selected_h_val] is not None:
                h2h_pct_home = h2h_cover_home[selected_h_val]
                edge_h2h_home = (h2h_pct_home / 100) - impl_tip1
                st.write(f"   H2H pokriće ({h2h_total} mečeva): {h2h_pct_home:.1f}%, edge H2H: {edge_h2h_home:+.2%}")
                if edge_h2h_home > VALUE_THRESHOLD:
                    st.success("H2H VALUE!")
            st.write(f"   ML model: {ml_prob_home:.2%}, edge ML: {edge_ml_home:+.2%}")
            if vol_home > VOLATILITY_THRESHOLD:
                st.warning(f"⚠️ Visoka volatilnost domaćina ({vol_home:.2f}) – povećan rizik.")
            else:
                st.info(f"Volatilnost domaćina: {vol_home:.2f}")
            if edge_model1 > VALUE_THRESHOLD or edge_hist_home > VALUE_THRESHOLD or edge_ml_home > VALUE_THRESHOLD:
                st.success("VALUE!")
            else:
                st.error("Nema valuea.")

            st.write(f"**Tip 2 (gost)** – model: {1-tip1_prob:.2%}, implicirana: {impl_tip2:.2%}, edge model: {edge_model2:+.2%}")
            if total_away > 0:
                hist_pct_away = hist_cover_away[selected_h_val]
                edge_hist_away = (hist_pct_away / 100) - impl_tip2
                st.write(f"   Povijesno pokriće gosta ({total_away} mečeva): {hist_pct_away:.1f}%, edge povijesni: {edge_hist_away:+.2%}")
            if h2h_total >= 3 and h2h_cover_away[selected_h_val] is not None:
                h2h_pct_away = h2h_cover_away[selected_h_val]
                edge_h2h_away = (h2h_pct_away / 100) - impl_tip2
                st.write(f"   H2H pokriće ({h2h_total} mečeva): {h2h_pct_away:.1f}%, edge H2H: {edge_h2h_away:+.2%}")
                if edge_h2h_away > VALUE_THRESHOLD:
                    st.success("H2H VALUE!")
            st.write(f"   ML model: {ml_prob_away:.2%}, edge ML: {edge_ml_away:+.2%}")
            if vol_away > VOLATILITY_THRESHOLD:
                st.warning(f"⚠️ Visoka volatilnost gosta ({vol_away:.2f}) – povećan rizik.")
            else:
                st.info(f"Volatilnost gosta: {vol_away:.2f}")
            if edge_model2 > VALUE_THRESHOLD or edge_ml_away > VALUE_THRESHOLD:
                st.success("VALUE!")
            else:
                st.error("Nema valuea.")

        st.markdown("---")
        st.markdown("### Fer koeficijenti za točan rezultat")
        correct_score_table = []
        for (a,b), prob in dist_sorted:
            fer = fer_koef(prob)
            correct_score_table.append({
                "Score": f"{a}:{b}",
                "Probability": f"{prob:.2%}",
                "Fair odds": f"{fer:.2f}" if fer else "-"
            })
        st.dataframe(correct_score_table, use_container_width=True)

    st.header("📜 Head-to-head (H2H)")
    h2h = h2h_history(p1_id, p2_id, all_matches)
    if not h2h:
        st.info("No recorded H2H matches.")
    else:
        h2h_table = []
        for m in h2h:
            if m['p1id'] == p1_id:
                dom_name, gost_name = p1_name, p2_name
                dom_legs, gost_legs = m['p1legs'], m['p2legs']
            else:
                dom_name, gost_name = p2_name, p1_name
                dom_legs, gost_legs = m['p2legs'], m['p1legs']
            pokrice_plus15 = "✔️ Tip 1" if (dom_legs + 1.5 > gost_legs) else "❌ Tip 2"
            h2h_table.append({
                "Date": m.get('date', '?'),
                "Home": dom_name,
                "Score": f"{dom_legs}:{gost_legs}",
                "+1.5 covered": pokrice_plus15
            })
        st.dataframe(h2h_table, use_container_width=True)

# ========== PLAYER HANDICAP (Rainbet) ==========
elif page == "👤 Handicap on player (Rainbet)":
    st.title("👤 Player Handicap – Rainbet rules")
    st.markdown(
        "Handicap is **directly tied to the selected player**, regardless of home/away. "
        "Example: `Leg handicap Trueman, Danny (-2.5)` means **Danny Trueman** gives up 2.5 legs, "
        "so he must win by at least 3 legs."
    )

    player_names = list(st.session_state.players.values())
    player_ids = list(st.session_state.players.keys())

    col1, col2 = st.columns(2)
    with col1:
        selected_name = st.selectbox("🎯 Player (handicap applies to)", player_names, index=0)
    with col2:
        opponent_name = st.selectbox("🥊 Opponent", player_names, index=1)

    selected_id = player_ids[player_names.index(selected_name)]
    opponent_id = player_ids[player_names.index(opponent_name)]

    if selected_id == opponent_id:
        st.warning("Choose two different players.")
        st.stop()

    p_leg = leg_prob(selected_id, opponent_id, elo_ratings)
    e_sel = elo_ratings.get(selected_id, 1500)
    e_opp = elo_ratings.get(opponent_id, 1500)

    st.metric(f"Elo {selected_name}", f"{e_sel:.0f}")
    st.metric(f"Elo {opponent_name}", f"{e_opp:.0f}")
    st.metric(f"Probability {selected_name} wins a leg", f"{p_leg:.3f}")

    best_h, best_prob = gold_tip_player(p_leg)

    st.header("💡 Gold Tip (safest handicap for this player)")
    st.success(f"We recommend **{selected_name} {best_h:+.1f}** (pass probability: {best_prob:.2%})")

    with st.expander("📈 Expert Tips"):
        tip = generate_expert_tip_rainbet_combined(selected_id, opponent_id, selected_name, opponent_name, all_matches, AVAILABLE_HANDICAPS)
        st.markdown(tip)

    st.subheader("Probabilities for all lines")
    lines = []
    for h in sorted(AVAILABLE_HANDICAPS):
        ph = handicap_prob(p_leg, h)
        lines.append({
            "Handicap": f"{selected_name} {h:+.1f}",
            "Pass probability": f"{ph:.2%}"
        })
    st.dataframe(lines, use_container_width=True)

    with st.expander("💎 Value calculator & correct score"):
        st.markdown("### Compare with bookmaker odds")
        player_matches = [m for m in all_matches if m['p1id'] == selected_id or m['p2id'] == selected_id]
        total_player = len(player_matches)
        hist_cover = {}
        if total_player > 0:
            for h in AVAILABLE_HANDICAPS:
                cnt = 0
                for m in player_matches:
                    if m['p1id'] == selected_id:
                        legs_sel = m['p1legs']
                        legs_opp = m['p2legs']
                    else:
                        legs_sel = m['p2legs']
                        legs_opp = m['p1legs']
                    if legs_sel + h > legs_opp:
                        cnt += 1
                hist_cover[h] = cnt / total_player * 100
        else:
            for h in AVAILABLE_HANDICAPS:
                hist_cover[h] = 0.0

        h2h_matches = h2h_history(selected_id, opponent_id, all_matches)
        h2h_total = len(h2h_matches)
        h2h_hist = {}
        if h2h_total >= 3:
            for h in AVAILABLE_HANDICAPS:
                cnt = 0
                for m in h2h_matches:
                    if m['p1id'] == selected_id:
                        legs_sel = m['p1legs']
                        legs_opp = m['p2legs']
                    else:
                        legs_sel = m['p2legs']
                        legs_opp = m['p1legs']
                    if legs_sel + h > legs_opp:
                        cnt += 1
                h2h_hist[h] = cnt / h2h_total * 100
        else:
            for h in AVAILABLE_HANDICAPS:
                h2h_hist[h] = None

        value_h = st.selectbox("Select handicap line for value check", 
                               [f"{selected_name} {h:+.1f}" for h in sorted(AVAILABLE_HANDICAPS)],
                               index=0)
        value_h_val = float(value_h.split()[-1])
        prob_line = handicap_prob(p_leg, value_h_val)
        hist_pct = hist_cover[value_h_val]

        ml_prob, vol = predict_handicap_advanced(advanced_models, selected_id, opponent_id, value_h_val, elo_ratings, all_matches)

        st.write(f"**Modelirana vjerojatnost:** {prob_line:.2%}")
        st.write(f"**Povijesno pokriće ({total_player} mečeva):** {hist_pct:.1f}%")
        if h2h_total >= 3 and h2h_hist[value_h_val] is not None:
            h2h_pct = h2h_hist[value_h_val]
            st.write(f"**H2H pokriće ({h2h_total} mečeva):** {h2h_pct:.1f}%")
        st.write(f"**ML model vjerojatnost:** {ml_prob:.2%}")
        if vol > VOLATILITY_THRESHOLD:
            st.warning(f"⚠️ Visoka volatilnost igrača ({vol:.2f}) – povećan rizik.")
        else:
            st.info(f"Volatilnost: {vol:.2f}")

        odds_line = st.number_input(f"Koeficijent za {value_h}", 1.01, 20.0, 2.0, 0.05, key="odds_player")
        if st.button("Izračunaj value"):
            impl_line = 1 / odds_line
            edge_model = prob_line - impl_line
            edge_hist = (hist_pct / 100) - impl_line
            edge_ml = ml_prob - impl_line

            st.write(f"**Implicirana vjerojatnost:** {impl_line:.2%}")
            st.write(f"**Edge (model):** {edge_model:+.2%} {'✅' if edge_model > 0 else '❌'}")
            st.write(f"**Edge (povijesni):** {edge_hist:+.2%} {'✅' if edge_hist > 0 else '❌'}")
            if h2h_total >= 3 and h2h_hist[value_h_val] is not None:
                edge_h2h = (h2h_hist[value_h_val] / 100) - impl_line
                st.write(f"**Edge (H2H):** {edge_h2h:+.2%} {'✅' if edge_h2h > 0 else '❌'}")
            st.write(f"**Edge (ML):** {edge_ml:+.2%} {'✅' if edge_ml > 0 else '❌'}")

            if edge_hist > VALUE_THRESHOLD or edge_model > VALUE_THRESHOLD or edge_ml > VALUE_THRESHOLD:
                st.success("VALUE pronađen!")
            else:
                st.warning("Nema značajnog valuea.")

        st.markdown("---")
        st.markdown("### Pregled svih linija za ovog igrača")
        overview = []
        for h in sorted(AVAILABLE_HANDICAPS):
            model_prob = handicap_prob(p_leg, h)
            hist_pct_h = hist_cover[h]
            fair_model = round(1/model_prob, 2) if model_prob > 0 else None
            fair_hist = round(100/hist_pct_h, 2) if hist_pct_h > 0 else None
            h2h_pct_str = f"{h2h_hist[h]:.1f}%" if (h2h_total >= 3 and h2h_hist[h] is not None) else "N/A"
            ml_prob_h, _ = predict_handicap_advanced(advanced_models, selected_id, opponent_id, h, elo_ratings, all_matches)
            overview.append({
                "Hendikep": f"{selected_name} {h:+.1f}",
                "Model vjerojatnost": f"{model_prob:.2%}",
                "Povijesno pokriće": f"{hist_pct_h:.1f}%",
                "H2H pokriće": h2h_pct_str,
                "ML vjerojatnost": f"{ml_prob_h:.2%}",
                "Fer koef. (model)": fair_model if fair_model else "-",
                "Fer koef. (povijest)": fair_hist if fair_hist else "-"
            })
        st.dataframe(overview, use_container_width=True)
        st.caption("Ako je povijesno pokriće veće od modela, to može ukazivati na podcijenjenost igrača.")

        st.markdown("---")
        st.markdown("### Fair odds for correct score")
        dist = match_distribution(p_leg)
        dist_sorted = sorted(dist.items(), key=lambda x: (-x[0][0], x[0][1]))
        correct_score_table = []
        for (a,b), prob in dist_sorted:
            fer = fer_koef(prob)
            correct_score_table.append({
                "Score": f"{a}:{b}",
                "Probability": f"{prob:.2%}",
                "Fair odds": f"{fer:.2f}" if fer else "-"
            })
        st.dataframe(correct_score_table, use_container_width=True)

    st.caption("E.g. -2.5 → player must win 4:0, 4:1; +1.5 → just not lose by more than 1 leg.")

# ========== 180s PREDICTION ==========
elif page == "🎯 180s Prediction":
    st.title("🎯 180s Prediction")
    st.markdown("Based on historical data, we calculate probabilities for individual and total 180s.")

    player_names = list(st.session_state.players.values())
    player_ids = list(st.session_state.players.keys())

    col1, col2 = st.columns(2)
    with col1:
        p1_name = st.selectbox("👤 Home (Player 1)", player_names, index=0, key="p180_1")
    with col2:
        p2_name = st.selectbox("👤 Away (Player 2)", player_names, index=1, key="p180_2")

    p1_id = player_ids[player_names.index(p1_name)]
    p2_id = player_ids[player_names.index(p2_name)]

    if p1_id == p2_id:
        st.warning("Choose two different players.")
        st.stop()

    dist1 = get_180_distribution(p1_id, stats_180)
    dist2 = get_180_distribution(p2_id, stats_180)

    avg1 = sum(k * v for k, v in dist1.items())
    avg2 = sum(k * v for k, v in dist2.items())

    p1_over = 1 - dist1.get(0, 0)
    p2_over = 1 - dist2.get(0, 0)

    total_over = 0.0
    for a, pa in dist1.items():
        for b, pb in dist2.items():
            if a + b >= 2:
                total_over += pa * pb

    st.header("📊 180s Statistics")
    colA, colB = st.columns(2)
    colA.metric(f"{p1_name} – avg 180s per match", f"{avg1:.2f}")
    colB.metric(f"{p2_name} – avg 180s per match", f"{avg2:.2f}")

    st.subheader("Probabilities for betting markets")
    colP1, colP2, colTot = st.columns(3)
    colP1.metric(f"Over 0.5 ({p1_name})", f"{p1_over:.1%}")
    colP2.metric(f"Over 0.5 ({p2_name})", f"{p2_over:.1%}")
    colTot.metric("Total over 1.5 (both)", f"{total_over:.1%}")

# ========== ADD MATCH ==========
elif page == "➕ Add match":
    st.title("➕ Add new match to database")
    st.markdown("Select existing players and enter score (legs).")

    players_list = list(st.session_state.players.values())
    ids_list = list(st.session_state.players.keys())

    col1, col2 = st.columns(2)
    with col1:
        p1_name = st.selectbox("Home (Player 1)", players_list, key="add_p1")
    with col2:
        p2_name = st.selectbox("Away (Player 2)", players_list, key="add_p2")

    col3, col4 = st.columns(2)
    with col3:
        p1legs = st.number_input("Home legs", 0, 4, 0, key="add_legs1")
    with col4:
        p2legs = st.number_input("Away legs", 0, 4, 0, key="add_legs2")

    with st.expander("Details (optional)"):
        match_date = st.date_input("Match date", value=date.today())
        series = st.text_input("Series", "Modus Super Series")
        week = st.text_input("Week", "")
        group = st.text_input("Group", "")
        time = st.text_input("Time", "")

    if st.button("Add match to session"):
        if p1_name == p2_name:
            st.error("Players must be different.")
        elif not ((p1legs == 4 and p2legs < 4) or (p2legs == 4 and p1legs < 4)):
            st.error("Invalid score. One player must have 4 legs, the other fewer.")
        else:
            p1_id = ids_list[players_list.index(p1_name)]
            p2_id = ids_list[players_list.index(p2_name)]
            new_match = {
                "id": max([m['id'] for m in all_matches] + [0]) + 1,
                "p1id": p1_id,
                "p2id": p2_id,
                "p1legs": p1legs,
                "p2legs": p2legs,
                "p1_180s": 0,
                "p2_180s": 0,
                "date": match_date.isoformat(),
                "_meta": {
                    "series": series,
                    "week": week,
                    "group": group,
                    "time": time
                }
            }

            duplicate, dup_match = is_duplicate(new_match, all_matches)
            similar, sim_match = similar_match_exists(new_match, all_matches)

            if duplicate:
                st.error("This match already exists in the database!")
                with st.expander("Show existing match"):
                    st.json(dup_match)
            else:
                if similar:
                    st.warning("Similar match with same players and date but different score/meta exists. Adding anyway.")
                st.session_state.new_matches.append(new_match)
                st.success("Match added to session! (Not yet saved to JSON)")

    if st.session_state.new_matches:
        st.subheader("Unsaved new matches")
        st.write(st.session_state.new_matches)
        if st.button("💾 Save all changes to JSON"):
            st.session_state.matches.extend(st.session_state.new_matches)
            save_to_json()
            st.success("Database updated!")
            st.rerun()

# ========== UPLOAD DAILY JSON ==========
elif page == "📤 Upload dnevnog JSON-a":
    st.title("📤 Upload dnevnog JSON-a (dijagnostički mod)")
    st.markdown("Učitaj JSON s mečevima – aplikacija će pokazati koji mečevi se dodaju, a koji preskaču i zašto.")

    uploaded_file = st.file_uploader("Odaberi JSON datoteku", type="json")
    if uploaded_file is not None:
        try:
            daily_data = json.load(uploaded_file)
        except Exception as e:
            st.error(f"Greška pri čitanju JSON-a: {e}")
            st.stop()

        current_matches = st.session_state.matches + st.session_state.new_matches
        existing_ids = {m['id'] for m in current_matches if 'id' in m}

        st.write(f"Učitano mečeva: {len(daily_data.get('matches', []))}")
        st.write(f"Mečeva trenutno u bazi: {len(current_matches)}")

        matches_to_add = []
        matches_skipped = []
        for m in daily_data.get('matches', []):
            mid = m.get('id')
            if mid is not None and mid in existing_ids:
                matches_skipped.append((m, "ID već postoji u bazi"))
            else:
                dup, existing = is_duplicate(m, current_matches)
                if dup:
                    matches_skipped.append((m, f"Duplikat prema sadržaju"))
                else:
                    matches_to_add.append(m)

        st.subheader("Rezultat analize")
        col_a, col_b = st.columns(2)
        col_a.metric("Dodat će se", len(matches_to_add))
        col_b.metric("Preskočeno", len(matches_skipped))

        if matches_skipped:
            with st.expander("📋 Pregled preskočenih mečeva"):
                for i, (m, reason) in enumerate(matches_skipped):
                    st.write(f"**#{i+1}** – {reason}")
                    st.json(m)
                    if 'ID već postoji' in reason:
                        existing_match = next((x for x in current_matches if x.get('id') == m.get('id')), None)
                        if existing_match:
                            st.caption("Postojeći meč s istim ID‑em:")
                            st.json(existing_match)
                    st.markdown("---")

        col_op1, col_op2 = st.columns(2)
        with col_op1:
            if st.button("✅ Dodaj samo nove mečeve (preporučeno)"):
                updated_players = st.session_state.players.copy()
                for p in daily_data.get('players', []):
                    pid = p['id']
                    if pid not in updated_players:
                        updated_players[pid] = p['name']
                updated_matches = current_matches.copy()
                updated_matches.extend(matches_to_add)
                st.session_state.players = updated_players
                st.session_state.matches = updated_matches
                st.session_state.new_matches = []
                save_to_json()
                st.success(f"Dodano {len(matches_to_add)} mečeva.")
                st.rerun()
        with col_op2:
            if st.button("⚡ Prisilno dodaj SVE (zanemari duplikate)"):
                updated_players = st.session_state.players.copy()
                for p in daily_data.get('players', []):
                    pid = p['id']
                    if pid not in updated_players:
                        updated_players[pid] = p['name']
                updated_matches = current_matches.copy()
                updated_matches.extend(daily_data.get('matches', []))
                st.session_state.players = updated_players
                st.session_state.matches = updated_matches
                st.session_state.new_matches = []
                save_to_json()
                st.success(f"Prisilno dodano {len(daily_data.get('matches', []))} mečeva.")
                st.rerun()

# ========== CHECK TICKET (SuperSport) ==========
elif page == "🔍 Check ticket (SuperSport)":
    st.title("🔍 Ticket check – SuperSport rules")
    st.markdown("Enter the played match and your tip (handicap is added to home).")

    player_names = list(st.session_state.players.values())
    player_ids = list(st.session_state.players.keys())

    col1, col2 = st.columns(2)
    with col1:
        p1_name = st.selectbox("Home (Player 1)", player_names, key="check_p1_ss")
    with col2:
        p2_name = st.selectbox("Away (Player 2)", player_names, key="check_p2_ss")

    col3, col4 = st.columns(2)
    with col3:
        actual_p1legs = st.number_input("Actual home legs", 0, 4, 0, key="real_home_ss")
    with col4:
        actual_p2legs = st.number_input("Actual away legs", 0, 4, 0, key="real_away_ss")

    col5, col6 = st.columns(2)
    with col5:
        played_h_str = st.selectbox("Selected handicap on ticket", ["+1.5", "-1.5", "+2.5", "-2.5"], key="played_h_ss")
    with col6:
        played_side = st.radio("Your tip", ["1 (home)", "2 (away)"], key="played_side_ss")

    if st.button("Check ticket (SuperSport)"):
        p1_id = player_ids[player_names.index(p1_name)]
        p2_id = player_ids[player_names.index(p2_name)]

        played_h_val = float(played_h_str.replace("+", ""))
        home_winner = handicap_winner(actual_p1legs, actual_p2legs, played_h_val)
        real_winner_str = "1 (home)" if home_winner else "2 (away)"
        ticket_pass = (real_winner_str == played_side)

        p_leg = leg_prob(p1_id, p2_id, elo_ratings)
        gold_side, gold_h, gold_prob = gold_tip(p_leg)
        gold_winner = handicap_winner(actual_p1legs, actual_p2legs, gold_h)
        gold_winner_str = "1 (home)" if gold_winner else "2 (away)"
        gold_would_win = (gold_side == gold_winner_str)

        st.subheader("Check results")
        col_r1, col_r2, col_r3 = st.columns(3)
        col_r1.metric("Your ticket", "✅ Passed" if ticket_pass else "❌ Failed")
        col_r2.metric("Gold Tip recommendation", f"Tip {gold_side}, HCP {gold_h:+.1f}")
        col_r3.metric("Gold Tip would have...", "✅ Won" if gold_would_win else "❌ Lost")

        st.write(f"Actual handicap winner ({played_h_str}): {real_winner_str}")
        st.write(f"Gold Tip would have bet: Tip {gold_side} with {gold_h:+.1f} (probability {gold_prob:.1%})")

# ========== CHECK TICKET (Rainbet) ==========
elif page == "🔍 Check ticket (Rainbet)":
    st.title("🔍 Ticket check – Rainbet rules")
    st.markdown("Enter the played match and your tip (handicap is tied to selected player).")

    player_names = list(st.session_state.players.values())
    player_ids = list(st.session_state.players.keys())

    col1, col2 = st.columns(2)
    with col1:
        selected_name = st.selectbox("Player on ticket", player_names, key="rb_sel")
    with col2:
        opponent_name = st.selectbox("Opponent", player_names, key="rb_opp")

    col3, col4 = st.columns(2)
    with col3:
        legs_selected = st.number_input(f"Legs of {selected_name}", 0, 4, 0, key="rb_legs_sel")
    with col4:
        legs_opponent = st.number_input(f"Legs of {opponent_name}", 0, 4, 0, key="rb_legs_opp")

    col5, col6 = st.columns(2)
    with col5:
        played_h_str = st.selectbox("Handicap on ticket", ["+1.5", "-1.5", "+2.5", "-2.5"], key="rb_h")
    with col6:
        st.write("Your tip is always **that the handicap passes** (you bet on the selected player)")

    if st.button("Check ticket (Rainbet)"):
        selected_id = player_ids[player_names.index(selected_name)]
        opponent_id = player_ids[player_names.index(opponent_name)]

        played_h_val = float(played_h_str.replace("+", ""))
        ticket_pass = handicap_winner(legs_selected, legs_opponent, played_h_val)

        p_leg = leg_prob(selected_id, opponent_id, elo_ratings)
        best_h, best_prob = gold_tip_player(p_leg)
        gold_would_win = handicap_winner(legs_selected, legs_opponent, best_h)

        st.subheader("Check results")
        col_r1, col_r2, col_r3 = st.columns(3)
        col_r1.metric("Your ticket", "✅ Passed" if ticket_pass else "❌ Failed")
        col_r2.metric("Gold Tip recommendation", f"{selected_name} {best_h:+.1f} (pass: {best_prob:.1%})")
        col_r3.metric("Gold Tip would have...", "✅ Won" if gold_would_win else "❌ Lost")

        st.write(f"Actual outcome: {selected_name} {legs_selected} – {opponent_name} {legs_opponent}")
        st.write(f"With handicap {played_h_str}: {legs_selected + played_h_val:.1f} – {legs_opponent}")

# ========== 180s Bounce Back (opći) ==========
elif page == "🔄 180s Bounce Back (opći)":
    st.title("🔄 180s Bounce Back (opći pregled)")
    st.markdown("Igrači s prosjekom >1.0 180s – nakon meča s 0 180s, koliko često sljedeći meč u istom danu ima ≥1?")
    if st.button("Pokreni analizu"):
        res, total, succ, rate = analyze_180_bounce(all_matches, st.session_state.players)
        if total == 0:
            st.info("Nema podataka.")
        else:
            st.success(f"Slučajeva: {total}, bounce back: {succ} ({rate:.1f}%)")
            df = [{"Igrač": r[0], "Datum": r[1], "Redni meč": r[2], "Sljedeći 180s": r[3], "Over 0.5": "✔️" if r[4] else "❌"} for r in res[:50]]
            st.dataframe(df, use_container_width=True)

# ========== 180s Player Bounce ==========
elif page == "📊 180s Player Bounce":
    st.title("📊 180s Bounce – pojedinačni igrači")
    st.markdown("Odaberi dva igrača za usporedbu prosjeka 180‑ica i postotka *bounce back* nakon meča s 0 180s.")
    player_names = list(st.session_state.players.values())
    player_ids = list(st.session_state.players.keys())

    col1, col2 = st.columns(2)
    with col1:
        p1_name = st.selectbox("Igrač 1", player_names, index=0)
    with col2:
        p2_name = st.selectbox("Igrač 2", player_names, index=1)

    p1_id = player_ids[player_names.index(p1_name)]
    p2_id = player_ids[player_names.index(p2_name)]

    if p1_id == p2_id:
        st.warning("Odaberi dva različita igrača.")
    else:
        avg1, rate1, tot1, succ1 = player_180_bounce(p1_id, all_matches)
        avg2, rate2, tot2, succ2 = player_180_bounce(p2_id, all_matches)

        st.subheader(f"📊 {p1_name}")
        col_a1, col_a2 = st.columns(2)
        col_a1.metric("Prosjek 180s/meč", f"{avg1:.2f}")
        col_a2.metric("Bounce back (0 → ≥1)", f"{rate1:.1f}% ({succ1}/{tot1})" if tot1 > 0 else "N/A")

        st.subheader(f"📊 {p2_name}")
        col_b1, col_b2 = st.columns(2)
        col_b1.metric("Prosjek 180s/meč", f"{avg2:.2f}")
        col_b2.metric("Bounce back (0 → ≥1)", f"{rate2:.1f}% ({succ2}/{tot2})" if tot2 > 0 else "N/A")

# ========== Bounce nakon 4:0 ==========
elif page == "🔄 Bounce nakon 4:0":
    st.title("🔄 Bounce nakon 4:0 – Under/Over 5.5 u sljedećem meču")
    st.markdown(
        "Koliko često **nakon što igrač izgubi 4:0**, njegov sljedeći meč "
        "u istom danu završi s **manje od 6 legova** (under 5.5)?\n\n"
        "Prikazujemo i **over 5.5** (≥6 legova) te statistiku po protivniku."
    )

    total, under, rate, per_player, per_player_opponent = analyze_bounce_after_4_0(all_matches)
    if total == 0:
        st.info("Nema podataka (nijedan igrač nije izgubio 4:0 i zatim odigrao još jedan meč istog dana).")
    else:
        over = total - under
        over_rate = 100.0 - rate
        st.metric("Ukupno slučajeva (4:0 → sljedeći meč)", total)
        col_u, col_o = st.columns(2)
        col_u.metric("Under 5.5 u sljedećem meču", f"{under} ({rate:.1f}%)")
        col_o.metric("Over 5.5 u sljedećem meču", f"{over} ({over_rate:.1f}%)")

        st.subheader("Statistika po igraču")
        player_names = list(st.session_state.players.values())
        player_ids = list(st.session_state.players.keys())
        selected_name = st.selectbox("Odaberi igrača", player_names)
        selected_id = player_ids[player_names.index(selected_name)]
        pdata = per_player.get(selected_id, {'total': 0, 'under': 0})
        if pdata['total'] > 0:
            p_over = pdata['total'] - pdata['under']
            p_under_rate = pdata['under'] / pdata['total'] * 100
            p_over_rate = 100.0 - p_under_rate
            st.write(f"**{selected_name}**: ukupno {pdata['total']} puta")
            col_pu, col_po = st.columns(2)
            col_pu.metric("Under 5.5", f"{pdata['under']} ({p_under_rate:.1f}%)")
            col_po.metric("Over 5.5", f"{p_over} ({p_over_rate:.1f}%)")
        else:
            st.write(f"**{selected_name}** nema takvih slučajeva.")

        # Forma igrača (zadnjih 5 mečeva)
        player_matches = [m for m in all_matches if m['p1id'] == selected_id or m['p2id'] == selected_id]
        sorted_pm = sorted(player_matches, key=lambda x: (x['date'], x.get('_meta', {}).get('time', '')), reverse=True)
        last5 = sorted_pm[:5]
        if last5:
            margins = []
            wins = []
            for m in last5:
                if m['p1id'] == selected_id:
                    mg = m['p1legs'] - m['p2legs']
                    w = 1 if m['p1legs'] > m['p2legs'] else 0
                else:
                    mg = m['p2legs'] - m['p1legs']
                    w = 1 if m['p2legs'] > m['p1legs'] else 0
                margins.append(mg)
                wins.append(w)
            avg_margin = sum(margins) / len(margins)
            winrate = sum(wins) / len(wins) * 100
            st.subheader("📈 Forma (posljednjih 5 mečeva)")
            col_f1, col_f2 = st.columns(2)
            col_f1.metric("Prosječna margina", f"{avg_margin:+.1f}")
            col_f2.metric("Win rate", f"{winrate:.0f}%")
        else:
            st.info("Nema podataka o formi.")

        # Analiza po protivniku
        opp_data = per_player_opponent.get(selected_id, {})
        if opp_data:
            st.subheader("📊 Prema protivniku u sljedećem meču")
            opp_names = [(opp_id, st.session_state.players.get(opp_id, str(opp_id))) for opp_id in opp_data.keys()]
            opp_names.sort(key=lambda x: x[1])
            selected_opp_name = st.selectbox("Odaberi protivnika", [name for _, name in opp_names])
            selected_opp_id = [oid for oid, nm in opp_names if nm == selected_opp_name][0]
            odata = opp_data[selected_opp_id]
            if odata['total'] > 0:
                o_over = odata['total'] - odata['under']
                o_under_rate = odata['under'] / odata['total'] * 100
                o_over_rate = 100.0 - o_under_rate
                st.write(f"Protiv **{selected_opp_name}**: {odata['total']} puta")
                col_ou, col_oo = st.columns(2)
                col_ou.metric("Under 5.5", f"{odata['under']} ({o_under_rate:.1f}%)")
                col_oo.metric("Over 5.5", f"{o_over} ({o_over_rate:.1f}%)")
            else:
                st.write("Nema podataka za ovog protivnika.")
        else:
            st.info("Nema detaljnih podataka o protivnicima za ovog igrača.")

# ========== Over 5.5 Legs ==========
elif page == "🔢 Over 5.5 Legs":
    st.title("🔢 Over 5.5 Legs – Empirijski gap‑bucket model")
    st.markdown(
        "Vjerojatnost da meč ima **barem 6 legova** (over 5.5) temelji se na "
        "**razlici u ukupnom postotku osvojenih legova** između igrača (gap). "
        "Tijesni mečevi (mali gap) najčešće idu u over, a veliki gap favorizira under."
    )

    player_names = list(st.session_state.players.values())
    player_ids = list(st.session_state.players.keys())

    col1, col2 = st.columns(2)
    with col1:
        p1_name = st.selectbox("👤 Player 1", player_names, index=0)
    with col2:
        p2_name = st.selectbox("👤 Player 2", player_names, index=1)

    p1_id = player_ids[player_names.index(p1_name)]
    p2_id = player_ids[player_names.index(p2_name)]

    if p1_id == p2_id:
        st.warning("Odaberi dva različita igrača.")
        st.stop()

    # Izračunaj gap
    wr = compute_leg_winrates(all_matches)
    gap = abs(wr.get(p1_id, 0.5) - wr.get(p2_id, 0.5)) * 100

    # Odredi bucket
    if gap < 5:
        bucket = "0–5%"
    elif gap < 10:
        bucket = "5–10%"
    elif gap < 15:
        bucket = "10–15%"
    elif gap < 20:
        bucket = "15–20%"
    else:
        bucket = "20%+"

    bucket_data = GAP_OVER_TABLE.get(bucket, {'total': 0, 'over': 0, 'pct': 0.0})
    over_prob = bucket_data['pct'] / 100.0
    under_prob = 1.0 - over_prob

    # Elo model (informativno)
    p_leg = leg_prob(p1_id, p2_id, elo_ratings)
    dist = match_distribution(p_leg)
    elo_over = sum(prob for (a,b), prob in dist.items() if a+b >= 6)

    # Prikaz
    st.subheader("📊 Gap analiza")
    col_gap, col_bucket = st.columns(2)
    col_gap.metric("Razlika u leg‑win%", f"{gap:.1f}%")
    col_bucket.metric("Bucket", bucket)

    st.write(f"**Empirijski over 5.5 u ovom bucketu:** {over_prob:.2%} "
             f"({bucket_data['over']}/{bucket_data['total']} mečeva)")
    st.write(f"**Empirijski under 5.5 u ovom bucketu:** {under_prob:.2%}")
    st.write(f"**Elo model (informativno) – over 5.5:** {elo_over:.2%}") 

        # ML model vjerojatnost (Random Forest)
    ml_over_prob = predict_over_under(over_under_model[0], over_under_model[1],
                                      p1_id, p2_id, elo_ratings, all_matches)
    st.write(f"**ML model (Random Forest) – over 5.5:** {ml_over_prob:.2%}") 

    # Profil rezultata igrača
    with st.expander("📋 Profil rezultata igrača (kratki 4:0/4:1 vs dugi 4:2/4:3)"):
        cur_bucket = gap_bucket_name(gap)

        def prof_rows(pid, pname):
            out = []
            for lbl, kw in [("Karijera", {}), ("Zadnjih 10", {"last_n": 10}),
                            (f"Bucket {cur_bucket}", {"bucket": cur_bucket})]:
                c, n = player_score_profile(pid, all_matches, LEG_WR, **kw)
                if n == 0:
                    out.append({"Igrač": pname, "Period": lbl, "n": 0, "4:0": "-", "4:1": "-",
                                "4:2": "-", "4:3": "-", "KRATKI U5.5": "-", "DUGI O5.5": "-"})
                    continue
                short = c[0] + c[1]; lng = c[2] + c[3]
                out.append({"Igrač": pname, "Period": lbl, "n": n,
                            "4:0": f"{c[0]/n*100:.0f}%", "4:1": f"{c[1]/n*100:.0f}%",
                            "4:2": f"{c[2]/n*100:.0f}%", "4:3": f"{c[3]/n*100:.0f}%",
                            "KRATKI U5.5": f"{short/n*100:.0f}%", "DUGI O5.5": f"{lng/n*100:.0f}%"})
            return out

        st.dataframe(prof_rows(p1_id, p1_name) + prof_rows(p2_id, p2_name),
                     use_container_width=True)
        st.caption("KRATKI = 4:0/4:1 (under 5.5) · DUGI = 4:2/4:3 (over 5.5). "
                   "'Bucket' red = samo mečevi protiv protivnika u istom gap tieru. Samo bo7.")

    # Tablica svih bucketa za referencu
    with st.expander("📋 Cijela gap‑bucket tablica"):
        table_rows = []
        for b in ["0–5%", "5–10%", "10–15%", "15–20%", "20%+"]:
            d = GAP_OVER_TABLE.get(b, {'total': 0, 'over': 0, 'pct': 0.0})
            table_rows.append({
                "Bucket": b,
                "Mečeva": d['total'],
                "Over 5.5": d['over'],
                "Postotak over": f"{d['pct']:.1f}%",
                "Postotak under": f"{100-d['pct']:.1f}%"
            })
        st.dataframe(table_rows, use_container_width=True)

    # Value kalkulator – odvojeni inputi za over i under
    with st.expander("💎 Value kalkulator"):
        st.write("Unesite koeficijente za over i/ili under. Edge se računa koristeći empirijsku vjerojatnost iz bucketa.")
        col_odds_over, col_odds_under = st.columns(2)
        with col_odds_over:
            odds_over = st.number_input("Koeficijent za Over 5.5", 1.01, 20.0, 2.0, 0.05, key="over55_odds")
        with col_odds_under:
            odds_under = st.number_input("Koeficijent za Under 5.5", 1.01, 20.0, 2.0, 0.05, key="under55_odds")

        if st.button("Izračunaj value", key="calc_over55"):
            # Over edge
            if odds_over > 1.0:
                impl_over = 1 / odds_over
                edge_over = over_prob - impl_over
                edge_ml_over = ml_over_prob - impl_over
                st.write(f"**Over 5.5** – empirijska vjerojatnost: {over_prob:.2%}, implicirana: {impl_over:.2%}, edge: {edge_over:+.2%}")
                if edge_over > VALUE_THRESHOLD:
                    st.success(f"VALUE! Očekivani povrat: {(over_prob * odds_over - 1) * 100:.1f}%")
                elif edge_over > 0:
                    st.info("Mala prednost, ali ispod praga značajnosti.")
                else:
                    st.warning("Nema valuea.")
            else:
                st.write("**Over 5.5** – nije unesen koeficijent.")

            st.markdown("---")

            st.write(f"**ML model edge:** {edge_ml_over:+.2%}")
            if edge_ml_over > VALUE_THRESHOLD:
               st.success("ML VALUE!")
            

            # Under edge
            if odds_under > 1.0:
                impl_under = 1 / odds_under
                edge_under = under_prob - impl_under
                st.write(f"**Under 5.5** – empirijska vjerojatnost: {under_prob:.2%}, implicirana: {impl_under:.2%}, edge: {edge_under:+.2%}")
                if edge_under > VALUE_THRESHOLD:
                    st.success(f"VALUE! Očekivani povrat: {(under_prob * odds_under - 1) * 100:.1f}%")
                elif edge_under > 0:
                    st.info("Mala prednost, ali ispod praga značajnosti.")
                else:
                    st.warning("Nema valuea.")
            else:
                st.write("**Under 5.5** – nije unesen koeficijent.")

# ========== Under 5.5 (Gap + Forma) ==========
elif page == "🎯 Under 5.5 (Gap + Forma)":
    st.title("🎯 Under 5.5 Legs – Gap + hladnoća autsajdera")
    st.markdown(
        "Dvostruki filter: **razlika u leg‑win% (gap)** + **forma slabijeg igrača (roll3)**. "
        "Signal je *hladan autsajder* protiv jačeg → blowout → manje legova → under.\n\n"
        "- **Režim A** (dnevni): gap ≥15% i autsajder roll3 ≤0.35\n"
        "- **Režim B** (jača ivica, rjeđi): gap ≥20% i autsajder roll3 ≤0.25"
    )

    player_names = list(st.session_state.players.values())
    player_ids = list(st.session_state.players.keys())
    col1, col2 = st.columns(2)
    with col1:
        p1_name = st.selectbox("👤 Player 1", player_names, index=0, key="u55_p1")
    with col2:
        p2_name = st.selectbox("👤 Player 2", player_names, index=1, key="u55_p2")
    p1_id = player_ids[player_names.index(p1_name)]
    p2_id = player_ids[player_names.index(p2_name)]
    if p1_id == p2_id:
        st.warning("Odaberi dva različita igrača.")
        st.stop()

    res = classify_under55(p1_id, p2_id, LEG_WR, ROLL3, UNDER55_REGIME)
    if res is None:
        st.error("Nedovoljno podataka za jednog od igrača.")
        st.stop()

    fav_name = st.session_state.players.get(res['fav_id'], '?')
    dog_name = st.session_state.players.get(res['und_id'], '?')
    ur_str = f"{res['und_roll3']:.2f}" if res['und_roll3'] is not None else "N/A"

    st.subheader("📊 Analiza")
    c1, c2, c3 = st.columns(3)
    c1.metric("Gap (leg‑win%)", f"{res['gap']:.1f}%")
    c2.metric("Favorit", fav_name)
    c3.metric("Autsajder roll3", ur_str)
    st.caption(f"Autsajder: **{dog_name}** — roll3 = udio osvojenih legova u zadnja 3 meča.")

    if res['regime'] is None:
        st.error("⛔ NEMA STAVE – par ne ispunjava uvjete ni za Režim A ni B.")
        st.markdown("Treba **gap ≥15% i roll3 ≤0.35** (A) ili **gap ≥20% i roll3 ≤0.25** (B). "
                    "Bolje preskočiti nego forsirati.")
    else:
        be = res['be_odds']
        st.success(f"✅ **REŽIM {res['regime']}** → kladi se na **UNDER 5.5**")
        m1, m2, m3 = st.columns(3)
        m1.metric("Empirijski under 5.5", f"{res['under_pct']:.1f}%")
        m2.metric("Break-even kvota", f"{be:.3f}")
        m3.metric("Min. kvota za stavu", f"≥ {be:.2f}")
        st.caption(f"Bazirano na {res['n']} povijesnih mečeva (leak-free).")
        if res['regime'] == 'B':
            st.info("💪 Režim B = jača ivica, ali rijedak i tanji uzorak (razmotri veći stake).")

    with st.expander("💎 Value kalkulator"):
        odds_under = st.number_input("Koeficijent za Under 5.5", 1.01, 20.0, 1.80, 0.01, key="u55_odds")
        if st.button("Izračunaj value", key="u55_calc"):
            if res['regime'] is None:
                st.warning("Par nije kvalifikovan – value se ne računa.")
            else:
                p = res['under_pct'] / 100.0
                impl = 1 / odds_under
                edge = p - impl
                ev = (p * odds_under - 1) * 100
                st.write(f"**Empirijska vjerojatnost:** {p:.2%}")
                st.write(f"**Implicirana iz kvote:** {impl:.2%}")
                st.write(f"**Edge:** {edge:+.2%}")
                if odds_under >= res['be_odds'] and edge > VALUE_THRESHOLD:
                    st.success(f"VALUE! Očekivani povrat: {ev:+.1f}%")
                elif edge > 0:
                    st.info(f"Mala prednost (EV {ev:+.1f}%), ispod praga – oprezno.")
                else:
                    st.error(f"Nema valuea – ispod break-evena ({res['be_odds']:.2f}). EV {ev:+.1f}%")

    with st.expander("📋 Tablica režima (leak-free backtest)"):
        labels = {'BASE15': 'gap ≥15% (bez forme)',
                  'A': 'A: gap ≥15% & roll3 ≤0.35',
                  'B': 'B: gap ≥20% & roll3 ≤0.25'}
        rows = []
        for k in ['BASE15', 'A', 'B']:
            v = UNDER55_REGIME.get(k, {})
            if v.get('n'):
                rows.append({"Režim": labels[k], "Mečeva": v['n'], "Under 5.5": v['under'],
                             "Postotak": f"{v['pct']:.1f}%",
                             "Break-even": f"{v['be']:.3f}" if v['be'] else "-"})
        st.dataframe(rows, use_container_width=True)
        st.caption("Min. kvota = break-even. Stava ima smisla samo iznad nje.")

# ========== Očisti duplikate ==========
elif page == "🧹 Očisti duplikate":
    st.title("🧹 Očisti duplikate u bazi")
    st.markdown("Pronađi i ukloni mečeve koji su identični (isti igrači, datum, rezultat i meta), a imaju različite ID‑eve.")

    if st.button("🔍 Skeniraj duplikate"):
        duplicates = find_all_duplicates(all_matches)
        if not duplicates:
            st.success("Nema duplikata! Baza je čista.")
        else:
            total_duplicates = sum(len(v) - 1 for v in duplicates.values())
            st.warning(f"Pronađeno {len(duplicates)} grupa duplikata, ukupno {total_duplicates} viška mečeva.")
            
            for i, (key, group) in enumerate(duplicates.items()):
                with st.expander(f"Grupa #{i+1} – {len(group)} mečeva"):
                    st.write("Ključ:", key)
                    for j, m in enumerate(group):
                        st.write(f"Meč #{j+1}: ID={m.get('id')}, {st.session_state.players.get(m['p1id'], '?')} vs {st.session_state.players.get(m['p2id'], '?')} ({m['p1legs']}:{m['p2legs']}) na {m['date']}")
                    st.dataframe([{
                        "ID": m.get('id'),
                        "Domaćin": st.session_state.players.get(m['p1id'], '?'),
                        "Gost": st.session_state.players.get(m['p2id'], '?'),
                        "Rezultat": f"{m['p1legs']}:{m['p2legs']}",
                        "Datum": m['date']
                    } for m in group], use_container_width=True)
            
            if st.button("🗑️ Obriši sve duplikate (zadrži samo prvi iz svake grupe)"):
                ids_to_delete = set()
                for key, group in duplicates.items():
                    for m in group[1:]:
                        ids_to_delete.add(m['id'])
                cleaned = [m for m in all_matches if m['id'] not in ids_to_delete]
                before = len(all_matches)
                after = len(cleaned)
                st.session_state.matches = cleaned
                st.session_state.new_matches = []
                save_to_json()
                st.success(f"Obrisano {before - after} mečeva. Baza sada ima {after} unosa.")
                st.rerun()