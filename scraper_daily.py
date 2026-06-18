"""
MODUS SUPER SERIES - Daily Scraper v5.4 (week-lock + date guard)
=============================================================
Pokreni:  python scraper_daily.py
Output:   novi_dan.json (samo danasnji mecevi)
           master.json (ako MERGE = True)
"""

import json, re, os, hashlib, shutil
from datetime import datetime, timedelta
from playwright.sync_api import sync_playwright, TimeoutError as PWTimeout

RESULTS_URL        = "https://modussuperseries.com/results"
OUTPUT_FILE        = "master.json"
OUTPUT_FILE_DAILY  = "novi_dan.json"
PROGRESS_FILE      = "novi_dan_progress.json"

# ============================================================
# PODESAVANJA — mijenjaj ovo svaki dan
# ============================================================
TARGET_SERIES = "Series 14"
TARGET_WEEK   = "W 8 - June 17"   # week broj se izvlaci automatski
TARGET_GROUPS = ["Group A"]

SKIP_ZERO_ZERO = True   # preskoci neodigrane meceve (0-0)
MERGE          = True   # merge s postojecim master.json (UVIJEK TRUE!)

# Koliko dana traje jedan tjedan (za filter datuma). Modus = pon-sub.
WEEK_SPAN_DAYS = 6      # prihvaca meceve [pocetak .. pocetak+6]
ENFORCE_DATE_WINDOW = True  # odbaci meceve izvan tjedna (sprjecava W8 da udje)
# ============================================================

WAIT_AFTER_DROPDOWN = 1800
WAIT_AFTER_GROUP    = 1400
WAIT_DETAIL         = 1500
WAIT_BACK           = 800
HEADLESS            = False

# =============================================================================

def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)

def wait(page, ms):
    page.wait_for_timeout(ms)

def clean(text):
    return re.sub(r"\s+", " ", text or "").strip()

def normalize_group(name):
    return clean(name).title()

def player_id(name):
    h = hashlib.sha256(name.strip().upper().encode()).hexdigest()
    return int(h[:13], 16)

def parse_week_date(label):
    m = re.search(r"([A-Za-z]+)\s+(\d{1,2})\s*$", label)
    if m:
        now = datetime.now()
        try:
            month_num = datetime.strptime(m.group(1), "%B").month
        except ValueError:
            try:
                month_num = datetime.strptime(m.group(1), "%b").month
            except ValueError:
                return now.strftime("%Y-%m-%d"), now.year
        day = int(m.group(2))
        yr = now.year - 1 if month_num > now.month else now.year
        return f"{yr}-{month_num:02d}-{day:02d}", yr
    now = datetime.now()
    return now.strftime("%Y-%m-%d"), now.year

def week_window(week_date_str):
    """Vraca (start_date, end_date) za filter datuma tjedna."""
    try:
        start = datetime.strptime(week_date_str, "%Y-%m-%d").date()
    except ValueError:
        return None, None
    return start, start + timedelta(days=WEEK_SPAN_DAYS)

def parse_match_date(raw, year_hint):
    raw = clean(raw).split("/")[0].strip()
    for fmt in ("%d %b %Y", "%d %B %Y"):
        try:
            dt = datetime.strptime(raw, fmt)
            return dt.strftime("%Y-%m-%d")
        except ValueError:
            pass
    try:
        dt = datetime.strptime(raw, "%b %d %Y")
        return dt.strftime("%Y-%m-%d")
    except:
        return raw

def make_match_key(series, week, group, date, time_str, p1id, p2id):
    # NAPOMENA: 'week' namjerno IZOSTAVLJEN iz kljuca jer su week oznake
    # nekonzistentne u povijesnim podacima. Serija+grupa+datum+vrijeme+igraci
    # su dovoljni za jedinstvenost.
    a, b = min(p1id, p2id), max(p1id, p2id)
    return f"{series}|{group}|{date}|{time_str}|{a}|{b}"

def js_click(page, el):
    page.evaluate("el => el.click()", el)

def get_group_names(page):
    buttons = page.query_selector_all(".group-tabs button")
    groups = []
    for btn in buttons:
        text = clean(btn.inner_text())
        if text:
            groups.append(normalize_group(text))
    return groups

def current_week_value(page):
    try:
        return page.eval_on_selector("#weekSelect", "el => el.value")
    except Exception:
        return None

def ensure_week(page, week_value):
    """Ponovo postavi tjedan ako se prikaz resetirao (npr. nakon detalja meca)."""
    cur = current_week_value(page)
    if cur != week_value:
        try:
            page.select_option("#weekSelect", value=week_value)
            wait(page, WAIT_AFTER_DROPDOWN)
            log(f"    ↻ Tjedan vracen na value={week_value} (bio {cur})")
        except Exception as e:
            log(f"    ⚠️  Ne mogu re-selektirati tjedan: {e}")

def click_group(page, gname):
    try:
        page.click(f".group-tabs button:has-text('{gname}')")
        wait(page, WAIT_AFTER_GROUP)
        return True
    except Exception as e:
        log(f"  ⚠️  Ne mogu kliknuti grupu '{gname}': {e}")
        return False

def ensure_context(page, week_value, gname):
    """Zajamci da je prikaz na ispravnom tjednu I grupi."""
    ensure_week(page, week_value)
    click_group(page, gname)

def get_match_stats(page, fixture_card, year_hint):
    """Otvara detalje meca (klikom na karticu) i cita podatke."""
    result = {"legs1": 0, "legs2": 0, "s180_1": 0, "s180_2": 0, "date": None}
    try:
        js_click(page, fixture_card)
        try:
            page.wait_for_selector(".stat-row .stat-label:has-text('180s')", timeout=5000)
        except PWTimeout:
            log("    ⚠️  Redak s 180s nije pronaden, nastavljam...")
        wait(page, 1000)

        date_tab = page.query_selector(".meta-right a.tab")
        if date_tab:
            raw_date = clean(date_tab.inner_text())
            result["date"] = parse_match_date(raw_date, year_hint)

        score_spans = page.query_selector_all(".score-area span")
        if len(score_spans) >= 2:
            t1 = clean(score_spans[0].inner_text())
            t2 = clean(score_spans[1].inner_text())
            result["legs1"] = int(t1) if t1.isdigit() else 0
            result["legs2"] = int(t2) if t2.isdigit() else 0

        stat_row = page.query_selector(".stat-row:has(.stat-label:has-text('180s'))")
        if stat_row:
            left = stat_row.query_selector(".stat-left")
            right = stat_row.query_selector(".stat-right")
            if left:
                v = clean(left.inner_text()).split("/")[0]
                if v.isdigit():
                    result["s180_1"] = int(v)
            if right:
                v = clean(right.inner_text()).split("/")[0]
                if v.isdigit():
                    result["s180_2"] = int(v)
        else:
            log("    ⚠️  Nije pronaden redak za 180s")

        back_btn = page.query_selector(".meta-left a.tab")
        if back_btn:
            js_click(page, back_btn)
        else:
            page.go_back()
        wait(page, WAIT_BACK)

    except Exception as e:
        log(f"    ⚠️  stats error: {e}")
        try:
            page.go_back()
            wait(page, WAIT_BACK)
        except:
            pass

    return result

# ---------- Resume / Save ----------
def load_daily_progress():
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, "r", encoding="utf-8") as f:
            p = json.load(f)
        log(f"  ↩ RESUME: {len(p.get('done_groups',[]))} gotovo, {len(p.get('daily_new',[]))} novih meceva")
        return p
    return {"done_groups": [], "daily_new": []}

def save_daily_progress(done_groups, daily_new):
    with open(PROGRESS_FILE, "w", encoding="utf-8") as f:
        json.dump({
            "done_groups": done_groups,
            "daily_new": daily_new,
            "saved_at": datetime.now().isoformat(),
        }, f, ensure_ascii=False)

def load_existing():
    if not MERGE or not os.path.exists(OUTPUT_FILE):
        return {"players": {}, "matches": [], "seen": set()}
    with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    players = {}
    for p in data.get("players", []):
        players[p["name"].strip().upper()] = int(p["id"])
    matches = data.get("matches", [])
    seen = set()
    for m in matches:
        meta = m.get("_meta", {})
        k = make_match_key(
            meta.get("series",""), meta.get("week",""), meta.get("group",""),
            m.get("date",""), meta.get("time",""),
            m.get("p1id",0), m.get("p2id",0)
        )
        seen.add(k)
    log(f"  Ucitano iz {OUTPUT_FILE}: {len(players)} igraca, {len(matches)} meceva")
    return {"players": players, "matches": matches, "seen": seen}

def save_output(players, matches, also_save_daily=True, daily_matches=None):
    players_out = sorted(
        [{"id": pid, "name": name} for name, pid in players.items()],
        key=lambda x: x["name"].lower(),
    )

    # SIGURNOSNA PROVJERA: ne dopusti da se baza smanji (brisanje)
    if os.path.exists(OUTPUT_FILE):
        try:
            with open(OUTPUT_FILE, "r", encoding="utf-8") as fc:
                existing_count = len(json.load(fc).get("matches", []))
        except Exception:
            existing_count = 0
        if len(matches) < existing_count:
            log(f"  ❌ ODBIJENO pisanje: {len(matches)} < {existing_count} postojecih meceva.")
            raise RuntimeError("Sprjecavam brisanje baze. Provjeri MERGE i logiku.")
        # Backup prije pisanja
        shutil.copy2(OUTPUT_FILE, OUTPUT_FILE.replace(".json", "_backup.json"))
        log(f"  🔒 Backup: {OUTPUT_FILE.replace('.json', '_backup.json')}")

    output = {
        "exported_at": datetime.now().isoformat(),
        "source": "Modus Super Series",
        "players": players_out,
        "matches": matches,
    }
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    log(f"  💾 {OUTPUT_FILE}: {len(players_out)} igraca, {len(matches)} meceva")
    if also_save_daily and daily_matches is not None:
        daily_out = {
            "exported_at": datetime.now().isoformat(),
            "source": f"Modus Daily - {TARGET_SERIES} {TARGET_WEEK}",
            "players": players_out,
            "matches": daily_matches,
        }
        with open(OUTPUT_FILE_DAILY, "w", encoding="utf-8") as f:
            json.dump(daily_out, f, ensure_ascii=False, indent=2)
        log(f"  💾 {OUTPUT_FILE_DAILY}: {len(daily_matches)} novih meceva")

# =============================================================================

def scrape_daily():
    log("=" * 60)
    log("  MODUS DAILY SCRAPER v5.4 (week-lock + date guard)")
    log(f"  Serija : {TARGET_SERIES}")
    log(f"  Tjedan : {TARGET_WEEK}")
    log(f"  Grupe  : {TARGET_GROUPS}")
    log(f"  Skip 0-0: {SKIP_ZERO_ZERO}")
    log("=" * 60)

    existing = load_existing()
    players = existing["players"]
    matches = existing["matches"]
    seen_matches = existing["seen"]
    daily_new = []

    week_date_str, year_hint = parse_week_date(TARGET_WEEK)
    win_start, win_end = week_window(week_date_str)
    log(f"\nDatum tjedna: {week_date_str} (year_hint={year_hint})")
    if ENFORCE_DATE_WINDOW and win_start:
        log(f"Prozor tjedna: {win_start} .. {win_end} (sve izvan se odbacuje)\n")

    def get_or_create(name):
        key = name.strip().upper()
        if key not in players:
            players[key] = player_id(key)
        return players[key]

    daily_prog = load_daily_progress()
    done_groups = daily_prog.get("done_groups", [])
    for m in daily_prog.get("daily_new", []):
        meta = m.get("_meta", {})
        k = make_match_key(
            meta.get("series",""), meta.get("week",""), meta.get("group",""),
            m.get("date",""), meta.get("time",""),
            m.get("p1id",0), m.get("p2id",0)
        )
        if k not in seen_matches:
            seen_matches.add(k)
            matches.append(m)
            daily_new.append(m)

    new_total = skipped_zero = skipped_dup = skipped_outside = 0

    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=HEADLESS)
        ctx = browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            viewport={"width": 1400, "height": 900},
        )
        page = ctx.new_page()

        log(f"Ucitavam {RESULTS_URL} ...")
        try:
            page.goto(RESULTS_URL, wait_until="networkidle", timeout=30000)
        except PWTimeout:
            page.goto(RESULTS_URL, timeout=30000)
        wait(page, 2500)

        # --- Serija ---
        try:
            series_value = TARGET_SERIES.split()[-1]
            page.select_option("#seriesSelect", value=series_value)
            wait(page, WAIT_AFTER_DROPDOWN)
            log(f"✅ Serija: {TARGET_SERIES}")
        except Exception as e:
            log(f"GRESKA: Ne mogu odabrati seriju '{TARGET_SERIES}' – {e}")
            browser.close()
            return

        # --- Tjedan ---
        try:
            week_number = re.search(r"W\s*(\d+)", TARGET_WEEK).group(1)
            target_text = f"Week {week_number}"
            options = page.query_selector_all("#weekSelect option")
            week_value = None
            for opt in options:
                if target_text in opt.inner_text():
                    week_value = opt.get_attribute("value")
                    break
            if not week_value:
                log(f"GRESKA: Tjedan '{target_text}' nije pronaden")
                browser.close()
                return
            page.select_option("#weekSelect", value=week_value)
            wait(page, WAIT_AFTER_DROPDOWN)
            log(f"✅ Tjedan: {target_text} (value={week_value})")
        except Exception as e:
            log(f"GRESKA: Ne mogu odabrati tjedan – {e}")
            browser.close()
            return

        # --- Grupe ---
        available = get_group_names(page)
        log(f"Dostupne grupe: {available}")
        groups_to_scrape = [g for g in TARGET_GROUPS if g in available]
        if not groups_to_scrape:
            log("⚠️  Niti jedna trazena grupa nije dostupna!")
            browser.close()
            return
        log(f"Scrapamo: {groups_to_scrape}\n")

        for gname in groups_to_scrape:
            if gname in done_groups:
                log(f"── GRUPA: {gname} — preskacem (resume)")
                continue

            log(f"── GRUPA: {gname} ──")
            # ZAJAMCI tjedan PRIJE klika grupe (sprjecava da se cita W8)
            ensure_week(page, week_value)
            if not click_group(page, gname):
                log(f"  ⚠️  Ne mogu kliknuti '{gname}', preskacem")
                continue

            cards = page.query_selector_all("article.fixture-card")
            log(f"  Meceva u grupi: {len(cards)}")
            group_new = 0

            for idx in range(len(cards)):
                try:
                    cards = page.query_selector_all("article.fixture-card")
                    if idx >= len(cards):
                        break
                    card = cards[idx]

                    player_rows = card.query_selector_all("div.player-row")
                    if len(player_rows) < 2:
                        continue

                    row1 = player_rows[0]
                    score1_el = row1.query_selector("span.score")
                    name1_el = row1.query_selector("span:not(.score)") or row1.query_selector("span:last-child")
                    p1_leg = clean(score1_el.inner_text()) if score1_el else "0"
                    p1_name = clean(name1_el.inner_text()) if name1_el else ""

                    row2 = player_rows[1]
                    score2_el = row2.query_selector("span.score")
                    name2_el = row2.query_selector("span:not(.score)") or row2.query_selector("span:last-child")
                    p2_leg = clean(score2_el.inner_text()) if score2_el else "0"
                    p2_name = clean(name2_el.inner_text()) if name2_el else ""

                    if not p1_name or not p2_name:
                        continue

                    p1_legs_raw = int(p1_leg) if p1_leg.isdigit() else 0
                    p2_legs_raw = int(p2_leg) if p2_leg.isdigit() else 0

                    if SKIP_ZERO_ZERO and p1_legs_raw == 0 and p2_legs_raw == 0:
                        log(f"  [{idx+1}] {p1_name} vs {p2_name} — PRESKACEM (0-0)")
                        skipped_zero += 1
                        continue

                    p1id = get_or_create(p1_name)
                    p2id = get_or_create(p2_name)
                    time_str = ""

                    log(f"  [{idx+1}] {p1_name} {p1_legs_raw}-{p2_legs_raw} {p2_name}")

                    stats = get_match_stats(page, card, year_hint)

                    # Nakon detalja: VRATI tjedan i grupu (prikaz se resetira!)
                    ensure_context(page, week_value, gname)

                    date = stats["date"] or week_date_str

                    # === FILTER DATUMA: odbaci sve izvan tjedna (npr. W8) ===
                    if ENFORCE_DATE_WINDOW and win_start:
                        try:
                            md = datetime.strptime(date, "%Y-%m-%d").date()
                            if not (win_start <= md <= win_end):
                                log(f"    ⛔ IZVAN TJEDNA ({date}) — preskacem (ne pripada {TARGET_WEEK})")
                                skipped_outside += 1
                                continue
                        except ValueError:
                            log(f"    ⚠️  Neispravan datum '{date}' — preskacem")
                            skipped_outside += 1
                            continue

                    mk = make_match_key(TARGET_SERIES, TARGET_WEEK, gname, date, time_str, p1id, p2id)
                    if mk in seen_matches:
                        log(f"    ↩ Duplikat ({date})")
                        skipped_dup += 1
                        continue

                    seen_matches.add(mk)
                    match_obj = {
                        "id":      len(matches) + 1,
                        "p1id":    p1id,
                        "p2id":    p2id,
                        "p1legs":  stats["legs1"] or p1_legs_raw,
                        "p2legs":  stats["legs2"] or p2_legs_raw,
                        "p1_180s": stats["s180_1"],
                        "p2_180s": stats["s180_2"],
                        "date":    date,
                        "_meta": {
                            "series": TARGET_SERIES,
                            "week":   TARGET_WEEK,
                            "group":  gname,
                            "time":   time_str,
                        },
                    }
                    matches.append(match_obj)
                    daily_new.append(match_obj)
                    group_new += 1
                    new_total += 1
                    log(f"    ✅  180s {stats['s180_1']}/{stats['s180_2']}  {date}  (novi: {new_total})")

                except Exception as e:
                    log(f"  ⚠️  Mec {idx}: {e}")
                    try:
                        ensure_context(page, week_value, gname)
                    except:
                        pass
                    continue

            done_groups.append(gname)
            save_daily_progress(done_groups, daily_new)
            log(f"  └─ {gname}: +{group_new} novih  💾 checkpoint\n")

        browser.close()

    save_output(players, matches, daily_matches=daily_new)

    if os.path.exists(PROGRESS_FILE):
        os.remove(PROGRESS_FILE)
        log("  Progress file obrisan.")

    log("\n" + "=" * 60)
    log("  ✅  DAILY SCRAPING ZAVRSEN!")
    log(f"  Novi mecevi      : {new_total}")
    log(f"  Preskoceni 0-0   : {skipped_zero}")
    log(f"  Izvan tjedna     : {skipped_outside}")
    log(f"  Duplikati        : {skipped_dup}")
    log(f"  Ukupno u bazi    : {len(matches)}")
    log("=" * 60)
    log(f"Importaj {OUTPUT_FILE_DAILY} u Darts Oracle → tab ⬆️ Import")

if __name__ == "__main__":
    scrape_daily()