import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

sys.path.append(os.path.dirname(__file__))
from scraper_core import (
    collect_all_urls,
    create_driver,
    get_config,
    load_url_cache,
    save_url_cache,
    scrape_product,
)

CATEGORY_LABEL = "Serums_&_Essences"
CATEGORY_SLUG = "Serums_%26_Essences"
CACHE_KEY = CATEGORY_SLUG
START_PAGE = 1
END_PAGE = 260
OUTPUT_FILE = Path(__file__).with_name(f"{CATEGORY_LABEL}.csv")

SHORT_WAIT_EVERY = 10
SHORT_WAIT_SECONDS = 5
LONG_WAIT_EVERY = 100
LONG_WAIT_SECONDS = 30
ERROR_BACKOFF_SECONDS = 10
BACK_BUTTON_DELAY = 0.5


def _normalize_cell(value: object) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    text = str(value).strip()
    return "" if text.lower() in {"nan", "none"} else text


def _configure() -> dict:
    cfg = get_config()
    cfg["HEADLESS"] = True
    cfg["PAGELOAD_TIMEOUT"] = max(45, cfg.get("PAGELOAD_TIMEOUT", 35))
    cfg["WAIT_FOR_BODY_TIMEOUT"] = max(12, cfg.get("WAIT_FOR_BODY_TIMEOUT", 8))
    cfg["MIN_DELAY"] = 0.05
    cfg["MAX_DELAY"] = 0.25
    cfg["COOLDOWN_AFTER_CONSECUTIVE_PRODUCT_ERRORS"] = 0
    cfg["GLOBAL_COOLDOWN_EVERY_N_SAVES"] = 0
    cfg["GLOBAL_COOLDOWN_SECONDS"] = 0
    cfg["USE_URL_CACHE"] = True
    cfg["RETRIES_PER_PRODUCT"] = max(2, cfg.get("RETRIES_PER_PRODUCT", 2))
    cfg["PRODUCT_HARD_TIMEOUT"] = max(15, cfg.get("PRODUCT_HARD_TIMEOUT", 15))
    cfg["PAGE_BATCH_SIZE"] = min(max(20, cfg.get("PAGE_BATCH_SIZE", 100)), END_PAGE - START_PAGE + 1)
    return cfg


def _dedupe_urls(urls: List[str]) -> List[str]:
    unique: List[str] = []
    seen = set()
    for raw in urls:
        if not isinstance(raw, str):
            continue
        url = raw.strip()
        if url and url not in seen:
            unique.append(url)
            seen.add(url)
    return unique


def _record_is_complete(record: Optional[Dict[str, str]]) -> bool:
    if not record:
        return False
    return bool(record.get("name") and record.get("ingredients"))


def _load_existing(output_file: Path) -> Dict[str, Dict[str, str]]:
    if not output_file.exists():
        return {}
    try:
        df = pd.read_csv(output_file)
    except Exception as exc:
        print(f"[WARN] Could not read existing CSV: {exc}")
        return {}
    existing: Dict[str, Dict[str, str]] = {}
    for _, row in df.iterrows():
        url = _normalize_cell(row.get("product_url"))
        if not url or url in existing:
            continue
        status = (_normalize_cell(row.get("status")) or "").lower()
        existing[url] = {
            "name": _normalize_cell(row.get("name")),
            "ingredients": _normalize_cell(row.get("ingredients")),
            "status": status,
        }
    return existing


def _update_progress(progress: Dict[str, Dict[str, str]], url: str, name: str, ingredients: str, status: str):
    progress[url] = {
        "name": _normalize_cell(name),
        "ingredients": _normalize_cell(ingredients),
        "status": status.lower() if status else "",
    }


def _write_output(urls: List[str], progress: Dict[str, Dict[str, str]]):
    rows = []
    for idx, url in enumerate(urls, start=1):
        record = progress.get(url, {})
        rows.append({
            "id": idx,
            "category": CATEGORY_LABEL,
            "product_url": url,
            "name": record.get("name", ""),
            "ingredients": record.get("ingredients", ""),
            "status": record.get("status", ""),
        })
    df = pd.DataFrame(rows, columns=["id", "category", "product_url", "name", "ingredients", "status"])
    df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8")


def _restart_driver(driver_ref: Dict[str, Optional[object]], cfg: Dict, reason: str):
    print(f"[INFO] Restarting browser session ({reason}). Waiting {ERROR_BACKOFF_SECONDS}s.")
    try:
        driver = driver_ref.get("driver")
        if driver:
            driver.quit()
    except Exception:
        pass
    time.sleep(ERROR_BACKOFF_SECONDS)
    driver_ref["driver"] = create_driver(cfg)


def _maybe_wait(scrape_attempts: int):
    if scrape_attempts <= 0:
        return
    if scrape_attempts % LONG_WAIT_EVERY == 0:
        print(f"[PAUSE] {scrape_attempts} new products processed. Waiting {LONG_WAIT_SECONDS}s.")
        time.sleep(LONG_WAIT_SECONDS)
    elif scrape_attempts % SHORT_WAIT_EVERY == 0:
        print(f"[PAUSE] {scrape_attempts} new products processed. Waiting {SHORT_WAIT_SECONDS}s.")
        time.sleep(SHORT_WAIT_SECONDS)


def _ensure_urls(driver, cfg) -> List[str]:
    urls, names = load_url_cache(CACHE_KEY, START_PAGE, END_PAGE)
    refresh = cfg.get("REFRESH_URL_CACHE") or not urls
    if refresh:
        urls, names = collect_all_urls(
            driver,
            cfg,
            CATEGORY_SLUG,
            START_PAGE,
            END_PAGE,
            cfg["PAGE_BATCH_SIZE"],
        )
        if cfg.get("USE_URL_CACHE"):
            save_url_cache(CACHE_KEY, START_PAGE, END_PAGE, urls, names)
    return _dedupe_urls(urls)


def _retry_missing(driver_ref: Dict[str, Optional[object]], cfg: Dict, urls: List[str], progress: Dict[str, Dict[str, str]], scrape_attempts: int) -> int:
    pending = [u for u in urls if not _record_is_complete(progress.get(u))]
    if not pending:
        return scrape_attempts
    print(f"\n--- RETRY PASS: {len(pending)} URLs still missing data ---")
    for url in pending:
        if driver_ref.get("driver") is None:
            _restart_driver(driver_ref, cfg, reason="driver missing before retry")
        driver = driver_ref["driver"]
        try:
            data = scrape_product(driver, cfg, url)
        except Exception as exc:
            print(f"[ERROR] Retry exception for {url}: {exc}")
            data = {"name": None, "ingredients": None}
        scrape_attempts += 1
        if data.get("name"):
            _update_progress(progress, url, data.get("name"), data.get("ingredients"), "scraped")
            print(f"[RETRY-OK] {url}")
        else:
            _update_progress(progress, url, "", "", "skipped")
            print(f"[RETRY-SKIP] {url}")
            try:
                driver.back()
                time.sleep(BACK_BUTTON_DELAY)
            except Exception:
                pass
            _restart_driver(driver_ref, cfg, reason=f"retry error on {url}")
        _write_output(urls, progress)
        _maybe_wait(scrape_attempts)
    return scrape_attempts


def main():
    cfg = _configure()
    driver_ref: Dict[str, Optional[object]] = {"driver": create_driver(cfg)}
    try:
        urls = _ensure_urls(driver_ref["driver"], cfg)
        if not urls:
            print("[ERROR] URL list is empty; aborting.")
            return

        progress = _load_existing(OUTPUT_FILE)
        changed = False
        for record in progress.values():
            if _record_is_complete(record) and not record.get("status"):
                record["status"] = "cached"
                changed = True
        if changed:
            _write_output(urls, progress)

        scrape_attempts = 0

        for url in urls:
            record = progress.get(url)
            status = (record.get("status") if record else "").lower()
            if _record_is_complete(record) and status not in ("", "skipped", "error"):
                print(f"[CACHE] Using saved data for {url}")
                continue

            if driver_ref.get("driver") is None:
                _restart_driver(driver_ref, cfg, reason="driver missing before scrape")
            driver = driver_ref["driver"]
            try:
                data = scrape_product(driver, cfg, url)
            except Exception as exc:
                print(f"[ERROR] scrape exception for {url}: {exc}")
                data = {"name": None, "ingredients": None}

            scrape_attempts += 1

            if data.get("name"):
                _update_progress(progress, url, data.get("name"), data.get("ingredients"), "scraped")
                print(f"[OK] {url}")
            else:
                _update_progress(progress, url, "", "", "skipped")
                print(f"[SKIP] {url}")
                try:
                    driver.back()
                    time.sleep(BACK_BUTTON_DELAY)
                except Exception:
                    pass
                _restart_driver(driver_ref, cfg, reason=f"error on {url}")

            _write_output(urls, progress)
            _maybe_wait(scrape_attempts)

        scrape_attempts = _retry_missing(driver_ref, cfg, urls, progress, scrape_attempts)

        stats = {"scraped": 0, "cached": 0, "skipped": 0}
        for url in urls:
            record = progress.get(url, {})
            status = (record.get("status") or "").lower()
            if status == "scraped":
                stats["scraped"] += 1
            elif status == "skipped":
                stats["skipped"] += 1
            elif record.get("name"):
                stats["cached"] += 1
            else:
                stats["skipped"] += 1

        _write_output(urls, progress)
        print(f"[DONE] Wrote {len(urls)} rows to {OUTPUT_FILE.name}")
        print(f"[SUMMARY] scraped={stats['scraped']} cached={stats['cached']} skipped={stats['skipped']}")
    finally:
        driver = driver_ref.get("driver")
        if driver is not None:
            try:
                driver.quit()
            except Exception:
                pass


if __name__ == "__main__":
    main()
