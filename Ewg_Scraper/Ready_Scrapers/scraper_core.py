import os
import re
import time
import math
import random
import subprocess
from typing import Dict, List, Optional, Tuple

import pandas as pd
from pathlib import Path
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
from webdriver_manager.chrome import ChromeDriverManager


BASE_URL = "https://www.ewg.org/skindeep"


def _env_bool(name: str, default: bool) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return str(v).lower() in ("1", "true", "yes", "y", "on")


def get_config() -> Dict:
    return {
        # browser/session
        "BROWSER": os.getenv("BROWSER", "chrome").lower(),
        "HEADLESS": _env_bool("HEADLESS", False),
        "USER_DATA_DIR": os.getenv("USER_DATA_DIR"),
        "PROFILE_DIR": os.getenv("PROFILE_DIR", "Default"),
        # timing
        "MIN_DELAY": float(os.getenv("MIN_DELAY", "0.3")),
        "MAX_DELAY": float(os.getenv("MAX_DELAY", "0.9")),
        "PAGELOAD_TIMEOUT": int(os.getenv("PAGELOAD_TIMEOUT", "35")),
        "WAIT_FOR_BODY_TIMEOUT": int(os.getenv("WAIT_FOR_BODY_TIMEOUT", "8")),
        "RETRIES_PER_PRODUCT": int(os.getenv("RETRIES_PER_PRODUCT", "2")),
        "PRODUCT_HARD_TIMEOUT": int(os.getenv("PRODUCT_HARD_TIMEOUT", "10")),
        # cooldowns
        "COOLDOWN_AFTER_CONSECUTIVE_PRODUCT_ERRORS": int(os.getenv("COOLDOWN_AFTER_ERRORS", "60")),
        "GLOBAL_COOLDOWN_EVERY_N_SAVES": int(os.getenv("GLOBAL_COOLDOWN_EVERY_N", "50")),
        "GLOBAL_COOLDOWN_SECONDS": int(os.getenv("GLOBAL_COOLDOWN_SECONDS", "90")),
        # paging (batch size per listing fetch; not a hard cap)
        "PAGE_BATCH_SIZE": int(os.getenv("PAGE_BATCH_SIZE", "1000")),
        # url cache
        "USE_URL_CACHE": _env_bool("USE_URL_CACHE", True),
        "REFRESH_URL_CACHE": _env_bool("REFRESH_URL_CACHE", False),
    }


def human_delay(cfg: Dict, a: Optional[float] = None, b: Optional[float] = None):
    lo = cfg["MIN_DELAY"] if a is None else a
    hi = cfg["MAX_DELAY"] if b is None else b
    time.sleep(random.uniform(lo, hi))


def human_scroll(driver, cfg: Dict):
    try:
        height = driver.execute_script(
            "return Math.max(document.body.scrollHeight, document.documentElement.scrollHeight);"
        ) or 2000
        steps = random.randint(3, 5)
        for i in range(steps):
            y = int(height * ((i + 1) / (steps + 1)))
            driver.execute_script(f"window.scrollTo({{top:{y}, behavior:'smooth'}});")
            human_delay(cfg, 0.3, 0.8)
        driver.execute_script("window.scrollTo({top:0, behavior:'smooth'});")
    except Exception:
        pass


def _first_env_path(*names: str) -> Optional[str]:
    for name in names:
        val = os.getenv(name)
        if not val:
            continue
        candidate = os.path.abspath(os.path.expanduser(val.strip()))
        if candidate:
            return candidate
    return None


def create_chrome_driver(cfg: Dict):
    options = webdriver.ChromeOptions()
    if cfg["HEADLESS"]:
        options.add_argument('--headless=new')
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    options.add_argument('--disable-gpu')
    options.add_argument('--disable-features=NetworkService')
    options.add_argument('--disable-notifications')
    options.add_argument('--no-first-run')
    options.add_argument('--no-default-browser-check')
    options.add_argument('--disable-extensions')
    options.add_argument('--disable-blink-features=AutomationControlled')
    options.add_experimental_option('excludeSwitches', ['enable-automation', 'enable-logging'])
    options.add_experimental_option('useAutomationExtension', False)
    options.add_argument('user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36')
    options.page_load_strategy = 'eager'

    if cfg.get("USER_DATA_DIR"):
        options.add_argument(f"--user-data-dir={cfg['USER_DATA_DIR']}")
        if cfg.get("PROFILE_DIR"):
            options.add_argument(f"--profile-directory={cfg['PROFILE_DIR']}")

    chrome_binary = _first_env_path('CHROME_BINARY', 'GOOGLE_CHROME_SHIM', 'CHROME_PATH')
    if chrome_binary:
        options.binary_location = chrome_binary

    driver_candidate = _first_env_path('CHROME_DRIVER_PATH', 'CHROMEDRIVER', 'WEBDRIVER_CHROME_DRIVER')
    service = None
    if driver_candidate:
        if os.path.isdir(driver_candidate):
            expected = os.path.join(driver_candidate, 'chromedriver')
            if os.path.exists(expected):
                driver_candidate = expected
        if os.path.exists(driver_candidate):
            service = ChromeService(driver_candidate, log_output=subprocess.DEVNULL)
        else:
            print(f"[WARN] Driver path set but not found: {driver_candidate}. Falling back to auto download.")

    if service is None:
        service = ChromeService(ChromeDriverManager().install(), log_output=subprocess.DEVNULL)

    drv = webdriver.Chrome(service=service, options=options)
    drv.set_page_load_timeout(cfg["PAGELOAD_TIMEOUT"])
    drv.implicitly_wait(4)
    try:
        drv.execute_cdp_cmd(
            'Page.addScriptToEvaluateOnNewDocument',
            {'source': 'Object.defineProperty(navigator, "webdriver", {get: () => undefined});'}
        )
    except Exception:
        pass
    return drv


def create_driver(cfg: Dict):
    # Opera fallback → Chrome (Selenium 4 removed official Opera integration)
    return create_chrome_driver(cfg)


def _sanitize_filename(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", text)


def _cache_dir() -> Path:
    base = Path(__file__).parent / "url_cache"
    base.mkdir(parents=True, exist_ok=True)
    return base


def _cache_file_for(category_name: str, start_page: int, end_page: Optional[int]) -> Path:
    end_str = str(end_page) if end_page is not None else "end"
    fname = f"{_sanitize_filename(category_name)}_{start_page}..{end_str}_urls.csv"
    return _cache_dir() / fname


def load_url_cache(category_name: str, start_page: int, end_page: Optional[int]) -> Tuple[List[str], Dict[str, str]]:
    p = _cache_file_for(category_name, start_page, end_page)
    if not p.exists():
        return [], {}
    try:
        df = pd.read_csv(p)
        urls = df.get("url", pd.Series([], dtype=str)).dropna().tolist()
        names_series = df.get("name", pd.Series([], dtype=str))
        names: Dict[str, str] = {}
        if names_series is not None:
            for u, n in zip(df.get("url"), names_series):
                if isinstance(u, str) and isinstance(n, str) and u:
                    names[u] = n
        print(f"[CACHE] Loaded {len(urls)} URLs from {p.name}")
        return urls, names
    except Exception as e:
        print(f"[CACHE] Failed to read cache {p.name}: {e}")
        return [], {}


def save_url_cache(category_name: str, start_page: int, end_page: Optional[int], urls: List[str], names: Dict[str, str]):
    p = _cache_file_for(category_name, start_page, end_page)
    try:
        rows = []
        for u in urls:
            rows.append({"url": u, "name": names.get(u, "")})
        df = pd.DataFrame(rows)
        df.to_csv(p, index=False, encoding="utf-8")
        print(f"[CACHE] Saved {len(urls)} URLs to {p.name}")
    except Exception as e:
        print(f"[CACHE] Failed to save cache {p.name}: {e}")


def restart_session(cfg: Dict, driver_ref: Dict, cooldown_sec: Optional[int] = None):
    cooldown = cooldown_sec if cooldown_sec is not None else cfg["COOLDOWN_AFTER_CONSECUTIVE_PRODUCT_ERRORS"]
    try:
        if driver_ref.get("driver"):
            driver_ref["driver"].quit()
    except Exception:
        pass
    print(f"[INFO] Restarting session. Waiting {cooldown}s...")
    time.sleep(cooldown)
    driver_ref["driver"] = create_driver(cfg)


def get_product_links_batch(driver, cfg: Dict, category_slug: str, start_page: int, page_batch_size: int = 10, end_page: Optional[int] = None) -> Tuple[List[str], int, Dict[str, str]]:
    product_links: List[str] = []
    link_names: Dict[str, str] = {}
    page = start_page
    max_page = start_page + page_batch_size - 1
    if end_page is not None:
        max_page = min(max_page, end_page)

    consecutive_page_errors = 0

    empty_pages = 0
    while page <= max_page:
        category_display = category_slug.replace("_", " ").replace("  ", "/")
        url = f"{BASE_URL}/browse/category/{category_slug}/?category={category_display}&page={page}"
        print(f"  Page {page} scanning...")
        try:
            try:
                driver.get(url)
            except TimeoutException:
                pass
            # Guard against stuck 'data:' or wrong location
            cur = driver.current_url or ""
            if cur.startswith("data:") or cur.startswith("about:") or "ewg.org" not in cur:
                try:
                    driver.execute_script("window.location.href=arguments[0];", url)
                    time.sleep(0.5)
                except Exception:
                    pass
            WebDriverWait(driver, cfg["WAIT_FOR_BODY_TIMEOUT"]).until(EC.presence_of_element_located((By.TAG_NAME, "body")))
            human_delay(cfg, 0.5, 1.0)
            human_scroll(driver, cfg)
            consecutive_page_errors = 0
        except Exception as e:
            consecutive_page_errors += 1
            print(f"  [WARN] Page {page} failed, skipping: {e} (consecutive: {consecutive_page_errors})")
            if consecutive_page_errors >= 3:
                print("  [INFO] 3+ consecutive page errors. Restarting session.")
                restart_session(cfg, {"driver": driver})
                consecutive_page_errors = 0
            page += 1
            continue

        anchors = driver.find_elements(By.CSS_SELECTOR, "a[href*='/skindeep/products/']")
        if not anchors:
            # Sometimes a page can fail transiently; tolerate a few empties
            empty_pages += 1
            if empty_pages >= 2:
                print(f"  No products on page {page} (empty {empty_pages}x). Batch complete.")
                break
            print(f"  No products on page {page}. Advancing to next page (tolerating transient empty).")
            page += 1
            continue
        else:
            empty_pages = 0

        page_count = 0
        for a in anchors:
            href = a.get_attribute("href")
            if href and "/skindeep/products/" in href and href not in product_links:
                product_links.append(href)
                page_count += 1
                try:
                    txt = a.text.strip()
                    if txt:
                        link_names[href] = txt
                except Exception:
                    pass

        print(f"  Page {page} found {page_count} products (batch total: {len(product_links)})")
        if page_count == 0:
            print(f"  No new products on page {page}. Batch complete.")
            break
        page += 1

    return product_links, page, link_names


def collect_all_urls(driver, cfg: Dict, category_name: str, start_page: int, end_page: Optional[int], page_batch_size: int) -> Tuple[List[str], Dict[str, str]]:
    all_urls: List[str] = []
    names: Dict[str, str] = {}
    cur = start_page
    while end_page is None or cur <= end_page:
        batch_end = cur + page_batch_size - 1 if end_page is None else min(cur + page_batch_size - 1, end_page)
        print(f"\n--- Collect URLs: Page {cur} - {batch_end} ---")
        urls, next_page, link_names = get_product_links_batch(driver, cfg, category_name, cur, page_batch_size, end_page)
        for u in urls:
            if u not in all_urls:
                all_urls.append(u)
        names.update(link_names)
        if not urls:
            break
        cur = next_page
        if end_page is None and next_page == cur:
            break
        if end_page is not None and cur > end_page:
            break
    return all_urls, names


def scrape_product(driver, cfg: Dict, url: str) -> Dict[str, Optional[str]]:
    start = time.time()
    for attempt in range(1, cfg["RETRIES_PER_PRODUCT"] + 1):
        try:
            try:
                driver.get(url)
            except TimeoutException:
                pass
            cur = driver.current_url or ""
            if cur.startswith("data:") or cur.startswith("about:") or "ewg.org" not in cur:
                try:
                    driver.execute_script("window.location.href=arguments[0];", url)
                    time.sleep(0.5)
                except Exception:
                    pass
            WebDriverWait(driver, cfg["WAIT_FOR_BODY_TIMEOUT"]).until(EC.presence_of_element_located((By.TAG_NAME, "body")))
            human_delay(cfg, 0.4, 1.1)
            human_scroll(driver, cfg)

            soup = BeautifulSoup(driver.page_source, 'html.parser')
            h1_elems = soup.select("h1")
            if len(h1_elems) > 1:
                name_elem = h1_elems[1]
            else:
                name_elem = soup.select_one("h1") or soup.select_one("h2")
            name = name_elem.get_text(strip=True) if name_elem else "Unknown Product"
            if name.startswith("##"):
                name = name.lstrip("# ").strip()

            ingredients_text = "Unknown Ingredients"
            ingredients_elem = soup.find(string=lambda x: x and 'Ingredients from' in x)
            if ingredients_elem:
                parent = ingredients_elem.find_parent()
                next_elem = parent.find_next_sibling() if parent else None
                if next_elem:
                    ingredients_text = next_elem.get_text(" ", strip=True).replace("\n", " ").strip()

            return {'name': name, 'ingredients': ingredients_text}
        except Exception as e:
            print(f"[RETRY] Attempt {attempt}/{cfg['RETRIES_PER_PRODUCT']} - Error {url}: {str(e)[:100]}")
            if attempt < cfg["RETRIES_PER_PRODUCT"]:
                time.sleep(min(3, int(math.pow(2, attempt))))
            continue
        finally:
            if time.time() - start > cfg["PRODUCT_HARD_TIMEOUT"]:
                print(f"[SKIP] Per-product time budget exceeded (> {cfg['PRODUCT_HARD_TIMEOUT']}s): {url}")
                return {'name': None, 'ingredients': None}
    return {'name': None, 'ingredients': None}


def safe_print_saved(id_counter: int, product_name: Optional[str]):
    try:
        print(f"Saved: ID {id_counter} - {product_name}")
    except UnicodeEncodeError:
        safe_name = (product_name or '').encode('ascii', 'ignore').decode('ascii')
        print(f"Saved: ID {id_counter} - {safe_name}")


def press_back_with_pause(driver, cfg: Dict):
    try:
        print("[ACTION] After 10 saves: wait 5s + driver.back()")
        time.sleep(5)
        driver.back()
        human_delay(cfg, 0.4, 0.9)
    except Exception:
        pass


def run_category(category_name: str, start_page: int, end_page: Optional[int], output_file: str, page_batch_size: int = 10):
    cfg = get_config()
    driver = create_driver(cfg)

    id_counter = 1
    scraped_urls = set()
    saved_count = 0
    skipped_duplicates = 0
    skipped_errors = 0

    if os.path.exists(output_file):
        try:
            existing_df = pd.read_csv(output_file)
            if len(existing_df) > 0:
                id_counter = int(existing_df['id'].max()) + 1 if 'id' in existing_df.columns else len(existing_df) + 1
            if 'product_url' in existing_df.columns:
                scraped_urls = set(existing_df['product_url'].dropna().tolist())
        except Exception as e:
            print(f"[WARN] CSV read error: {e}. Starting fresh counters.")
            id_counter = 1
            scraped_urls = set()

    # PHASE 1: Liste sayfalarını teker teker tarayıp ürün URL'lerini topla
    print(f"\n=== PHASE 1: Listing pages for {category_name} ({start_page}..{end_page or 'end'}) ===")
    # URL önbelleğini her durumda kontrol et (eşik yeterliyse taramayı atlamak için)
    all_urls: List[str] = []
    link_name_map: Dict[str, str] = {}
    cached_urls: List[str] = []
    cached_names: Dict[str, str] = {}
    if cfg["USE_URL_CACHE"]:
        cached_urls, cached_names = load_url_cache(category_name, start_page, end_page)

    # Eşik: (end_page - 1) * 12 adet URL varsa, liste taramasını atla
    expected_min = None
    if end_page is not None:
        try:
            expected_min = max(0, (int(end_page) - 1) * 12)
        except Exception:
            expected_min = None

    if expected_min and len(cached_urls) >= expected_min:
        print(f"[SKIP] URL crawl skipped (cached >= expected_min {expected_min}): cached={len(cached_urls)}")
        all_urls = cached_urls
        link_name_map = cached_names
    else:
        # Eğer önbellek var ve REFRESH_URL_CACHE kapalıysa önbelleği kullan; değilse yeniden topla
        if cached_urls and cfg["USE_URL_CACHE"] and not cfg["REFRESH_URL_CACHE"]:
            all_urls = cached_urls
            link_name_map = cached_names
            print(f"[CACHE] Using cached URLs: {len(all_urls)}")
        else:
            all_urls, link_name_map = collect_all_urls(driver, cfg, category_name, start_page, end_page, page_batch_size)
            if cfg["USE_URL_CACHE"]:
                save_url_cache(category_name, start_page, end_page, all_urls, link_name_map)
    pages_scanned = (end_page - start_page + 1) if end_page is not None else None
    print(f"[INFO] Collected URLs: {len(all_urls)}")
    print(f"=== PHASE 1 DONE: {category_name} ===")

    consecutive_product_failures = 0

    print(f"\n=== PHASE 2: Scraping products for {category_name} (total URLs: {len(all_urls)}) ===")
    for url in all_urls:
        if url in scraped_urls:
            print(f"[SKIP] Already scraped: {url}")
            skipped_duplicates += 1
            continue

        product_data = scrape_product(driver, cfg, url)
        if product_data['name'] is None:
            print(f"[WARN] SKIPPED (bad product): {url}")
            skipped_errors += 1
            consecutive_product_failures += 1
            if consecutive_product_failures == 2:
                print(f"[COOLDOWN] Two consecutive product errors. Waiting {cfg['COOLDOWN_AFTER_CONSECUTIVE_PRODUCT_ERRORS']}s...")
                time.sleep(cfg['COOLDOWN_AFTER_CONSECUTIVE_PRODUCT_ERRORS'])
            elif consecutive_product_failures > 2:
                print("[INFO] More errors after cooldown. Restarting session.")
                restart_session(cfg, {"driver": driver})
                consecutive_product_failures = 0
            continue
        else:
            consecutive_product_failures = 0

        df_row = pd.DataFrame([{
            "id": id_counter,
            "category": category_name,
            "product_url": url,
            "name": product_data['name'],
            "ingredients": product_data['ingredients']
        }])
        header = not os.path.exists(output_file)
        df_row.to_csv(output_file, mode='a', header=header, index=False, encoding='utf-8')
        scraped_urls.add(url)
        safe_print_saved(id_counter, product_data.get('name', 'N/A'))
        id_counter += 1
        saved_count += 1

        if saved_count % 10 == 0:
            press_back_with_pause(driver, cfg)

        if cfg['GLOBAL_COOLDOWN_EVERY_N_SAVES'] > 0 and saved_count % cfg['GLOBAL_COOLDOWN_EVERY_N_SAVES'] == 0:
            print(f"[COOLDOWN] After {saved_count} saves, waiting {cfg['GLOBAL_COOLDOWN_SECONDS']}s...")
            time.sleep(cfg['GLOBAL_COOLDOWN_SECONDS'])

    print("\n--- SECOND PASS: Retry missing URLs ---")
    # Use cache if available to avoid re-listing
    recollect_urls: List[str]
    recollect_names: Dict[str, str]
    if cfg["USE_URL_CACHE"] and not cfg["REFRESH_URL_CACHE"]:
        recollect_urls, recollect_names = load_url_cache(category_name, start_page, end_page)
        if not recollect_urls:
            recollect_urls, recollect_names = all_urls, link_name_map
    else:
        recollect_urls, recollect_names = collect_all_urls(driver, cfg, category_name, start_page, end_page, page_batch_size)
        if cfg["USE_URL_CACHE"]:
            # merge and save updated cache
            merged = list(dict.fromkeys(all_urls + recollect_urls).keys())
            merged_names = {**link_name_map, **recollect_names}
            save_url_cache(category_name, start_page, end_page, merged, merged_names)
    missing = [u for u in recollect_urls if u not in scraped_urls]
    print(f"[INFO] Missing URL count: {len(missing)}")
    for url in missing:
        product_data = scrape_product(driver, cfg, url)
        if product_data['name'] is None:
            skipped_errors += 1
            continue
        df_row = pd.DataFrame([{
            "id": id_counter,
            "category": category_name,
            "product_url": url,
            "name": product_data['name'],
            "ingredients": product_data['ingredients']
        }])
        header = not os.path.exists(output_file)
        df_row.to_csv(output_file, mode='a', header=header, index=False, encoding='utf-8')
        scraped_urls.add(url)
        safe_print_saved(id_counter, product_data.get('name', 'N/A'))
        id_counter += 1
        saved_count += 1

    print("\n--- RECONCILIATION: Ensure all collected URLs exist in CSV ---")
    try:
        final_df = pd.read_csv(output_file)
        csv_urls = set(final_df.get('product_url', pd.Series([], dtype=str)).dropna().tolist())
    except Exception:
        csv_urls = set()
    requeue = [u for u in recollect_urls if u not in csv_urls]
    print(f"[INFO] Not in CSV after passes: {len(requeue)}")
    for url in requeue:
        product_data = scrape_product(driver, cfg, url)
        if product_data['name'] is None:
            continue
        df_row = pd.DataFrame([{
            "id": id_counter,
            "category": category_name,
            "product_url": url,
            "name": product_data['name'],
            "ingredients": product_data['ingredients']
        }])
        header = not os.path.exists(output_file)
        df_row.to_csv(output_file, mode='a', header=header, index=False, encoding='utf-8')
        scraped_urls.add(url)
        safe_print_saved(id_counter, product_data.get('name', 'N/A'))
        id_counter += 1
        saved_count += 1

    try:
        # Eğer kullanıcı tarayıcıyı açık bırakmak istiyorsa kapatma
        keep_open = os.getenv("KEEP_BROWSER_OPEN", "false").lower() in ("1","true","yes","y","on")
        if not keep_open:
            try:
                driver.close()
            except Exception:
                pass
            try:
                driver.quit()
            except Exception:
                pass
        else:
            print("[INFO] KEEP_BROWSER_OPEN enabled; driver not closed.")
    except Exception:
        pass

    print("\n===== SUMMARY =====")
    pages_txt = f"{start_page} .. {end_page}" if end_page is not None else f"{start_page} .. end"
    print(f"Pages: {pages_txt}")
    print(f"Saved: {saved_count}")
    print(f"Output: {output_file}")


def run_scraper_main(category_name: str, default_start_page: int = 1, default_end_page: Optional[int] = None, default_output_file: Optional[str] = None):
    # Varsayılan olarak URL cache kullanımını açık bırak, tazelemeyi kullanıcı belirlesin
    if os.getenv("USE_URL_CACHE") is None:
        os.environ["USE_URL_CACHE"] = "true"
    cfg = get_config()
    start_page = int(os.getenv("START_PAGE", str(default_start_page)))
    end_env = os.getenv("END_PAGE")
    end_page = int(end_env) if end_env is not None else default_end_page
    output_file = default_output_file or f"{category_name}.csv"
    page_batch_size = cfg["PAGE_BATCH_SIZE"]

    run_category(category_name, start_page, end_page, output_file, page_batch_size)
