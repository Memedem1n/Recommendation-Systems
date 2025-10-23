import time
import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
import os  # Dosya varlığını kontrol etmek için os modülünü ekledik

BASE_URL = "https://www.ewg.org/skindeep"
CATEGORIES = [
    # {"name": "Anti-aging", "product_count": 500},# 1 kişi
    # {"name": "Around-eye_cream", "product_count": 1000}, # 1 kişi
    # {"name": "BB_Cream", "product_count": 250}, #1 kişi
    {"name": "CC_Cream", "product_count": 250}, #1 kişi
    # {"name": "Facial_cleanser", "product_count": 3200}, # 3 kişi
    # {"name": "Facial_moisturizer__treatment", "product_count": 5000},  # 5 kişi
    # {"name": "Makeup_remover", "product_count": 600}, # 1 kişi
    # {"name": "Mask", "product_count": 3000}, # 3 kişi
    # {"name": "Oil_controller", "product_count": 50}, # 1 kişi
    # {"name": "Pore_strips", "product_count": 60},   # 1 kişi
    # {"name": "Serums_&_Essences", "product_count": 2500}, # 2 kişi
    # {"name": "Skin_fading__lightener", "product_count": 50}, # 1 kişi
    # {"name": "Toners__astringents", "product_count": 1000} # 1 kişi
]

# Selenium tarayıcı başlat - ULTRA AGGRESSIVE MODE
options = webdriver.ChromeOptions()
options.add_argument('--headless')
options.add_argument('--no-sandbox')
options.add_argument('--disable-dev-shm-usage')
options.add_argument('user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36')

# 🚀 PERFORMANS BOOST: Gereksiz yükleri kapat
options.add_argument('--disable-blink-features=AutomationControlled')
options.add_argument('--disable-extensions')
options.add_argument('--disable-plugins')
options.add_argument('--disable-images')  # RESİMLERİ YÜKLEME
options.add_argument('--blink-settings=imagesEnabled=false')
options.add_experimental_option("prefs", {
    "profile.managed_default_content_settings.images": 2,  # Resimleri engelle
    "profile.default_content_setting_values.notifications": 2,  # Bildirimleri engelle
    "profile.managed_default_content_settings.stylesheets": 2,  # CSS'i engelle (opsiyonel)
})

# 🧠 MEMORY OPTIMIZATION
options.add_argument('--disable-gpu')
options.add_argument('--disable-software-rasterizer')
options.add_argument('--disable-background-networking')
options.add_argument('--disable-default-apps')
options.add_argument('--disable-sync')
options.add_argument('--metrics-recording-only')
options.add_argument('--mute-audio')

# ⚡ PAGE LOAD STRATEGY: Normal yerine 'eager' kullan (DOM ready olunca devam et)
options.page_load_strategy = 'eager'  # 'normal' yerine 'eager' - DOM ready'de devam eder

driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

# ⏱️ ULTRA FAST MODE: Minimum timeout (emergency stop var artık)
driver.set_page_load_timeout(20)  # Emergency stop var, 20 saniye yeter
driver.implicitly_wait(3)  # 3 saniye yeterli


def get_product_links_batch(category_slug, start_page, page_batch_size=10, already_collected=0, limit=250):
    """Belirli sayıda sayfayı tarayıp URL'leri toplar"""
    product_links = []
    page = start_page
    max_page = start_page + page_batch_size

    while page < max_page and (already_collected + len(product_links)) < limit:
        # Sayfa numarası ile URL oluştur
        category_display = category_slug.replace("_", " ").replace("  ", "/")
        url = f"{BASE_URL}/browse/category/{category_slug}/?category={category_display}&page={page}"

        print(f"  Sayfa {page} taranıyor...")
        
        # Retry mekanizması ile sayfa yükleme
        page_loaded = False
        for retry_attempt in range(2):  # 2 deneme yeter (hız için)
            try:
                driver.get(url)
                time.sleep(0.8)  # ⚡ 2.5 → 0.8 saniye (eager mode var)
                page_loaded = True
                break
            except Exception as e:
                if retry_attempt < 1:  # Son denemede değilse
                    print(f"  ⚠️ Sayfa {page} yükleme hatası (Deneme {retry_attempt + 1}/2), tekrar deneniyor...")
                    time.sleep(2)  # ⚡ 5 → 2 saniye
                else:
                    print(f"  ❌ Sayfa {page} 2 denemede de yüklenemedi, atlanıyor: {str(e)[:100]}")
        
        # Sayfa yüklenemediyse bir sonraki sayfaya geç
        if not page_loaded:
            page += 1
            continue

        products = driver.find_elements(By.CSS_SELECTOR, "a[href*='/skindeep/products/']")

        # Eğer sayfada ürün yoksa, son sayfaya ulaşmışız demektir
        if not products:
            print(f"  Sayfa {page}'de ürün bulunamadı. Batch taraması tamamlandı.")
            break

        page_product_count = 0
        for elem in products:
            href = elem.get_attribute("href")
            if href and "/skindeep/products/" in href and href not in product_links:
                product_links.append(href)
                page_product_count += 1
                if (already_collected + len(product_links)) >= limit:
                    break

        print(f"  Sayfa {page}'den {page_product_count} ürün bulundu (Batch Toplam: {len(product_links)})")

        # Eğer bu sayfadan yeni ürün eklenmediyse, döngüden çık
        if page_product_count == 0:
            print(f"  Sayfa {page}'de yeni ürün bulunamadı. Batch taraması tamamlandı.")
            break

        page += 1

    return product_links, page


def scrape_product(url, retries=2):
    """ULTRA FAST MODE: Minimum bekleme, emergency stop"""
    for attempt in range(1, retries + 1):
        try:
            # 🚨 CRITICAL FIX: Page load timeout'tan önce emergency stop
            # Timeout'u geçici olarak 10 saniyeye düşür
            driver.set_page_load_timeout(10)
            
            try:
                driver.get(url)
                time.sleep(0.3)  # DOM parse için minimal bekleme
            except Exception:
                # Timeout alındıysa, yüklemeyi durdur ve mevcut HTML'i al
                try:
                    driver.execute_script("window.stop();")
                    print(f"  🛑 Timeout! Sayfa zorla durduruldu: {url[:80]}...")
                    time.sleep(0.5)
                except Exception:
                    pass
            
            # Timeout'u geri yükselt
            driver.set_page_load_timeout(20)
            
            soup = BeautifulSoup(driver.page_source, 'html.parser')

            # ✅ Ürün adı: Tüm h1'leri çek, ikincisini al (ilk "Advanced Search" oluyor)
            h1_elems = soup.select("h1")
            if len(h1_elems) > 1:
                name_elem = h1_elems[1]  # İkinci h1 gerçek ürün adı
            else:
                name_elem = soup.select_one("h1")  # Eğer sadece bir tane varsa onu al
            if not name_elem:
                name_elem = soup.select_one("h2")  # Fallback h2

            if name_elem:
                name_text = name_elem.get_text(strip=True)
                # "## " ile başlıyorsa temizle
                if name_text.startswith("##"):
                    name_text = name_text.lstrip("# ").strip()
                name = name_text
            else:
                name = "Bilinmeyen Ürün"

            # Ingredients from label or packaging (genel arama için 'Ingredients from' kullan)
            ingredients_text = "Bilinmeyen Bileşenler"
            ingredients_elem = soup.find(string=lambda x: x and 'Ingredients from' in x)
            if ingredients_elem:
                next_elem = ingredients_elem.find_parent().find_next_sibling()
                if next_elem:
                    ingredients_text = next_elem.get_text(" ", strip=True).replace("\n", " ").strip()  # Ekstra temizlik

            return {
                'name': name,
                'ingredients': ingredients_text
            }
        except Exception as e:
            print(f"⚡ Deneme {attempt}/{retries} - Hata {url}: {str(e)[:100]}")
            if attempt < retries:
                time.sleep(2)  # ⚡ 5 → 2 saniye (hız için)
            continue
    return {'name': None, 'ingredients': None}


def restart_driver():
    """Driver'ı yeniden başlat - memory leak'i önlemek için"""
    global driver
    try:
        driver.quit()
    except Exception:
        pass
    
    # Yeni driver oluştur
    options = webdriver.ChromeOptions()
    options.add_argument('--headless')
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    options.add_argument('user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36')
    options.add_argument('--disable-blink-features=AutomationControlled')
    options.add_argument('--disable-extensions')
    options.add_argument('--disable-plugins')
    options.add_argument('--disable-images')
    options.add_argument('--blink-settings=imagesEnabled=false')
    options.add_experimental_option("prefs", {
        "profile.managed_default_content_settings.images": 2,
        "profile.default_content_setting_values.notifications": 2,
        "profile.managed_default_content_settings.stylesheets": 2,
    })
    options.add_argument('--disable-gpu')
    options.add_argument('--disable-software-rasterizer')
    options.add_argument('--disable-background-networking')
    options.add_argument('--disable-default-apps')
    options.add_argument('--disable-sync')
    options.add_argument('--metrics-recording-only')
    options.add_argument('--mute-audio')
    options.page_load_strategy = 'eager'
    
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    driver.set_page_load_timeout(20)  # ⚡ 30 → 20
    driver.implicitly_wait(3)  # ⚡ 8 → 3
    
    return driver


if __name__ == "__main__":
    # 🔧 FIX: CSV dosyasını script ile aynı dizinde oluştur
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_file = os.path.join(script_dir, "CC_Cream.csv")
    
    PAGE_BATCH_SIZE = 15  # ⚡ 10 → 15 sayfa (daha az restart = daha hızlı)
    
    # ✅ RESUME CAPABILITY: Eğer CSV varsa, kaldığı yerden devam et
    id_counter = 1
    existing_urls = set()
    resume_enabled = False
    
    if os.path.exists(output_file):
        try:
            existing_df = pd.read_csv(output_file)
            if not existing_df.empty:
                id_counter = existing_df['id'].max() + 1
                existing_urls = set(existing_df['product_url'].tolist())
                resume_enabled = True
                print("\n" + "="*70)
                print("📂 RESUME MODE AKTIF!")
                print(f"   Mevcut CSV: {output_file}")
                print(f"   Toplam ürün: {len(existing_df)}")
                print(f"   Son ID: {id_counter - 1}")
                print(f"   Yeni ID: {id_counter} (devam edecek)")
                print(f"   {len(existing_urls)} URL duplicate kontrolünde")
                print("="*70 + "\n")
        except Exception as e:
            print(f"⚠️ CSV okuma hatası: {e}. Sıfırdan başlanıyor.")
            resume_enabled = False
    else:
        print(f"\n📝 Yeni CSV oluşturulacak: {output_file}\n")

    for category in CATEGORIES:
        category_name = category["name"]
        product_count = category["product_count"]
        print(f"Kategori işleniyor: {category_name} ({product_count} ürün)")

        current_page = 1
        
        # 🔧 RESUME: Eğer zaten ürün varsa, total_collected'ı ayarla
        if resume_enabled and len(existing_urls) > 0:
            total_collected = len(existing_urls)
            print(f"🔄 Resume Mode: Mevcut {total_collected} ürün, hedef {product_count} ürün")
        else:
            total_collected = 0
        
        # Batch batch işle: her seferde PAGE_BATCH_SIZE sayfa tara, scrape et, kaydet
        while total_collected < product_count:
            print(f"\n⚡ ULTRA FAST BATCH: Sayfa {current_page} - {current_page + PAGE_BATCH_SIZE - 1} ⚡")
            
            # Bu batch'teki URL'leri topla
            product_urls, next_page = get_product_links_batch(
                category_name, 
                start_page=current_page, 
                page_batch_size=PAGE_BATCH_SIZE,
                already_collected=total_collected,
                limit=product_count
            )
            
            # Eğer URL bulunamadıysa, işlem tamamlanmış demektir
            if not product_urls:
                print("Daha fazla ürün bulunamadı. Kategori tamamlandı.")
                break
            
            # Bu batch'teki URL'leri hemen scrape et ve kaydet
            skipped_count = 0
            for url in product_urls:
                # 🔍 Duplicate kontrolü: Bu URL zaten scrape edildiyse atla
                if url in existing_urls:
                    skipped_count += 1
                    continue  # Sayıma dahil etme, sadece skip et
                
                product_data = scrape_product(url)

                # ❌ Eğer ürün bilgisi alınamadıysa (name=None), atla ve kaydetme
                if product_data['name'] is None:
                    print(f"⚠️ ATLANDI (Hatalı ürün): {url}")
                    continue

                # Anlık olarak DataFrame oluştur
                df_row = pd.DataFrame([{
                    "id": id_counter,
                    "category": category_name,
                    # "product_count": product_count,
                    "product_url": url,
                    "name": product_data['name'],
                    "ingredients": product_data['ingredients']
                }])

                # Dosyanın var olup olmadığını kontrol ederek başlık (header) eklenip eklenmeyeceğine karar ver
                # Dosya yoksa header=True, varsa header=False olacak
                header = not os.path.exists(output_file)

                # 'append' modunda ('a') dosyaya ekle
                df_row.to_csv(output_file, mode='a', header=header, index=False, encoding="utf-8")

                print(f"Kaydedildi: ID {id_counter} - {product_data.get('name', 'Hata!')}")

                # URL'i hafızaya ekle (duplicate önlemi)
                existing_urls.add(url)
                
                id_counter += 1
                total_collected += 1
                
                # Eğer limit'e ulaştıysak dur
                if total_collected >= product_count:
                    break
            
            # Bir sonraki batch için sayfa numarasını güncelle
            current_page = next_page
            
            # Batch özeti
            if skipped_count > 0:
                print(f"\n✅ Batch tamamlandı! {total_collected}/{product_count} ürün | ⏭️ {skipped_count} duplicate skip")
            else:
                print(f"\n✅ Batch tamamlandı! {total_collected}/{product_count} ürün")
            
            # 🔄 HER BATCH SONRASI DRIVER'I RESTART ET (Memory leak önlemi)
            if total_collected < product_count:
                print("🔄 Driver yeniden başlatılıyor (memory temizliği)...")
                driver = restart_driver()
                time.sleep(0.5)  # ⚡ 2 → 0.5 saniye

    # Tarayıcıyı kapat
    driver.quit()

    print(f"\nScraping tamamlandı. Sonuç {output_file} dosyasına yazıldı.")

    # Sonucun önizlemesini göstermek için dosyayı oku
    try:
        final_df = pd.read_csv(output_file)
        print("Veri önizlemesi:")
        print(final_df.head())
    except pd.errors.EmptyDataError:
        print("CSV dosyası boş, hiçbir veri kazınamadı.")