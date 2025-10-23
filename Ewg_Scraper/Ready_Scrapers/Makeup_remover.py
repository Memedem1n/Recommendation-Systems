import time
import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
import os  # Dosya varlığını kontrol etmek için os modülünü ekledik

BASE_URL = "https://www.ewg.org/skindeep"
CATEGORIES = [
    # {"name": "Anti-aging", "product_count": 500},# 1 kişi
    # {"name": "Around-eye_cream", "product_count": 1000}, # 1 kişi
    # {"name": "BB_Cream", "product_count": 250}, #1 kişi
    # {"name": "CC_Cream", "product_count": 250}, #1 kişi
    # {"name": "Facial_cleanser", "product_count": 3200}, # 3 kişi
    # {"name": "Facial_moisturizer__treatment", "product_count": 5000},  # 5 kişi
    {"name": "Makeup_remover", "product_count": 670}, # 1 kişi
    # {"name": "Mask", "product_count": 3000}, # 3 kişi
    # {"name": "Oil_controller", "product_count": 50}, # 1 kişi
    # {"name": "Pore_strips", "product_count": 60},   # 1 kişi
    # {"name": "Serums_&_Essences", "product_count": 2500}, # 2 kişi
    # {"name": "Skin_fading__lightener", "product_count": 50}, # 1 kişi
    # {"name": "Toners__astringents", "product_count": 1000} # 1 kişi
]

# Selenium tarayıcı başlat
options = webdriver.ChromeOptions()
options.add_argument('--headless')
options.add_argument('--no-sandbox')
options.add_argument('--disable-dev-shm-usage')
options.add_argument('user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36')
driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

# ⏱️ HARD MODE: Agresif timeout ayarları
driver.set_page_load_timeout(15)
driver.implicitly_wait(5)


def get_product_links_batch(category_slug, start_page, page_batch_size=10, already_collected=0, limit=670):
    """Belirli sayıda sayfayı tarayıp URL'leri toplar"""
    product_links = []
    page = start_page
    max_page = start_page + page_batch_size

    while page < max_page and (already_collected + len(product_links)) < limit:
        # Sayfa numarası ile URL oluştur
        category_display = category_slug.replace("_", " ").replace("  ", "/")
        url = f"{BASE_URL}/browse/category/{category_slug}/?category={category_display}&page={page}"

        print(f"  Sayfa {page} taranıyor...")
        try:
            driver.get(url)
            time.sleep(1.5)
        except Exception as e:
            print(f"  ⚠️ Sayfa {page} yüklenemedi, atlanıyor: {e}")
            break

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
    """HARD MODE: Hızlı scraping, minimal bekleme"""
    for attempt in range(1, retries + 1):
        try:
            driver.get(url)
            WebDriverWait(driver, 8).until(EC.presence_of_element_located((By.TAG_NAME, "body")))
            time.sleep(2)
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
                time.sleep(3)
            continue
    return {'name': None, 'ingredients': None}


if __name__ == "__main__":
    output_file = "Makeup_remover.csv"
    id_counter = 1
    PAGE_BATCH_SIZE = 10  # Her seferde 10 sayfa işle

    for category in CATEGORIES:
        category_name = category["name"]
        product_count = category["product_count"]
        print(f"Kategori işleniyor: {category_name} ({product_count} ürün)")

        current_page = 1
        total_collected = 0
        
        # Batch batch işle: her seferde PAGE_BATCH_SIZE sayfa tara, scrape et, kaydet
        while total_collected < product_count:
            print(f"\n--- Batch işleniyor: Sayfa {current_page} - {current_page + PAGE_BATCH_SIZE - 1} ---")
            
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
            for url in product_urls:
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

                id_counter += 1
                total_collected += 1
                
                # Eğer limit'e ulaştıysak dur
                if total_collected >= product_count:
                    break
            
            # Bir sonraki batch için sayfa numarasını güncelle
            current_page = next_page
            
            print(f"Batch tamamlandı. Toplam {total_collected}/{product_count} ürün toplandı.")

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