# 🧴 EWG Skin Deep Scraper

A Python-based web scraper project that automatically collects cosmetic product information from the EWG Skin Deep website.

## 📋 Table of Contents

- [Requirements](#-requirements)
- [Installation](#-installation)
- [Usage](#-usage)
- [Project Structure](#-project-structure)
- [Output Format](#-output-format)
- [Technical Details](#️-technical-details)
- [Important Notes](#-important-notes)
- [Contributing](#-contributing)

## 🎯 About the Project

This project contains scrapers that collect product names and ingredients from various cosmetic categories on the [EWG Skin Deep](https://www.ewg.org/skindeep) platform. A separate Python script has been prepared for each category.

## 📋 Requirements

To run this project, you need:

- **Python 3.10 or higher** ([Download](https://www.python.org/downloads/))
- **Google Chrome** (latest version must be installed)
- **Internet connection** (for scraping operations)

## 🚀 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/Ewg_Scraper.git
cd Ewg_Scraper
```

### 2. Create a Virtual Environment

**Windows (PowerShell):**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**Windows (CMD):**
```cmd
python -m venv .venv
.venv\Scripts\activate.bat
```

**macOS/Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Required Packages

```bash
pip install -r requirements.txt
```

## 💻 Usage

### Step 1: Select a Scraper File

Find the Python file prepared for your assigned category in the `Ready_Scrapers` folder.

### Step 2: Run the Scraper

```bash
cd Ready_Scrapers
python Anti-aging.py
```

### Step 3: Check the CSV File

When scraping is complete, a CSV file with the category name will be created in the same folder:
- `Anti-aging.py` → `Anti-aging.csv`
- `BB_Cream.py` → `BB_Cream.csv`

## 📁 Project Structure

```
Ewg_Scraper/
│
├── Ready_Scrapers/              # All scraper files
│   ├── Anti-aging.py           
│   ├── Around-eye_cream.py     
│   ├── BB_Cream.py             
│   ├── CC_Cream.py            
│   ├── Facial_cleanser_Part_1.py
│   ├── Facial_cleanser_Part_2.py
│   ├── Facial_cleanser_Part_3.py
│   ├── Facial_moisturizer__treatment_Part_1.py
│   ├── Facial_moisturizer__treatment_Part_2.py
│   ├── Facial_moisturizer__treatment_Part_3.py
│   ├── Makeup_remover.py
│   ├── Mask_Part_1.py
│   ├── Mask_Part_2.py
│   ├── Mask_Part_3.py
│   ├── Oil_controller.py
│   ├── Pore_strips.py
│   ├── Serums_&_Essences_Part_1.py
│   ├── Serums_&_Essences_Part_2.py
│   ├── Skin_fading__lightener.py
│   └── Toners__astringents.py
│
├── requirements.txt            # Python dependencies
├── .gitignore                  # Git ignore rules
└── README.md                   # This file
```

## 📄 Output Format

Each scraper creates a CSV file with the following columns:

| Column | Description | Example |
|--------|-------------|---------|
| `id` | Unique product ID | 1, 2, 3... |
| `product_url` | Product's EWG page | https://www.ewg.org/skindeep/products/... |
| `name` | Product name | "Neutrogena Hydro Boost" |
| `category` | Product category | "Anti-aging", "Face Mask" |
| `ingredients` | Ingredients list | "Water, Glycerin, Dimethicone..." |

### Sample CSV Output:

```csv
id,product_url,name,category,ingredients
1,https://www.ewg.org/skindeep/products/123456,Product Name 1,Anti-aging,"Water, Glycerin, Sodium Hyaluronate"
2,https://www.ewg.org/skindeep/products/123457,Product Name 2,Face Mask,"Aqua, Butylene Glycol, Niacinamide"
```

## ⚙️ Technical Details

### Technologies Used

- **Selenium**: Web automation and dynamic content loading
- **BeautifulSoup4**: HTML parsing
- **Pandas**: Data processing and CSV creation
- **webdriver-manager**: Automatic Chrome WebDriver management

### Scraper Features

- ✅ **Headless mode**: Runs without browser visibility
- ✅ **Auto-retry**: Retries 3 times in case of errors
- ✅ **Incremental save**: Each product is saved to CSV immediately (prevents data loss)
- ✅ **User-agent spoofing**: To prevent bot detection
- ✅ **Rate limiting**: Wait time between each request


## 📝 Important Notes

⚠️ **Things to Consider:**

1. **Scraping Duration**: Varies depending on category and number of products
2. **Internet Connection**: Connection should not be interrupted during scraping
3. **CSV File**: Works in append mode, preserves existing data
4. **Rate Limiting**: Wait times added to avoid overwhelming the website
5. **Chrome Requirement**: Chrome browser must be installed on your system

## 🤝 Contributing

1. Run the scraper file assigned to you
2. Send the generated CSV file to the project manager
3. Report any errors or issues you encounter


## 📜 License

This project is for educational purposes. It should be used in accordance with EWG Skin Deep's terms of use.

---

⭐ **If you found this project useful, don't forget to give it a star!**

---
