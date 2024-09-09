import os
import zipfile
import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
import shutil
import glob
import random
import time
from datetime import datetime

# Установим seed для воспроизводимости
random.seed(datetime.now().timestamp())

# Параметры
base_dir = "F:\\git\\Practic"
zip_path = os.path.join(base_dir, "data.zip")
extract_path = os.path.join(base_dir, "extracted")
download_dir = os.path.join(base_dir, "downloads")
save_dir = os.path.join(base_dir, "rpg_char")
chrome_driver_path = 'F:\\git\\Practic\\chromedriver.exe'
chrome_binary_path = "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe"

# Создание необходимых директорий
os.makedirs(extract_path, exist_ok=True)
os.makedirs(download_dir, exist_ok=True)
os.makedirs(save_dir, exist_ok=True)

# Извлечение содержимого data.zip
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_path)

# Проверка содержимого извлеченной директории
extracted_files = os.listdir(extract_path)
print("Extracted files:", extracted_files)

# Альтернативный URL для скачивания спрайтов
BASE_URL = "https://sanderfrenken.github.io/Universal-LPC-Spritesheet-Character-Generator/#?"

# Функция для получения всех опций с веб-страницы
def get_all_options(base_url):
    try:
        page = requests.get(base_url)
        soup = BeautifulSoup(page.content, 'html.parser')

        all_items = []
        names = []
        labels = soup.find_all('li')
        for label in labels:
            name = str(label).split('name="', 1)
            if len(name) > 1:
                name = name[1]
            option = str(label.find_all('label')).split('-', 1)
            if len(option) > 1:
                option = option[1]
            name_opt = str(name).split('"', 1)[0] + '#' + str(option).split('"', 1)[0]
            names.append(str(name).split('"', 1)[0])
            all_items.append(name_opt)

        all_items = list(set(all_items))
        names = list(set(names))

        prefixes = ('hair-', 'hairs', '[', "'")
        forbidden = '<'
        for word in names[:]:
            if word.startswith(prefixes):
                names.remove(word)
        for word in all_items[:]:
            if word.startswith(prefixes) or forbidden in word:
                all_items.remove(word)

        categories = names
        only_male = []
        only_female = []
        for label in labels:
            if len(str(label).split('data-required="sex=', 1)) > 1:
                sex = str(label).split('data-required="sex=', 1)[1].split('" id=', 1)[0]
                equipment = str(label.find_all('label')).split('-', 1)
                if len(equipment) > 1:
                    equipment = equipment[1]
                if sex == 'male':
                    only_male.append(str(equipment).split('"', 1)[0])
                elif sex == 'female':
                    only_female.append(str(equipment).split('"', 1)[0])

        only_male = [i for i in only_male if i not in only_female]
        only_female = [i for i in only_female if i not in only_male]

        return categories, all_items, only_male, only_female
    except requests.exceptions.RequestException as e:
        print(f"Error fetching options from URL: {e}")
        return [], [], [], []

# Функция для генерации URL с случайными параметрами
def generate_random_url(base_url, categories, all_items, only_male, only_female):
    def rand_item(item, options):
        result = [i for i in options if i.startswith(item)]
        if len(result) > 1:
            selected_option = random.randint(0, len(result) - 1)
        else:
            selected_option = 0
        equipment = result[selected_option].split('#', 1)
        if len(equipment) > 1:
            equipment = equipment[1]
        return equipment

    def generate_URL(base_URL, item, option):
        base_URL += str(item + "=" + str(option) + "&")
        return base_URL

    def rand_URL(base_URL, minimum_equip=5, max_equip=15, must_equip=['body', 'head', 'sex'], categories=categories, all_items=all_items):
        rand_len = random.randint(minimum_equip, max_equip)
        rand_categories = random.sample(range(0, len(categories)), rand_len)
        for eq in must_equip:
            if categories.index(eq) not in rand_categories:
                rand_categories.insert(0, categories.index(eq))
            else:
                rand_categories.insert(0, rand_categories.pop(rand_categories.index(categories.index(eq))))

        sex = rand_item('sex', all_items)
        equip_URL = str(base_URL) + 'sex=' + str(sex) + '&'
        for category in rand_categories[1:]:
            selected_option = rand_item(categories[category], all_items)
            equip_URL = generate_URL(equip_URL, categories[category], selected_option)
        return equip_URL

    return rand_URL(base_url, minimum_equip=7, max_equip=20, must_equip=['body', 'head', 'sex'], categories=categories, all_items=all_items)

# Получение всех опций
categories, all_items, only_male, only_female = get_all_options(BASE_URL)

# Запуск Selenium для скачивания спрайтов
chrome_options = Options()
chrome_options.binary_location = chrome_binary_path
chrome_options.add_experimental_option("prefs", {
    "download.default_directory": download_dir,
    "download.prompt_for_download": False,
    "download.directory_upgrade": True,
    "safebrowsing.enabled": True
})
service = Service(chrome_driver_path)
driver = webdriver.Chrome(service=service, options=chrome_options)

for i in range(8000, 10000):
    URL = generate_random_url(BASE_URL, categories, all_items, only_male, only_female)
    print(URL)
    driver.get(URL)
    time.sleep(10)  # Увеличиваем время ожидания для завершения скачивания
    button = driver.find_element(By.XPATH, '//*[@id="saveAsPNG"]')
    button.click()
    time.sleep(10)  # Увеличиваем время ожидания для завершения скачивания
    list_of_files = glob.glob(os.path.join(download_dir, '*'))
    if list_of_files:
        file = max(list_of_files, key=os.path.getctime)
        if os.path.exists(file) and not os.path.isdir(file) and not os.path.islink(file):
            if os.path.getsize(file) < 130 * 1024:
                os.remove(file)
                print("Deleted file...")
            else:
                shutil.move(file, os.path.join(save_dir, str(i) + '.png'))
                print("Moved file")
    else:
        print("No files downloaded")

driver.quit()
