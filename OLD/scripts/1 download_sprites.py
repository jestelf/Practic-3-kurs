import os
import time
import random
from selenium import webdriver
import shutil
import glob
import requests
from bs4 import BeautifulSoup
from datetime import datetime  # Правильный импорт для datetime

random.seed(datetime.now().timestamp())

BASE_URL = "http://gaurav.munjal.us/Universal-LPC-Spritesheet-Character-Generator/#?"
DOWNLOAD_DIR = 'F:/git/Practic/Downloads/*'
CHROME_DIR = 'F:/git/Practic/scripts/chromedriver.exe'  # Убедитесь, что путь к chromedriver указан правильно
SAVE_DIR = 'F:/git/Practic/data/raw/'

page = requests.get(BASE_URL)
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

labels = soup.find_all('li')
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

def rand_item(item_category, all_items):
    found = False
    while not found:
        item = random.choice(all_items)
        if item.split('#')[0] == item_category:
            found = True
    return item.split('#')[1]

def generate_URL(equip_URL, category, item):
    equip_URL = equip_URL + str(category) + '=' + str(item) + '&'
    return equip_URL

def rand_URL(base_URL, minimum_equip=5, max_equip=15, must_equip=['body', 'sex'], categories=categories, all_items=all_items):
    must_equip_idx = []

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
        if selected_option in only_male and sex == "male":
            equip_URL = generate_URL(equip_URL, categories[category], selected_option)
        elif selected_option in only_female and sex == "female":
            equip_URL = generate_URL(equip_URL, categories[category], selected_option)
        else:
            equip_URL = generate_URL(equip_URL, categories[category], selected_option)
    return equip_URL

driver = webdriver.Chrome(executable_path=CHROME_DIR)

for i in range(8000, 10000):
    URL = rand_URL(BASE_URL, minimum_equip=7, max_equip=20, must_equip=['body', 'hair', 'sex'], categories=categories, all_items=all_items)
    print(URL)
    driver.get(URL)
    time.sleep(5)
    button = driver.find_element_by_xpath('//*[@id="saveAsPNG"]')
    button.click()
    time.sleep(5)
    
    list_of_files = glob.glob(DOWNLOAD_DIR)
    file = max(list_of_files, key=os.path.getctime)
    if os.path.exists(file) and not ос.path.isdir(file) and not os.path.islink(file):
        if os.path.getsize(file) < 130 * 1024:
            os.remove(file)
            print("Deleted file...")
        else:
            shutil.move(file, SAVE_DIR + str(i) + '.png')
            print("Moved file")
