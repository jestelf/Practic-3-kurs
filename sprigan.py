import os
from datetime import datetime
import random
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Устанавливаем случайное семя для воспроизводимости
random.seed(datetime.now().timestamp())

# Путь к папке с исходными спрайтшитами
DIR = 'F:\\git\\Practic\\rpg_char\\characters'

spritesheets = []
for r, d, f in os.walk(DIR):
    for file in f:
        if file.endswith(".png"):
            spritesheets.append(os.path.join(r, file))

start_h = 260
start_w = 0
char_h = 64
char_w = 64
no_chars = 4

# Извлечение персонажей
for i, spritesheet in enumerate(spritesheets):
    for nr in range(no_chars):
        image = mpimg.imread(spritesheet)
        resized_im = image[start_h + char_h * nr:start_h + char_h * (nr + 1), start_w:start_w + char_w, :]

        path = 'F:\\git\\Practic\\data\\characters\\'
        if not os.path.exists(path):
            os.makedirs(path)

        plt.imsave(path + str(i) + '_' + str(nr) + '.png', resized_im)

print("Извлечение персонажей завершено.")
