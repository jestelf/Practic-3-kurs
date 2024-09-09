import os
from PIL import Image
from realesrgan import RealESRGAN
import torch

def preprocess_image(img_path, output_path, model):
    # Открываем изображение с помощью PIL
    img = Image.open(img_path).convert("RGBA")
    
    # Увеличиваем разрешение изображения с помощью Real-ESRGAN
    img = model.predict(img)
    
    # Изменяем размер изображения до 256x256, если необходимо
    img = img.resize((256, 256), Image.LANCZOS)
    
    # Сохраняем изображение
    img.save(output_path)

# Убедитесь, что у вас установлен Real-ESRGAN и Torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = RealESRGAN(device, scale=4)
model.load_weights('RealESRGAN_x4.pth')

input_dir = 'F:/git/Practic/data/raw/'
output_dir = 'F:/git/Practic/data/processed/'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

for root, dirs, files in os.walk(input_dir):
    for img_name in files:
        if img_name.endswith('.png'):
            img_path = os.path.join(root, img_name)
            output_path = os.path.join(output_dir, img_name)
            preprocess_image(img_path, output_path, model)
