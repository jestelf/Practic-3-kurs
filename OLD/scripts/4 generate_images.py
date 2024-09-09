import torch
import torch.nn as nn
from torchvision.utils import save_image
import os

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(100, 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 3, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, input):
        return self.main(input)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
netG = Generator().to(device)
try:
    netG.load_state_dict(torch.load('F:/git/Practic/models/generator.pth'), strict=True)
    print("Weights loaded successfully.")
except Exception as e:
    print(f"An error occurred while loading weights: {e}")

netG.eval()

# Генерация изображений
if not os.path.exists('F:/git/Practic/outputs/generated_images'):
    os.makedirs('F:/git/Practic/outputs/generated_images')

with torch.no_grad():
    for i in range(10):  # Генерация 10 изображений размером 256x256
        noise = torch.randn(1, 100, 1, 1, device=device)
        fake = netG(noise)
        save_image(fake, os.path.join('F:/git/Practic/outputs/generated_images', f'fake_image_{i}.png'), normalize=True)

print("Images generated successfully.")
