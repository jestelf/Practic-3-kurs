import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import matplotlib.pyplot as plt
import os

# Параметры обучения
dataroot = "F:\\git\\Practic\\data\\characters"
batch_size = 128
image_size = 64
nc = 3  # Количество каналов в изображениях (RGB)
latent_size = 40  # Размер вектора латентного пространства
feature_map_size = 64
num_epochs = 100
lr = 0.0001
beta1 = 0.5
ngpu = 1  # Количество GPU (0 для использования CPU)

# Устройство (GPU или CPU)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Кастомный датасет для загрузки изображений из одной папки
class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_files = [f for f in os.listdir(root_dir) if os.path.isfile(os.path.join(root_dir, f))]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.image_files[idx])
        image = Image.open(img_name).convert('RGB')  # Преобразование изображения в RGB
        if self.transform:
            image = self.transform(image)
        return image

# Создание датасета и DataLoader
dataset = CustomDataset(root_dir=dataroot,
                        transform=transforms.Compose([
                            transforms.Resize(image_size),
                            transforms.CenterCrop(image_size),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                        ]))

dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Определение архитектуры автокодировщика
class Encoder(nn.Module):
    def __init__(self, ngpu):
        super(Encoder, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            nn.Conv2d(nc, feature_map_size, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2),
            nn.Conv2d(feature_map_size, feature_map_size * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_map_size * 2),
            nn.LeakyReLU(0.2),
            nn.Conv2d(feature_map_size * 2, feature_map_size * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_map_size * 4),
            nn.LeakyReLU(0.2),
            nn.Conv2d(feature_map_size * 4, feature_map_size * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_map_size * 8),
            nn.LeakyReLU(0.2),
            nn.Conv2d(feature_map_size * 8, feature_map_size, 4, 1, 0, bias=False),
            nn.BatchNorm2d(feature_map_size),
            nn.LeakyReLU(0.2),
        )
        self.fc = nn.Sequential(
            nn.Linear(feature_map_size * 1 * 1, latent_size),
            nn.ReLU(True)
        )

    def forward(self, input):
        x = self.main(input)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class Decoder(nn.Module):
    def __init__(self, ngpu):
        super(Decoder, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            nn.ConvTranspose2d(latent_size, feature_map_size * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(feature_map_size * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(feature_map_size * 8, feature_map_size * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_map_size * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(feature_map_size * 4, feature_map_size * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_map_size * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(feature_map_size * 2, feature_map_size, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_map_size),
            nn.ReLU(True),
            nn.ConvTranspose2d(feature_map_size, nc, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, input):
        input = input.view(input.size(0), input.size(1), 1, 1)
        output = self.main(input)
        return output

class Autoencoder(nn.Module):
    def __init__(self, ngpu):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder(ngpu)
        self.decoder = Decoder(ngpu)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# Инициализация автокодировщика и установка функции потерь и оптимизатора
autoencoder = Autoencoder(ngpu).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(autoencoder.parameters(), lr=lr, betas=(beta1, 0.999))

# Обучение автокодировщика
for epoch in range(num_epochs):
    for i, data in enumerate(dataloader, 0):
        autoencoder.zero_grad()
        real_images = data.to(device)
        output = autoencoder(real_images)
        loss = criterion(output, real_images)
        loss.backward()
        optimizer.step()

        if i % 2000 == 0:
            print(f"Epoch [{epoch}/{num_epochs}] Step [{i}/{len(dataloader)}] Loss: {loss.item()}")

# Сохранение модели автокодировщика
torch.save(autoencoder.state_dict(), 'autoencoder.pth')
