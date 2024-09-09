import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt

# Путь к данным
dataroot = "F:/git/Practic/data/raw"
image_size = 64
batch_size = 100
workers = 2
ngpu = 1
latent_size = 40
feature_map_size = 64
num_epochs = 350
lr = 1e-4
beta1 = 0.5
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

# Предобработка данных
transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.CenterCrop(image_size),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

# Фильтрация директорий, которые содержат поддерживаемые форматы изображений
valid_extensions = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')
filtered_dirs = [d for d in os.listdir(dataroot) if any(f.endswith(valid_extensions) for f in os.listdir(os.path.join(dataroot, d)))]

dataset = dset.ImageFolder(root=dataroot, transform=transform, is_valid_file=lambda x: any(x.endswith(ext) for ext in valid_extensions))
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=workers)

# Определение Encoder и Decoder
class Encoder(nn.Module):
    def __init__(self, ngpu):
        super(Encoder, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
            nn.Conv2d(512, 64, 4, 1, 0, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2)
        )
        self.fc_class = nn.Sequential(
            nn.Linear(64, latent_size),
            nn.Sigmoid()
        )

    def forward(self, input):
        x = self.main(input)
        x = x.view(x.size(0), -1)
        x = self.fc_class(x)
        return x

class Decoder(nn.Module):
    def __init__(self, ngpu):
        super(Decoder, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            nn.ConvTranspose2d(latent_size, 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, embedding):
        embedding = embedding.view(embedding.size(0), embedding.size(1), 1, 1)
        return self.main(embedding)

class Autoencoder(nn.Module):
    def __init__(self, ngpu):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder(ngpu).to(device)
        self.decoder = Decoder(ngpu).to(device)

    def forward(self, input, mode='train'):
        if mode == 'train':
            x = self.encoder(input)
            return self.decoder(x)
        if mode == 'generate':
            return self.decoder(input)

autoencoder = Autoencoder(ngpu).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(autoencoder.parameters(), lr=lr, betas=(beta1, 0.999))

# Обучение
losses = []
print("Starting Training Loop...")
for epoch in range(num_epochs):
    for i, data in enumerate(dataloader, 0):
        autoencoder.zero_grad()
        image = data[0].to(device)
        reconstructed_image = autoencoder(image).view(-1)
        err = criterion(reconstructed_image.view(image.size()), image)
        err.backward()
        optimizer.step()
        
        if i % 50 == 0:
            print(f'[{epoch}/{num_epochs}][{i}/{len(dataloader)}] Loss: {err.item()}')
        losses.append(err.item())

    lr *= 0.995
    optimizer = optim.Adam(autoencoder.parameters(), lr=lr, betas=(beta1, 0.999))

plt.figure(figsize=(10, 5))
plt.title("AE Loss During Training")
plt.plot(losses)
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.legend()
plt.show()
