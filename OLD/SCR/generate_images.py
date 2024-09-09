import torch
import torchvision.utils as vutils
import matplotlib.pyplot as plt
from train_autoencoder import Autoencoder
from train_gan import Generator

# Параметры
device = torch.device("cuda:0" if (torch.cuda.is_available() and torch.cuda.is_available()) else "cpu")
latent_size = 40
condition_size = 10
feature_map_size = 64
ngpu = 1

# Загрузка моделей
autoencoder = Autoencoder(ngpu).to(device)
autoencoder.load_state_dict(torch.load('autoencoder.pth'))
autoencoder.eval()

netG = Generator(ngpu).to(device)
netG.load_state_dict(torch.load('generator.pth'))
netG.eval()

# Генерация изображений
fixed_noise = torch.randn(100, latent_size, 1, 1, device=device)
with torch.no_grad():
    fake = netG(fixed_noise, torch.tensor([i % 4 for i in range(100)]).to(device)).cpu()
    reconstructed_fake = autoencoder(fake.to(device), 'generate').cpu()

plt.figure(figsize=(12, 12))
plt.axis("off")
plt.title("Generated Images")
plt.imshow(np.transpose(vutils.make_grid(reconstructed_fake[:64], padding=5, normalize=True), (1, 2, 0)))
plt.show()
