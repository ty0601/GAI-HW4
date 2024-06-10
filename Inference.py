import numpy as np
import torch
from torch import nn
from torchvision import datasets, transforms
from torchvision.models import resnet18, ResNet18_Weights
import matplotlib.pyplot as plt
import os
from denoising_diffusion_pytorch import Unet, GaussianDiffusion

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# 定義模型路徑
DDPM_MODEL_PATH = './result/ddpm_model.pth'
DIP_MODEL_PATH = './result/dip_model.pth'
DIP_RESNET18_MODEL_PATH = './result/dip_RestNet18_model.pth'
DDPM_DIP_MODEL_PATH = './result/ddpm_dip_model.pth'
DDPM_DIP_RESNET18_MODEL_PATH = './result/ddpm_dip_RestNet18_model.pth'

# 設置設備
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 資料預處理
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# 加載 CIFAR-10 數據集
testset = datasets.CIFAR10(root='./data', train=False, download=False, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=9, shuffle=False)

# 定義保存圖片的函數
def save_images(images, folder, filename, title):
    os.makedirs(folder, exist_ok=True)
    fig, axes = plt.subplots(3, 3, figsize=(6, 6))
    for i, img in enumerate(images):
        img = img.permute(1, 2, 0).cpu().numpy()
        img = (img * 255).astype(np.uint8)
        axes[i // 3, i % 3].imshow(img)
        axes[i // 3, i % 3].axis('off')
    plt.suptitle(title)
    save_path = os.path.join(folder, filename)
    plt.savefig(save_path)
    print(f"Saved images to {save_path}")
    plt.close(fig)

# 定義 DIP 模型
class DIPModel(nn.Module):
    def __init__(self):
        super(DIPModel, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 3, 3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

# 定義 DIP 模型，使用 ResNet18
class DIPResNet18Model(nn.Module):
    def __init__(self):
        super(DIPResNet18Model, self).__init__()
        self.encoder = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.encoder.fc = nn.Identity()
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 3, 4, stride=2, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), 512, 1, 1)
        x = self.decoder(x)
        # 確保輸出尺寸為32x32
        x = nn.functional.interpolate(x, size=(32, 32), mode='bilinear', align_corners=False)
        return x

# 初始化模型
ddpm_model = Unet(dim=64, dim_mults=(1, 2, 4, 8)).to(device)
ddpm_model.load_state_dict(torch.load(DDPM_MODEL_PATH))
ddpm_diffusion = GaussianDiffusion(ddpm_model, image_size=32, timesteps=1000).to(device)

dip_model = DIPModel().to(device)
dip_model.load_state_dict(torch.load(DIP_MODEL_PATH))
dip_model.eval()

dip_resnet18_model = DIPResNet18Model().to(device)
dip_resnet18_model.load_state_dict(torch.load(DIP_RESNET18_MODEL_PATH))
dip_resnet18_model.eval()

ddpm_dip_model = Unet(dim=64, dim_mults=(1, 2, 4, 8)).to(device)
ddpm_dip_model.load_state_dict(torch.load(DDPM_DIP_MODEL_PATH))
ddpm_dip_diffusion = GaussianDiffusion(ddpm_dip_model, image_size=32, timesteps=1000).to(device)

ddpm_dip_resnet18_model = Unet(dim=64, dim_mults=(1, 2, 4, 8)).to(device)
ddpm_dip_resnet18_model.load_state_dict(torch.load(DDPM_DIP_RESNET18_MODEL_PATH))
ddpm_dip_resnet18_diffusion = GaussianDiffusion(ddpm_dip_resnet18_model, image_size=32, timesteps=1000).to(device)

# 生成圖片並保存
with torch.no_grad():
    for i, data in enumerate(testloader):
        if i > 0:
            break  # 只處理第一個batch
        real_images, _ = data
        real_images = real_images.to(device)

        # DDPM
        ddpm_diffusion.eval()
        ddpm_samples = ddpm_diffusion.sample(batch_size=real_images.size(0))
        save_images(ddpm_samples, './output', 'DDPM.png', 'DDPM Samples')

        # DIP with DDPM
        dip_model.eval()
        dip_initial_input = dip_model(real_images)
        ddpm_dip_diffusion.eval()
        ddpm_dip_samples = ddpm_dip_diffusion.sample(batch_size=dip_initial_input.size(0))
        save_images(ddpm_dip_samples, './output', 'DDPM_DIP.png', 'DDPM with DIP Sample')

        # DIP with ResNet18 and DDPM
        dip_resnet18_model.eval()
        dip_resnet18_initial_input = dip_resnet18_model(real_images)
        ddpm_dip_resnet18_diffusion.eval()
        ddpm_dip_resnet18_samples = ddpm_dip_resnet18_diffusion.sample(batch_size=dip_resnet18_initial_input.size(0))
        save_images(ddpm_dip_resnet18_samples, './output', 'DDPM_DIP_RESNET18.png', 'DDPM with RestNet18-based DIP Sample')

print("Inference complete. Images are saved in the output folder.")
