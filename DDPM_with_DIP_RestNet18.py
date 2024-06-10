import torch
import torch.nn as nn
from denoising_diffusion_pytorch import Unet, GaussianDiffusion
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from ema_pytorch import EMA
from tqdm import tqdm
import os
from pytorch_fid import fid_score
import numpy as np
from PIL import Image
from torchvision.models import resnet18, ResNet18_Weights
from pytorch_fid import inception
import torch.nn.functional as F

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


# 定義 DIP 模型，使用 ResNet18
class DIPModel(nn.Module):
    def __init__(self):
        super(DIPModel, self).__init__()
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


def save_images(images, folder, prefix):
    os.makedirs(folder, exist_ok=True)
    for i, img in enumerate(images):
        img = img.permute(1, 2, 0).cpu().numpy()
        img = (img * 255).astype(np.uint8)
        img = Image.fromarray(img)
        img.save(os.path.join(folder, f"{prefix}_{i}.png"))


def calculate_fid_from_tensors(real_images, generated_images, device):
    block_idx = inception.InceptionV3.BLOCK_INDEX_BY_DIM[2048]
    model = inception.InceptionV3([block_idx]).to(device)

    def get_activations(images):
        images = F.interpolate(images, size=(299, 299), mode='bilinear', align_corners=False)
        pred = model(images)[0]
        if pred.size(2) != 1 or pred.size(3) != 1:
            pred = F.adaptive_avg_pool2d(pred, output_size=(1, 1))
        return pred.squeeze(3).squeeze(2).cpu().numpy()

    real_activations = get_activations(real_images)
    generated_activations = get_activations(generated_images)

    mu1 = np.mean(real_activations, axis=0)
    sigma1 = np.cov(real_activations, rowvar=False)
    mu2 = np.mean(generated_activations, axis=0)
    sigma2 = np.cov(generated_activations, rowvar=False)

    fid_value = fid_score.calculate_frechet_distance(mu1, sigma1, mu2, sigma2)
    return fid_value


def generate_samples_and_evaluate_fid(model, dataloader, device, num_samples=500):
    all_generated_images = []
    all_real_images = []

    with torch.no_grad():
        for i, data in enumerate(dataloader):
            if len(all_generated_images) >= num_samples:
                break
            real_images, _ = data
            real_images = real_images.to(device)
            all_real_images.append(real_images)

            generated_images = model.sample(batch_size=real_images.size(0))
            all_generated_images.append(generated_images)

    all_real_images = torch.cat(all_real_images, dim=0)[:num_samples]
    all_generated_images = torch.cat(all_generated_images, dim=0)[:num_samples]

    fid_score_value = calculate_fid_from_tensors(all_real_images, all_generated_images, device)
    return fid_score_value


def train_DDPM_with_DIP():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 初始化 DIP 模型
    dip_model = DIPModel().to(device)
    dip_model.load_state_dict(torch.load('./result/dip_RestNet18_model.pth'))  # 加載已訓練的 DIP 模型
    dip_model.eval()

    # 初始化 DDPM 模型
    ddpm_model = Unet(
        dim=64,
        dim_mults=(1, 2, 4, 8)
    ).to(device)

    diffusion = GaussianDiffusion(
        ddpm_model,
        image_size=32,
        timesteps=1000
    ).to(device)

    # 加載 CIFAR-10 訓練數據集
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    cifar10 = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    dataloader = torch.utils.data.DataLoader(cifar10, batch_size=8, shuffle=True)

    # 定義優化器
    optimizer = torch.optim.Adam(diffusion.parameters(), lr=8e-5)

    # 訓練參數
    train_num_steps = 30000
    gradient_accumulate_every = 2
    ema_decay = 0.995
    amp = True

    # 訓練 DDPM 模型
    scaler = torch.cuda.amp.GradScaler(enabled=amp)
    ema = EMA(ddpm_model, beta=ema_decay, update_every=10).to(device)

    step = 0
    loss_history = []
    fid_scores = []
    best_loss = np.inf

    with tqdm(total=train_num_steps, desc="Training") as pbar:
        while step < train_num_steps:
            epoch_loss = 0
            for data in dataloader:
                step += 1

                if step > train_num_steps:
                    break

                # 獲取批次數據
                batch_images, _ = data
                batch_images = batch_images.to(device)

                # 使用 DIP 模型的輸出作為初始輸入
                with torch.no_grad():
                    initial_input = dip_model(batch_images)

                optimizer.zero_grad()

                # 使用自動混合精度訓練
                with torch.cuda.amp.autocast(enabled=amp):
                    loss = diffusion(initial_input)
                    loss = loss / gradient_accumulate_every

                scaler.scale(loss).backward()

                if step % gradient_accumulate_every == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    ema.update()
                    optimizer.zero_grad()

                epoch_loss += loss.item()
                pbar.update(1)  # 更新 tqdm 進度條

                if step % 3000 == 0:
                    avg_loss = epoch_loss / 3000
                    loss_history.append(avg_loss)
                    print(f'Step {step}/{train_num_steps}, Loss: {avg_loss}')
                    epoch_loss = 0

                    ema.eval()
                    with torch.no_grad():
                        sampled_images = diffusion.sample(batch_size=8)
                    ema.train()

                    # 計算 FID
                    fid_score_value = generate_samples_and_evaluate_fid(diffusion, dataloader, device, num_samples=30)
                    fid_scores.append(fid_score_value)
                    print(f'Step {step}, FID: {fid_score_value}')

                    # 保存損失最小的 DDPM 模型
                    if avg_loss < best_loss:
                        best_loss = avg_loss
                        torch.save(ddpm_model.state_dict(), 'result/ddpm_dip_RestNet18_model.pth')

    # 可視化並保存損失圖和 FID 圖
    if not os.path.exists('./figure'):
        os.makedirs('./figure')

    plt.figure()
    plt.plot(range(1, len(loss_history) + 1), loss_history, marker='o')
    plt.title('DDPM with DIP RestNet18 Training Loss Over Steps')
    plt.xlabel('Step (3000)')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.savefig('./figure/ddpm_dip_RestNet18_training_loss.png')
    plt.show()

    plt.figure()
    plt.plot(range(1, len(fid_scores) + 1), fid_scores, marker='o')
    plt.title('DDPM with DIP RestNet18 FID Scores Over Steps')
    plt.xlabel('Step (3000)')
    plt.ylabel('FID Score')
    plt.grid(True)
    plt.savefig('./figure/ddpm_dip_RestNet18_fid_scores.png')
    plt.show()


if __name__ == "__main__":
    train_DDPM_with_DIP()
