import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18, ResNet18_Weights
from torch.optim import Adam
import matplotlib.pyplot as plt
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# 定義 DIP 模型，使用 ResNet18
class DIPModel(nn.Module):
    def __init__(self):
        super(DIPModel, self).__init__()
        self.encoder = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.encoder.fc = nn.Identity()  # 移除全連接層
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

# 設置設備
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 資料預處理
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# 載入 CIFAR-10 數據集
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=8, shuffle=True)

# 訓練 DIP 模型
dip_model = DIPModel().to(device)
dip_criterion = nn.MSELoss()
dip_optimizer = Adam(dip_model.parameters(), lr=0.001)

num_epochs = 30
loss_history = []
best_loss = np.inf

for epoch in range(num_epochs):
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, _ = data
        inputs = inputs.to(device)

        # 添加噪聲到輸入圖像
        noise = torch.randn_like(inputs) * 0.2
        noisy_inputs = inputs + noise

        dip_optimizer.zero_grad()
        outputs = dip_model(noisy_inputs)

        loss = dip_criterion(outputs, inputs)
        loss.backward()
        dip_optimizer.step()

        running_loss += loss.item()
        if i % 2000 == 1999:  # 每 2000 mini-batches 打印一次
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
            running_loss = 0.0

    avg_loss = running_loss / len(trainloader)
    loss_history.append(avg_loss)
    print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss}')

    # 保存損失最小的 DIP 模型
    if avg_loss < best_loss:
        best_loss = avg_loss
        torch.save(dip_model.state_dict(), 'result/dip_RestNet18_model.pth')

print('Finished DIP RestNet18 Training')

# 可視化並保存損失圖
if not os.path.exists('./figure'):
    os.makedirs('./figure')

plt.figure()
plt.plot(range(1, num_epochs + 1), loss_history, marker='o')
plt.title('DIP RestNet18 Training Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)
plt.savefig('./figure/dip_RestNet18_training_loss.png')
plt.show()
