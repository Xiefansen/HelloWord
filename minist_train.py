import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# 数据预处理
transform = transforms.Compose([
    transforms.ToTensor(),  # 转换为张量
    transforms.Normalize((0.5,), (0.5,))  # 数据归一化到 [-1, 1]
])

# 加载 MNIST 数据集
train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# 创建数据加载器
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)  # 输出: 16x28x28
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)  # 输出: 16x14x14
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)  # 输出: 32x14x14
        self.fc1 = nn.Linear(32 * 7 * 7, 128)  # 全连接层
        self.fc2 = nn.Linear(128, 10)  # 输出层（10 类）

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 32 * 7 * 7)  # 展平
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 初始化模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNN().to(device)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

from tqdm import tqdm  # 导入 tqdm

# 训练模型
num_epochs = 5
for epoch in range(num_epochs):
    model.train()  # 切换到训练模式
    running_loss = 0.0

    # 使用 tqdm 包装数据加载器
    progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1}/{num_epochs}")
    index = 0
    for batch_idx, (inputs, labels) in progress_bar:
        index += 1
        print(index)
        inputs, labels = inputs.to(device), labels.to(device)

        # 清空梯度
        optimizer.zero_grad()

        # 前向传播
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # 反向传播和优化
        loss.backward()
        optimizer.step()

        # 更新损失
        running_loss += loss.item()

        # 更新进度条显示信息
        progress_bar.set_postfix(loss=(running_loss / (batch_idx + 1)))

    print(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {running_loss / len(train_loader)}")


# 测试模型
model.eval()  # 切换到评估模式
correct = 0
total = 0

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Test Accuracy: {100 * correct / total}%")

import numpy as np

# 获取一批测试数据
data_iter = iter(test_loader)
images, labels = next(data_iter)

# 显示图像
def imshow(img):
    img = img / 2 + 0.5  # 反归一化
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)), cmap="gray")
    plt.show()

# 显示测试样本
imshow(torchvision.utils.make_grid(images))

# 显示预测结果
images = images.to(device)
outputs = model(images)
_, predicted = torch.max(outputs, 1)

print("Predicted: ", " ".join(str(predicted[j].item()) for j in range(8)))
