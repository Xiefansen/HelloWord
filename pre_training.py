import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import models
import matplotlib.pyplot as plt
import numpy as np

# 定义数据预处理方法
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),  # 转换为3通道图像（VGG的要求）
    transforms.Resize(224),  # 将图像大小调整为224x224
    transforms.ToTensor(),  # 将图像转换为Tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # VGG的标准归一化
])

# 3. 加载 MNIST 数据集
# 下载训练集和测试集
trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# 创建数据加载器
trainloader = DataLoader(trainset, batch_size=1, shuffle=True)
testloader = DataLoader(testset, batch_size=1, shuffle=False)

from torch.utils.data import Subset

# 获取前 1000 个样本的索引
subset_indices = list(range(1000))

# 使用 Subset 创建一个新的子数据集
train_subset = Subset(trainset, subset_indices)
test_subset = Subset(testset, subset_indices)
# 使用子数据集创建数据加载器
trainloader = DataLoader(train_subset, batch_size=64, shuffle=True)
testloader = DataLoader(test_subset, batch_size=64, shuffle=False)

# 4. 修改 VGG 模型的最后一层
# 初始化模型
vgg16 = models.vgg16(pretrained=False)  # 先加载模型结构

# 加载本地权重
state_dict = torch.load("./model/vgg16_pretrained.pth")
vgg16.load_state_dict(state_dict)

# 如果需要修改输出层
vgg16.classifier[6] = torch.nn.Linear(in_features=4096, out_features=10)

print(vgg16)


# 查看模型结构
print(vgg16)

# 将模型转移到设备（GPU或CPU）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vgg16 = vgg16.to(device)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(vgg16.parameters(), lr=0.001)

# 5. 设置损失函数和优化器
# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(vgg16.parameters(), lr=0.001)

# 6. 训练模型
# 训练模型
num_epochs = 10
for epoch in range(num_epochs):
    vgg16.train()  # 切换到训练模式
    running_loss = 0.0
    correct = 0
    total = 0
    index = 0
    num_samples_to_use = 1000
    for batch_idx, (inputs, labels) in enumerate(trainloader):
        if batch_idx >= num_samples_to_use:
            break  # 停止迭代

        # 将数据移到相同设备（GPU/CPU）
        inputs, labels = inputs.to(device), labels.to(device)

        # 清空梯度
        optimizer.zero_grad()

        # 前向传播
        outputs = vgg16(inputs)
        loss = criterion(outputs, labels)

        # 反向传播和优化
        loss.backward()
        optimizer.step()

        # 统计损失和准确率
        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    # 打印每个epoch的损失和准确率
    print(
        f"Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(trainloader)}, Accuracy: {100 * correct / total}%")

torch.save(vgg16.state_dict(), "vgg16_pretrained.pth")


# 7. 评估模型
# 测试模型
vgg16.eval()  # 切换到评估模式
correct = 0
total = 0

with torch.no_grad():
    for inputs, labels in testloader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = vgg16(inputs)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Test Accuracy: {100 * correct / total}%")


# 8. 可视化部分预测结果（可选）
# 显示一些测试图像及其预测结果
def imshow(img):
    img = img / 2 + 0.5  # 反归一化
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


dataiter = iter(testloader)
images, labels = next(dataiter)

# 显示图像
imshow(torchvision.utils.make_grid(images))
# 打印预测标签
outputs = vgg16(images.to(device))
_, predicted = torch.max(outputs, 1)
print('Predicted:', ' '.join(f'{predicted[j].item()}' for j in range(4)))
