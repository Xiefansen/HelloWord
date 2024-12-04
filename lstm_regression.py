import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# 1. 生成示例数据
np.random.seed(42)
torch.manual_seed(42)

# 假设温度数据为一个正弦波加噪声
time = np.linspace(0, 100, 1000)
temperature = 10 + 5 * np.sin(time) + np.random.normal(scale=1, size=time.shape)

# 数据标准化
scaler = MinMaxScaler(feature_range=(-1, 1))
temperature_scaled = scaler.fit_transform(temperature.reshape(-1, 1))


# 创建时间序列数据集
def create_dataset(data, look_back=10):
    X, y = [], []
    for i in range(len(data) - look_back):
        X.append(data[i:i + look_back])
        y.append(data[i + look_back])
    return np.array(X), np.array(y)


look_back = 10
X, y = create_dataset(temperature_scaled, look_back)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 转换为 PyTorch 张量
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)


# 2. 构建 LSTM 模型
class LSTMRegressor(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, num_layers=1, output_size=1):
        super(LSTMRegressor, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])  # 取最后一个时间步的输出
        return out


# 实例化模型
input_size = 1
hidden_size = 50
num_layers = 2
output_size = 1
model = LSTMRegressor(input_size, hidden_size, num_layers, output_size)

# 3. 训练模型
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 数据需要有3维形状: (batch_size, sequence_length, input_size)
X_train = X_train  # 形状已经是 (样本数, 时间步, 特征数)
X_test = X_test

print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)


# 模型训练部分
num_epochs = 100
for epoch in range(num_epochs):
    model.train()
    outputs = model(X_train)  # 输入是 (batch_size, sequence_length, input_size)
    loss = criterion(outputs, y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# 4. 模型评估
model.eval()
with torch.no_grad():
    predictions = model(X_test).squeeze().cpu().numpy()
    y_test = y_test.cpu().numpy()
    predictions = scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()
    y_test = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()

# 计算性能指标
mse = mean_squared_error(y_test, predictions)
mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"R^2 Score (R2): {r2:.4f}")

# 5. 可视化结果
plt.figure(figsize=(12, 6))
plt.plot(y_test, label='Actual Temperature')
plt.plot(predictions, label='Predicted Temperature')
plt.legend()
plt.title('Temperature Prediction')
plt.xlabel('Time Steps')
plt.ylabel('Temperature')
plt.show()
