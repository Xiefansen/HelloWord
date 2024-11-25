import math
import numpy as np
import matplotlib.pyplot as plt


# 计算俯仰角 (pitch) 和滚转角 (roll)
def calculate_orientation(accel_x, accel_y, accel_z):
    # 俯仰角 (pitch)，通常是X和Z轴的关系
    pitch = math.atan2(accel_y, math.sqrt(accel_x ** 2 + accel_z ** 2))

    # 滚转角 (roll)，通常是Y和Z轴的关系
    roll = math.atan2(-accel_x, accel_z)

    # 转换为度数
    pitch_deg = math.degrees(pitch)
    roll_deg = math.degrees(roll)

    return pitch_deg, roll_deg


# 计算加速度大小
def calculate_acceleration_magnitude(accel_x, accel_y, accel_z):
    return math.sqrt(accel_x ** 2 + accel_y ** 2 + accel_z ** 2)


# 加速度计数据模拟（假设为15000个数据点，采样率为50Hz）
# accel_data = [(x, y, z), ...]
# 为了简单起见，我们使用随机数来模拟真实的加速度计数据
np.random.seed(42)  # 为了可重复的结果
accel_data = np.random.randn(15000, 3)  # 模拟15000个数据点 (x, y, z)

# 初始化方向为180度
initial_direction = 180  # 初始方向（单位：度）

# 计算第一组加速度数据的俯仰角和滚转角
accel_x, accel_y, accel_z = accel_data[0]
initial_pitch, initial_roll = calculate_orientation(accel_x, accel_y, accel_z)

# 设置初始方向为 180 度
current_direction = initial_direction

# 记录每一时刻的加速度大小和方向
acceleration_magnitudes = []
directions = []

# 假设一个简单的低通滤波器参数（alpha值用于平滑）
alpha = 0.1

# 使用加速度计数据估算运动方向
for i in range(1, len(accel_data)):
    accel_x, accel_y, accel_z = accel_data[i]

    # 计算当前时刻的俯仰角和滚转角
    pitch, roll = calculate_orientation(accel_x, accel_y, accel_z)

    # 计算方向变化
    delta_pitch = pitch - initial_pitch
    delta_roll = roll - initial_roll

    # 基于pitch和roll的变化估算方向，考虑到初始方向
    direction_change = delta_pitch * 0.1 + delta_roll * 0.1  # 假设加权影响

    # 更新当前方向
    current_direction += direction_change

    # 方向归一化（确保在0到360度之间）
    current_direction = (current_direction + 360) % 360

    # 更新初始俯仰角和滚转角
    initial_pitch, initial_roll = pitch, roll

    # 使用低通滤波器来平滑方向变化
    current_direction = alpha * current_direction + (1 - alpha) * (initial_direction + direction_change)

    # 记录加速度大小和当前方向
    accel_magnitude = calculate_acceleration_magnitude(accel_x, accel_y, accel_z)
    acceleration_magnitudes.append(accel_magnitude)
    directions.append(current_direction)

# 绘制加速度大小趋势图
plt.figure(figsize=(10, 5))
plt.subplot(2, 1, 1)
plt.plot(acceleration_magnitudes, label='Acceleration Magnitude')
plt.title('Acceleration Magnitude over Time')
plt.xlabel('Time (samples)')
plt.ylabel('Acceleration (m/s^2)')
plt.grid(True)
plt.legend()


# 绘制运动方向趋势图
plt.subplot(2, 1, 2)
plt.plot(directions, label='Direction', color='orange')
plt.title('Estimated Direction over Time')
plt.xlabel('Time (samples)')
plt.ylabel('Direction (degrees)')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()

# 输出最终的估算方向
print(f"Final estimated direction: {current_direction:.2f} degrees")

import torch
x = torch.rand(5, 3)
print(x)

import torch
print(torch.cuda.is_available())

