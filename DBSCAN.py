import numpy as np
from geopy.distance import great_circle
import pandas as pd


# 计算经纬度之间的地理距离
def haversine(lat1, lon1, lat2, lon2):
    return great_circle((lat1, lon1), (lat2, lon2)).meters


# 计算所有点的距离矩阵
def compute_distance_matrix(df):
    coords = df[['lat', 'lon']].to_numpy()
    dist_matrix = np.zeros((len(coords), len(coords)))

    for i in range(len(coords)):
        for j in range(i + 1, len(coords)):
            dist = haversine(coords[i][0], coords[i][1], coords[j][0], coords[j][1])
            dist_matrix[i, j] = dist_matrix[j, i] = dist
    return dist_matrix


# DBSCAN 算法实现
def dbscan(df, eps=50, min_samples=1):
    # 计算距离矩阵
    dist_matrix = compute_distance_matrix(df)
    n_points = len(df)

    # 初始化簇标签，-1表示噪声
    labels = [-1] * n_points
    cluster_id = 0

    # 访问每一个点
    for point_idx in range(n_points):
        if labels[point_idx] != -1:  # 如果该点已经被访问过，跳过
            continue

        # 获取当前点的邻域点
        neighbors = np.where(dist_matrix[point_idx] <= eps)[0]

        # 如果邻域点数量小于 min_samples，标记为噪声
        if len(neighbors) < min_samples:
            labels[point_idx] = -1  # 噪声点
        else:
            # 扩展簇，开始聚类
            labels[point_idx] = cluster_id
            expand_cluster(point_idx, neighbors, dist_matrix, labels, cluster_id, eps, min_samples)
            cluster_id += 1  # 聚类编号增加

    return labels


# 扩展簇的函数
def expand_cluster(point_idx, neighbors, dist_matrix, labels, cluster_id, eps, min_samples):
    # 创建队列来处理邻域中的点
    to_visit = list(neighbors)
    while to_visit:
        current_point = to_visit.pop()

        # 如果当前点已经被访问过，跳过
        if labels[current_point] != -1:
            continue

        # 标记为当前簇的成员
        labels[current_point] = cluster_id

        # 获取当前点的邻域点
        new_neighbors = np.where(dist_matrix[current_point] <= eps)[0]

        # 如果邻域点数量大于等于 min_samples，继续扩展
        if len(new_neighbors) >= min_samples:
            to_visit.extend(new_neighbors)


# 示例数据
data = pd.DataFrame({
    'lat': [39.9042, 39.9045, 39.9050, 39.9120, 39.9130],  # 示例数据
    'lon': [116.4074, 116.4075, 116.4080, 116.4120, 116.4140]
})

# 设置参数：eps=50米, min_samples=1
labels = dbscan(data, eps=50, min_samples=1)

# 将聚类结果添加到原数据中
data['cluster'] = labels

# 输出结果
print(data)


import matplotlib.pyplot as plt
import seaborn as sns

# 选择颜色样式
sns.set(style="whitegrid")

# 创建散点图
plt.figure(figsize=(8, 6))

# 使用聚类标签来选择颜色
palette = sns.color_palette("tab10", len(data['cluster'].unique()))  # 使用不同的颜色调色板

# 绘制聚类结果，噪声点(-1)使用灰色，其他点按类别绘制
for cluster in data['cluster'].unique():
    if cluster == -1:
        cluster_data = data[data['cluster'] == -1]
        plt.scatter(cluster_data['lon'], cluster_data['lat'], color='gray', label='Noise', alpha=0.5, edgecolors='k')
    else:
        cluster_data = data[data['cluster'] == cluster]
        plt.scatter(cluster_data['lon'], cluster_data['lat'], color=palette[cluster % len(palette)], label=f'Cluster {cluster}', alpha=0.7)

# 添加标题和标签
plt.title('Clustering of GPS Points (50m Distance Threshold)', fontsize=14)
plt.xlabel('Longitude', fontsize=12)
plt.ylabel('Latitude', fontsize=12)

# 显示图例
plt.legend(title='Cluster', loc='best')

# 显示图形
plt.show()
