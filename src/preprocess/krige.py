import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
from pykrige.ok import OrdinaryKriging
from shapely.geometry import Point
import os
import json
import pandas as pd

from rasterio.transform import from_origin
from rasterio import Affine
import rasterio


def read_json_to_gdf(path):
    with open(path, 'r') as f: 
        data = json.load(f)
    features = data['features']
    gdf = gpd.GeoDataFrame.from_features(features)
    # 指定坐标系为 WGS84 (EPSG:4326)
    gdf.crs = 'EPSG:4326'
    # 去除所有香港以外的点
    gdf = gdf[gdf.geometry.x >= 113.5]
    gdf = gdf[gdf.geometry.x <= 114.5]
    gdf = gdf[gdf.geometry.y >= 22.1]
    gdf = gdf[gdf.geometry.y <= 22.5]
    
    return gdf


def save_tiff(save_path, data, transform, crs):
    # 获取数据维度
    height, width = data.shape

    # 创建输出文件
    with rasterio.open(
        save_path,
        'w',
        driver='GTiff',
        height=height,
        width=width,
        count=1,
        dtype=data.dtype,
        crs=crs,
        transform=transform,
    ) as dst:
        dst.write(data, 1)


# 文件路径
DIR = os.path.dirname(os.path.abspath(__file__))
BOUNDARY_PATH = os.path.join(DIR, '..', '..', 'data2', 'HK.json')  # 边界 GeoJSON 文件路径

STATION_PATH1 = os.path.join(DIR, '..', '..', 'data2', 'weather', 'output', 'RH', 'metadata.json') # 湿度
STATION_PATH2 = os.path.join(DIR, '..', '..', 'data2', 'weather', 'output', 'RF', 'metadata.json') # 降水量
STATION_PATH3 = os.path.join(DIR, '..', '..', 'data2', 'weather', 'output', 'WSPD', 'metadata.json')  # 风速
STATION_PATH4 = os.path.join(DIR, '..', '..', 'data2', 'weather', 'output', 'ALLTEMP', 'metadata.json') # 温度

SAVE_DIR = os.path.join(DIR, '..', '..', 'data2', 'weather', 'output')  # 输出目录

# 创建输出目录
os.makedirs(SAVE_DIR, exist_ok=True)

def kriging_interpolation_without_boundary(station_path, boundary_path, img_save_path, value='mean'):
    # 加载站点数据
    # station_gdf = gpd.read_file(station_path)
    station_gdf = read_json_to_gdf(station_path)
    
    # 加载边界数据
    boundary_gdf = gpd.read_file(boundary_path)

    # 确保两者使用相同的坐标系
    if station_gdf.crs != boundary_gdf.crs:
        boundary_gdf = boundary_gdf.to_crs(station_gdf.crs)

    # 将站点数据转换为 WGS84 坐标系
    station_gdf = station_gdf.to_crs(epsg=4326)

    # 提取站点坐标和 mean 值
    coordinates = np.array([point.coords[0] for point in station_gdf.geometry])  # 提取经纬度
    # mean_values = station_gdf['mean'].values  # 提取 mean 值

    # mean_values = station_gdf['extreme_months'].apply(lambda x: x['top3_mean'])  # 提取极端月份值
    # mean_values = station_gdf['extreme_months'].apply(lambda x: x['bottom3_mean'])  # 提取极端月份值
    # mean_values = station_gdf['extreme_months'].apply(lambda x: x['combined_extreme_mean'])  # 提取极端月份值

    if value == 'mean':
        mean_values = station_gdf['mean'].values
    elif value == 'low':
        mean_values = station_gdf['extreme_months'].apply(lambda x: x['bottom3_mean'])
    elif value == 'high':
        mean_values = station_gdf['extreme_months'].apply(lambda x: x['top3_mean'])
    elif value == 'combined':
        mean_values = station_gdf['extreme_months'].apply(lambda x: x['combined_extreme_mean'])
    else:
        raise ValueError("Invalid value for 'value'. Choose from ['mean', 'low', 'high', 'combined'].")


    # 获取边界多边形
    boundary_polygon = boundary_gdf.geometry.unary_union

    # 定义插值网格范围（基于边界范围）
    x_min, y_min, x_max, y_max = boundary_gdf.total_bounds
    grid_x = np.linspace(x_min, x_max, 300)
    grid_y = np.linspace(y_min, y_max, 300)
    grid_x, grid_y = np.meshgrid(grid_x, grid_y)

    # 使用克里金插值，调整变异函数参数
    OK = OrdinaryKriging(
        coordinates[:, 0],  # 经度
        coordinates[:, 1],  # 纬度
        mean_values,        # 值
        variogram_model='spherical',  # 变异函数模型为球面模型
        variogram_parameters=[10000.0, 1000.0, 0.1],  # 设置变异函数参数：较大的范围（10000），变异度（1000），偏差（0.1）
        verbose=False,
        enable_plotting=False,
    )

    # 在网格上进行插值
    grid_z, _ = OK.execute('grid', grid_x[0, :], grid_y[:, 0])

    # 创建掩码，只保留边界内的点
    # mask = np.array([boundary_polygon.contains(Point(x, y)) for x, y in zip(grid_x.ravel(), grid_y.ravel())])
    # mask = mask.reshape(grid_x.shape)
    # grid_z[~mask] = np.nan  # 将边界外的点设置为 NaN

    # 保存为 tiff 文件
    # save_path 为替换传入的 img_save_path 的后缀为 .tiff
    save_path = img_save_path.replace('.png', '.tiff')

    # 保存为 tiff 文件，设置坐标系为 WGS84 (EPSG:4326)
    transform = Affine.translation(x_min, y_max) * Affine.scale((x_max - x_min) / grid_x.shape[1], (y_min - y_max) / grid_y.shape[0])
    save_tiff(save_path, grid_z, transform, crs='EPSG:4326')

    # 绘制插值和边界叠加图
    plt.figure(figsize=(12, 10))
    contour = plt.contourf(grid_x, grid_y, grid_z, levels=15, cmap="viridis", alpha=0.7)
    plt.colorbar(contour, label="Mean Value")
    plt.scatter(coordinates[:, 0], coordinates[:, 1], c="red", label="Stations", edgecolor="black")
    
    # 绘制边界
    boundary_gdf.boundary.plot(ax=plt.gca(), color="black", linewidth=1, label="Boundary")
    
    # 添加图例和标题
    plt.legend()
    # plt.title("10 Years Mean Humidity Interpolation with Boundary (Kriging)")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.savefig(img_save_path)
    plt.close()


if __name__ == '__main__':
    # 输入文件路径
    save_image_path = os.path.join(SAVE_DIR, 'kriging_RH.png')  
    save_image_path2 = os.path.join(SAVE_DIR, 'kriging_RF.png')  
    save_image_path3 = os.path.join(SAVE_DIR, 'kriging_WSPD.png')  
    save_image_path4 = os.path.join(SAVE_DIR, 'kriging_ALLTEMP.png')    

    # print(read_json_to_gdf(STATION_PATH1))

    # 调用克里金插值函数
    kriging_interpolation_without_boundary(STATION_PATH1, BOUNDARY_PATH, save_image_path, value='low')
    kriging_interpolation_without_boundary(STATION_PATH2, BOUNDARY_PATH, save_image_path2, value='low')
    kriging_interpolation_without_boundary(STATION_PATH3, BOUNDARY_PATH, save_image_path3, value='high')
    kriging_interpolation_without_boundary(STATION_PATH4, BOUNDARY_PATH, save_image_path4, value='high')
