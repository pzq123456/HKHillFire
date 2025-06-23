import os
import numpy as np
from osgeo import gdal, osr
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import tqdm

# 设置字体为 Arial
plt.rcParams['font.sans-serif'] = ['Arial']

# 文件路径
DIR = os.path.dirname(os.path.abspath(__file__))
WDIR = os.path.join(DIR, '..', '..', 'data', '2020RasterGridsonLandUtilization_GEOTIFF')
PATH1 = os.path.join(WDIR, 'LUM_end2020.tif')
PATH2 = os.path.join(WDIR, 'ColourMap.clr')

# 输出路径
WDIR2 = os.path.join(DIR, '..', '..', 'data', 'co')
SAVEPATH = os.path.join(WDIR2, 'downLUM_end2020_2.tif')

# 目标栅格大小
targetRowCol = (46, 66)  # 栅格大小（46行 x 66列）

# 植被栅格值
VEGETATION_VALUES = {71, 72, 73, 74}

def read_image(image_path = PATH1):
    """读取并转换栅格数据到WGS84"""
    dataset = gdal.Open(image_path)
    
    # 定义坐标转换
    src_srs = osr.SpatialReference()
    src_srs.ImportFromWkt(dataset.GetProjection())
    dst_srs = osr.SpatialReference()
    dst_srs.ImportFromEPSG(4326)  # WGS84
    
    # 执行坐标转换
    warp_ds = gdal.Warp('', dataset, format='MEM', dstSRS=dst_srs.ExportToWkt())
    
    # 获取转换后数据
    data = warp_ds.ReadAsArray()
    
    # 计算地理范围
    geotrans = warp_ds.GetGeoTransform()
    ulx = geotrans[0]
    uly = geotrans[3]
    x_size = warp_ds.RasterXSize
    y_size = warp_ds.RasterYSize
    x_res = geotrans[1]
    y_res = geotrans[5]
    
    x_right = ulx + x_size * x_res
    y_bottom = uly + y_size * y_res
    extent = [ulx, x_right, y_bottom, uly]
    
    return data, extent

def calculate_vegetation_ratio(data, vegetation_values, targetRowCol):
    """分块计算每个栅格中植被所占的比率"""
    # 获取原始数据的行列数
    height, width = data.shape
    
    # 计算每个降采样栅格对应的原始栅格块大小
    block_height = height // targetRowCol[0]
    block_width = width // targetRowCol[1]
    
    # 初始化结果数组
    ratio = np.zeros(targetRowCol, dtype=np.float32)  # 使用浮点型存储
    
    # 遍历每个块
    for i in range(targetRowCol[0]):
        for j in range(targetRowCol[1]):
            # 获取当前块的原始数据
            block = data[i * block_height:(i + 1) * block_height,
                         j * block_width:(j + 1) * block_width]
            
            # 计算当前块中植被的比例
            vegetation_mask = np.isin(block, list(vegetation_values))
            vegetation_pixels = np.sum(vegetation_mask)  # 植被像素的数量
            total_pixels = block_height * block_width    # 块的总像素数
            ratio[i, j] = vegetation_pixels / total_pixels  # 植被比例
            # print(f"Block ({i}, {j}): Vegetation ratio = {ratio[i, j]}")
    
    return ratio

def save_tiff(data, extent, path):
    """保存栅格数据为 TIFF 文件"""
    driver = gdal.GetDriverByName('GTiff')
    out_ds = driver.Create(path, targetRowCol[1], targetRowCol[0], 1, gdal.GDT_Float32)
    
    lon_min, lon_max, lat_min, lat_max = extent
    x_res = (lon_max - lon_min) / targetRowCol[1]
    y_res = (lat_max - lat_min) / targetRowCol[0]
    geotransform = (lon_min, x_res, 0, lat_max, 0, -y_res)
    out_ds.SetGeoTransform(geotransform)
    
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(4326)
    out_ds.SetProjection(srs.ExportToWkt())

    # 设置为 float32 类型
    out_band = out_ds.GetRasterBand(1)
    out_band.WriteArray(data)
    out_band.SetNoDataValue(-9999)
    out_band.FlushCache()

    out_ds = None
    print(f"TIFF 文件已保存到: {path}")
    return path

if __name__ == '__main__':
    # 1. 读取 LUM_end2020.tif 并重采样
    data, extent = read_image(PATH1)
    
    # 2. 计算植被比率
    vegetation_ratio = calculate_vegetation_ratio(data, VEGETATION_VALUES, targetRowCol)

    # print(vegetation_ratio.shape)    
    # 3. 保存结果
    save_tiff(vegetation_ratio, extent, SAVEPATH)