import os
import numpy as np
from osgeo import gdal, osr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

# 设置字体为 Arial
plt.rcParams['font.sans-serif'] = ['Arial']

# 文件路径
DIR = os.path.dirname(os.path.abspath(__file__))
WDIR = os.path.join(DIR, '..', '..', 'data', 'WUI')

PATH1 = os.path.join(WDIR, 'Baseline_2020_Intermix_WUI.tif') # 二值
PATH2 = os.path.join(WDIR, 'ppp_2020_1km_Aggregated.tif') # 连续 > 0

# 输出路径
WDIR2 = os.path.join(DIR, '..', '..', 'data', 'co')

SAVEPATH = os.path.join(WDIR2, 'Baseline_2020_Intermix_WUI.tif')
SAVEPATH2 = os.path.join(WDIR2, 'ppp_2020_1km_Aggregated.tif')

# 绘图输出路径
PLOT_PATH1 = os.path.join(WDIR2, 'Baseline_2020_Intermix_WUI_plot.png')
PLOT_PATH2 = os.path.join(WDIR2, 'ppp_2020_1km_Aggregated_plot.png')

# 目标栅格大小
targetRowCol = (46, 66)  # 栅格大小（46行 x 66列）

# 栅格参数
extent = [113.8242, 114.4441, 22.1380, 22.5719]

def is_binary_data(data):
    """检查数据是否为二值数据"""
    unique_values = np.unique(data[~np.isnan(data)])
    return len(unique_values) <= 2

def determine_colormap(data):
    """根据数据特性确定合适的colormap"""
    if is_binary_data(data):
        return 'binary', 'Value'
    else:  # 连续数据
        return 'viridis', 'Value'

def print_statistics(data, name="Data"):
    """打印数据的统计信息"""
    valid_data = data[~np.isnan(data)]
    print(f"\n{name} Statistics:")
    print(f"Min: {np.min(valid_data):.4f}")
    print(f"Max: {np.max(valid_data):.4f}")
    print(f"Mean: {np.mean(valid_data):.4f}")
    print(f"Median: {np.median(valid_data):.4f}")
    print(f"Std Dev: {np.std(valid_data):.4f}")
    if is_binary_data(data):
        unique, counts = np.unique(valid_data, return_counts=True)
        print("Value Counts:")
        for val, cnt in zip(unique, counts):
            print(f"  {val}: {cnt} ({cnt/len(valid_data):.2%})")

def read_and_crop_image(image_path, extent=extent, default_gt0=True):
    """读取栅格数据并裁剪到指定范围"""
    dataset = gdal.Open(image_path)
    if dataset is None:
        raise ValueError(f"无法打开文件: {image_path}")
    
    # 获取原始地理信息
    geo_transform = dataset.GetGeoTransform()
    x_origin = geo_transform[0]
    y_origin = geo_transform[3]
    x_res = geo_transform[1]
    y_res = geo_transform[5]
    
    # 计算裁剪范围的行列索引
    x_min, x_max, y_min, y_max = extent
    col_min = int((x_min - x_origin) / x_res)
    col_max = int((x_max - x_origin) / x_res)
    row_min = int((y_max - y_origin) / y_res)  # y_res is negative
    row_max = int((y_min - y_origin) / y_res)
    
    # 确保不超出范围
    cols = dataset.RasterXSize
    rows = dataset.RasterYSize
    col_min = max(0, col_min)
    col_max = min(cols, col_max)
    row_min = max(0, row_min)
    row_max = min(rows, row_max)
    
    # 读取裁剪后的数据
    data = dataset.ReadAsArray(col_min, row_min, 
                              col_max - col_min, 
                              row_max - row_min)
    data = np.array(data, dtype=np.float32)

    # 若默认大于 0 则将所有小于等于0替换为0
    if default_gt0 :
        data[data <= 0] = 0
    
    # 计算裁剪后的实际范围
    actual_x_min = x_origin + col_min * x_res
    actual_x_max = x_origin + col_max * x_res
    actual_y_max = y_origin + row_min * y_res
    actual_y_min = y_origin + row_max * y_res
    actual_extent = [actual_x_min, actual_x_max, actual_y_min, actual_y_max]
    
    # 关闭数据集
    dataset = None
    
    return data, actual_extent

def resample_image(data, extent, target_size, is_binary=False):
    """将图像重采样到目标大小"""
    from scipy.ndimage import zoom
    
    # 计算缩放因子
    current_rows, current_cols = data.shape
    row_zoom = target_size[0] / current_rows
    col_zoom = target_size[1] / current_cols
    
    # 使用不同的插值方法
    if is_binary:
        # 对于二值数据使用最近邻插值
        resampled_data = zoom(data, (row_zoom, col_zoom), order=0)
        # 将结果二值化
        resampled_data = np.where(resampled_data > 0.5, 1, 0)
    else:
        # 对于连续数据使用线性插值
        resampled_data = zoom(data, (row_zoom, col_zoom), order=1)
    
    return resampled_data

def save_tiff(data, extent, path, is_binary=False):
    """保存栅格数据为 TIFF 文件"""
    driver = gdal.GetDriverByName('GTiff')
    rows, cols = data.shape
    
    out_ds = driver.Create(path, cols, rows, 1, gdal.GDT_Float32)
    
    lon_min, lon_max, lat_min, lat_max = extent
    x_res = (lon_max - lon_min) / cols
    y_res = (lat_max - lat_min) / rows
    geotransform = (lon_min, x_res, 0, lat_max, 0, -y_res)
    out_ds.SetGeoTransform(geotransform)
    
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(4326)
    out_ds.SetProjection(srs.ExportToWkt())

    out_band = out_ds.GetRasterBand(1)
    if is_binary:
        # 对于二值数据，保存为整数
        out_band.WriteArray(data.astype(np.int8))
    else:
        out_band.WriteArray(data)
    out_band.SetNoDataValue(-9999)
    out_band.FlushCache()
    out_ds = None
    
    print(f"TIFF 文件已保存到: {path}")
    return path

def plot_image(data, extent, save_path):
    """绘制图像并根据数据类型自动选择colormap"""
    cmap, legend_name = determine_colormap(data)
    
    plt.figure(figsize=(8, 6))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent(extent)
    ax.coastlines(color='black', linestyle='--')

    gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray',
                      alpha=0.5, linestyle='--')
    gl.top_labels = True
    gl.left_labels = True
    gl.right_labels = False
    gl.bottom_labels = False
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER

    # 设置经纬网标签字体大小为12
    gl.xlabel_style = {'size': 12}
    gl.ylabel_style = {'size': 12}

    # 绘制数据
    im = ax.imshow(data, extent=extent, transform=ccrs.PlateCarree(), 
                  cmap=cmap, origin='upper')
    
    # 添加颜色条
    cbar = plt.colorbar(im, ax=ax, orientation='vertical', 
                        label=legend_name, pad=0.05, fraction=0.05)
    cbar.ax.set_position([0.85, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
    cbar.ax.set_facecolor('white')

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"绘图已保存到: {save_path}")

def process_image(input_path, output_path, plot_path, extent, target_size):
    """完整的处理流程"""
    # 1. 读取并裁剪图像
    print(f"\n处理文件: {os.path.basename(input_path)}")
    print("读取并裁剪图像...")
    data, cropped_extent = read_and_crop_image(input_path, extent)
    
    # 2. 打印统计信息
    print_statistics(data, "原始裁剪数据")
    
    # 3. 绘制原始裁剪后的图像
    print("绘制原始裁剪图像...")
    plot_image(data, cropped_extent, plot_path.replace('.png', '_original.png'))
    
    # 4. 重采样图像
    print("重采样图像...")
    is_binary = is_binary_data(data)
    resampled_data = resample_image(data, cropped_extent, target_size, is_binary)
    
    # 5. 打印重采样后的统计信息
    print_statistics(resampled_data, "重采样数据")
    
    # 6. 绘制重采样后的图像
    print("绘制重采样图像...")
    plot_image(resampled_data, extent, plot_path)
    
    # 7. 保存结果
    save_tiff(resampled_data, extent, output_path, is_binary)
    
    return resampled_data

if __name__ == '__main__':
    # 确保输出目录存在
    os.makedirs(WDIR2, exist_ok=True)
    
    # 处理第一个文件 (二值数据)
    process_image(PATH1, SAVEPATH, PLOT_PATH1, extent, targetRowCol)
    
    # 处理第二个文件 (连续数据)
    process_image(PATH2, SAVEPATH2, PLOT_PATH2, extent, targetRowCol)