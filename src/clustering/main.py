import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from osgeo import gdal, osr
import tqdm

# 额外引入
from scipy.ndimage import generic_filter
from sklearn.cluster import KMeans

gdal.UseExceptions()

# ---------------------------
# 全局设置与常量定义
# ---------------------------
plt.rcParams['font.sans-serif'] = ['Arial']

# 文件路径
DIR = os.path.dirname(os.path.abspath(__file__))
WDIR = os.path.join(DIR, '..', '..', 'data2', 'files')
DATA_PATH = os.path.join(WDIR, 'all.csv')
TIFF_PATH = os.path.abspath(os.path.join(WDIR, 'fire_classification2.tiff'))

WDIR2 = os.path.join(DIR, '..', '..', 'data2', 'co')
SAVEPATH = os.path.join(WDIR2, 'downLUM_end2020.tif')

# 栅格参数
extent = [113.8242, 114.4441, 22.1380, 22.5719]
rowCol = (46, 66)  # (行, 列)

# 火灾类别映射
CATEGORY_MAP = {
    'NF': 0,   # 无火灾（空白）
    'RSD': 1,  # 稀有小火干季
    'RSW': 2,  # 稀有小火雨季
    'RLD': 3,  # 稀有大火干季
    'RLW': 4,  # 稀有大火雨季
    'CSD': 5,  # 常见小火干季
    'CSW': 6,  # 常见小火雨季
    'CLD': 7,  # 常见大火干季
    'CLW': 8,  # 常见大火雨季
}

COLOR_MAP = {
    'NF': 'aliceblue',
    'RSD': 'lightcoral',
    'RSW': 'lightsalmon',
    'RLD': 'darkred',
    'RLW': 'red',
    'CSD': 'lightgreen',
    'CSW': 'lime',
    'CLD': 'darkgreen',
    'CLW': 'green',
    'bg': 'white'
}

# ---------------------------
# 工具函数
# ---------------------------
def lonlat_to_xy(lon, lat):
    """
    将经纬度转换为栅格索引
    """
    lon_min, lon_max, lat_min, lat_max = extent
    x_res = (lon_max - lon_min) / rowCol[1]
    y_res = (lat_max - lat_min) / rowCol[0]
    x = int((lon - lon_min) / x_res)
    y = int((lat_max - lat) / y_res)
    return (x, y)

def classify_event(row):
    """
    根据火灾频率、规模和时间（火灾季/非火灾季）对事件进行分类
    """
    freq = row['frequency']
    size = row['size']
    season = row['season']

    if freq == 'Rare' and size == 'Small' and season == 'Dry':
        return 'RSD'
    elif freq == 'Rare' and size == 'Small' and season == 'Wet':
        return 'RSW'
    elif freq == 'Rare' and size == 'Large' and season == 'Dry':
        return 'RLD'
    elif freq == 'Rare' and size == 'Large' and season == 'Wet':
        return 'RLW'
    elif freq == 'Common' and size == 'Small' and season == 'Dry':
        return 'CSD'
    elif freq == 'Common' and size == 'Small' and season == 'Wet':
        return 'CSW'
    elif freq == 'Common' and size == 'Large' and season == 'Dry':
        return 'CLD'
    elif freq == 'Common' and size == 'Large' and season == 'Wet':
        return 'CLW'
    else:
        return 'NF'

def create_tiff(arr):
    """
    根据栅格数组生成 GeoTIFF 文件
    """
    driver = gdal.GetDriverByName('GTiff')
    if not os.path.exists(WDIR):
        os.makedirs(WDIR)
    out_ds = driver.Create(TIFF_PATH, rowCol[1], rowCol[0], 1, gdal.GDT_Byte)

    lon_min, lon_max, lat_min, lat_max = extent
    x_res = (lon_max - lon_min) / rowCol[1]
    y_res = (lat_max - lat_min) / rowCol[0]
    geotransform = (lon_min, x_res, 0, lat_max, 0, -y_res)
    out_ds.SetGeoTransform(geotransform)

    srs = osr.SpatialReference()
    srs.ImportFromEPSG(4326)
    out_ds.SetProjection(srs.ExportToWkt())

    out_band = out_ds.GetRasterBand(1)
    out_band.WriteArray(arr)
    out_band.FlushCache()
    out_ds = None

def plot_tiff(tiff_path):
    """
    可视化 GeoTIFF 数据
    """
    print("正在读取 TIFF 文件...")
    dataset = gdal.Open(tiff_path)
    if dataset is None:
        print("无法读取 TIFF 文件")
        return

    band = dataset.GetRasterBand(1)
    data = band.ReadAsArray()

    cmap = mcolors.ListedColormap(list(COLOR_MAP.values()))
    bounds = list(CATEGORY_MAP.values())
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    geotransform = dataset.GetGeoTransform()
    left = geotransform[0]
    right = geotransform[0] + geotransform[1] * dataset.RasterXSize
    top = geotransform[3]
    bottom = geotransform[3] + geotransform[5] * dataset.RasterYSize
    ext = (left, right, bottom, top)

    plt.figure(figsize=(10, 10))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent(ext, crs=ccrs.PlateCarree())
    ax.coastlines(color='black', linestyle='--')

    gl = ax.gridlines(draw_labels=True, linewidth=1, color='gray', alpha=0.5, linestyle='--')
    gl.xformatter, gl.yformatter = LONGITUDE_FORMATTER, LATITUDE_FORMATTER

    ax.imshow(data, extent=ext, origin='upper', cmap=cmap, norm=norm)

    legend_elements = [plt.Line2D([0], [0], color=color, marker='s', linestyle='None',
                          markersize=10, markeredgewidth=1, markeredgecolor='black', label=label)
                          for label, color in COLOR_MAP.items()]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=14, ncol=1)

    plt.show()



# ---------------------------
# 主处理流程
# ---------------------------
def cluster(df):
    # 读取数据
    df = pd.read_csv(DATA_PATH)
    df['Date'] = pd.to_datetime(df['Date'])
    df['Year'] = df['Date'].dt.year

    df['grid_coord'] = df.apply(lambda row: lonlat_to_xy(row['Longitude'], row['Latitude']), axis=1)

    # 计算每个栅格的年平均火灾频率
    grid_year_counts = df.groupby(['grid_coord', 'Year']).size().reset_index(name='yearly_count')
    grid_avg_counts = grid_year_counts.groupby('grid_coord')['yearly_count'].mean().reset_index(name='avg_count')

    threshold_freq = 2
    threshold_area = 10000
    threshold_humidity = 75

    df = df.merge(grid_avg_counts, on='grid_coord', how='left')
    df['frequency'] = np.where(df['avg_count'] > threshold_freq, 'Common', 'Rare')
    df['size'] = np.where(df['Area (M2)'] > threshold_area, 'Large', 'Small')

    # Humidity 根据湿度划分季节
    df['season'] = np.where(df['Humidity'] > threshold_humidity, 'Wet', 'Dry')


    # 分类
    df['category'] = df.apply(classify_event, axis=1)

    # 统计每类火灾的数量
    valid_categories = df[df['category'].isin(CATEGORY_MAP.keys())]
    category_counts = valid_categories['category'].value_counts()
    category_area = valid_categories.groupby('category')['Area (M2)'].sum()

    print("火灾分类统计（数量）：\n", category_counts)
    print("\n火灾分类统计（总面积）：\n", category_area)

    # 创建栅格数据
    arr = np.zeros(rowCol, dtype=np.uint8)

    # 按栅格分组，取每个栅格的主要类别
    grid_categories = valid_categories.groupby('grid_coord')['category'].agg(
        lambda x: x.mode()[0] if not x.empty else None
    ).reset_index()

    for _, row in tqdm.tqdm(grid_categories.iterrows(), total=len(grid_categories)):
        x, y = row['grid_coord']
        category = row['category']
        if category and 0 <= x < rowCol[1] and 0 <= y < rowCol[0]:
            arr[y, x] = CATEGORY_MAP[category]
    return arr

def post_process(arr):
    """
    根据参考栅格数据对生成的分类数组进行后处理
    """
    ds = gdal.Open(SAVEPATH)
    if ds is None:
        raise ValueError(f"无法打开文件: {SAVEPATH}")

    band = ds.GetRasterBand(1)
    data = band.ReadAsArray()
    data = np.where(data > 0, 1, 0)
    arr = np.where(data == 1, arr, -1)
    ds = None
    return arr

# ---------------------------
# 主函数入口
# ---------------------------
def main():
    df = pd.read_csv(DATA_PATH)
    # 聚类与分类：可通过参数切换不同平衡策略
    arr = cluster(df)
    arr = post_process(arr)
    create_tiff(arr)
    plot_tiff(TIFF_PATH)

if __name__ == '__main__':
    main()
