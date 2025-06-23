import os
from osgeo import gdal, osr

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cartopy.crs as ccrs

from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

import tqdm

# 设置字体为 arial
plt.rcParams['font.sans-serif'] = ['Arial']

# 文件路径
DIR = os.path.dirname(os.path.abspath(__file__))
WDIR = os.path.join(DIR, '..', '..', 'data', '2020RasterGridsonLandUtilization_GEOTIFF') 
PATH1 = os.path.join(WDIR, 'LUM_end2020.tif')
PATH2 = os.path.join(WDIR, 'ColourMap.clr')


# data\co 下
WDIR2 = os.path.join(DIR, '..', '..', 'data', 'co')

# 文件路径
WDIR3 = os.path.join(DIR, '..', '..', 'data', 'weather', 'output')

PATH3 = os.path.join(WDIR3, 'kriging_RH.tiff')     # 平均（全年）每日 相对湿度
PATH4 = os.path.join(WDIR3, 'kriging_RF.tiff')     # 平均（全年）每日 降雨量
PATH5 = os.path.join(WDIR3, 'kriging_WSPD.tiff')   # 平均（全年）每日 风速
PATH6 = os.path.join(WDIR3, 'kriging_ALLTEMP.tiff')# 平均（全年）每日 温度

targetRowCol = (46, 66)  # 栅格大小（230行 x 330列）

# 加载 tiff 重采样至指定行列大小
def resample_tiff(tiff_path, targetRowCol):
    """重采样 tiff 文件"""
    dataset = gdal.Open(tiff_path)
    warp_ds = gdal.Warp('', dataset, format='MEM',
                        width=targetRowCol[1], height=targetRowCol[0])
    data = warp_ds.ReadAsArray()
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


def load_colormap(cmap_file = PATH2):
    """加载颜色映射表"""
    color_map = {}
    with open(cmap_file, 'r') as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) >= 5:
                value = int(parts[0])
                r, g, b, a = map(lambda x: int(x)/255, parts[1:5])
                color_map[value] = (r, g, b, a)
    colors = [color_map[k] for k in sorted(color_map)]
    cmap = mcolors.ListedColormap(colors)
    bounds = sorted(color_map.keys()) + [max(color_map.keys())+1]
    norm = mcolors.BoundaryNorm(bounds, cmap.N)
    return cmap, norm

def read_image(image_path, targetRowCol):
    """读取并转换栅格数据到WGS84，并重采样至指定行列大小"""
    dataset = gdal.Open(image_path)
    
    # 定义坐标转换
    src_srs = osr.SpatialReference()
    src_srs.ImportFromWkt(dataset.GetProjection())
    dst_srs = osr.SpatialReference()
    dst_srs.ImportFromEPSG(4326)  # WGS84
    
    # 执行坐标转换并重采样
    warp_ds = gdal.Warp('', dataset, format='MEM', dstSRS=dst_srs.ExportToWkt(),
                        width=targetRowCol[1], height=targetRowCol[0])
    
    # print("1")
    
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

    # print(data)
    
    return data, extent

def plot_image(data, cmap, norm, extent):
    """绘制地图"""
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())

    # 绘制栅格数据
    img = ax.imshow(data, cmap=cmap, norm=norm,
                    extent=extent, origin='upper',
                    transform=ccrs.PlateCarree())
    
    # # 添加地图要素
    # ax.add_feature(cfeature.COASTLINE.with_scale('10m'))
    # ax.add_feature(cfeature.BORDERS, linestyle=':', edgecolor='black')

    Lagend = {73: 'Grassland', 71: 'Woodland', 72: 'Shrubland', 74: 'Mangrove/Swamp'}

    
    for key, value in Lagend.items():
        ax.plot([], [], color=cmap(norm(key)), label=value, marker='s', linestyle='None', markersize=10)
    
    # 添加灰色 'others'
    ax.plot([], [], color='gray', label='Others', marker='s', linestyle='None', markersize=10)

    # 右下角
    ax.legend(loc='lower right', fontsize=14)

    # 香港的经纬度范围 113.8, 114.5, 22.1, 22.6
    
    # 配置网格线
    gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray',
                      alpha=0.5, linestyle='--')
    gl.top_labels = True
    gl.left_labels = True
    gl.right_labels = False
    gl.bottom_labels = False
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    
    # 设置显示范围
    ax.set_extent(extent, crs=ccrs.PlateCarree())
    
    plt.show()
    return fig

def save_tiff(data,extent,path):
    driver = gdal.GetDriverByName('GTiff')
    out_ds = driver.Create(path, targetRowCol[1], targetRowCol[0], 1, gdal.GDT_Byte)
    
    lon_min, lon_max, lat_min, lat_max = extent
    x_res = (lon_max - lon_min) / targetRowCol[1]
    y_res = (lat_max - lat_min) / targetRowCol[0]
    geotransform = (lon_min, x_res, 0, lat_max, 0, -y_res)
    out_ds.SetGeoTransform(geotransform)
    
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(4326)
    out_ds.SetProjection(srs.ExportToWkt())
    
    out_band = out_ds.GetRasterBand(1)
    out_band.WriteArray(data)
    out_band.FlushCache()
    out_band.SetNoDataValue(0)
    
    out_ds = None
    print(f"TIFF 文件已保存到: {path}")
    return path


if __name__ == '__main__':

    # 2. 读取气象数据并重采样
    for path in tqdm.tqdm([PATH3, PATH4, PATH5, PATH6]):
        # 读取并转换数据
        data, extent = resample_tiff(path, targetRowCol)
        # 保存结果
        save_tiff(data, extent, os.path.join(WDIR2, os.path.basename(path)))
        print(f'{path} has been plotted.')
        # 打印 shape
        print(data.shape)