import os
from osgeo import gdal
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

# 设置字体为 arial
plt.rcParams['font.sans-serif'] = ['Arial']

# 文件路径
DIR = os.path.dirname(os.path.abspath(__file__))
WDIR = os.path.join(DIR, '..', '..', 'data2', 'weather', 'output')
PDIR = os.path.join(DIR, '..', '..', 'paper', 'imgs')

PATH1 = os.path.join(WDIR, 'kriging_RH.tiff')  # 平均（全年）每日 相对湿度
PATH2 = os.path.join(WDIR, 'kriging_RF.tiff')  # 平均（全年）每日 降雨量
PATH3 = os.path.join(WDIR, 'kriging_WSPD.tiff')# 平均（全年）每日 风速
PATH4 = os.path.join(WDIR, 'kriging_ALLTEMP.tiff')# 平均（全年）每日 温度

# 四个颜色条带风别代表 相对湿度、降雨量、风速、温度
CMAP1 = 'YlGnBu' # for Relative Humidity
CMAP2 = 'Blues' # for Rainfall
CMAP3 = 'OrRd' # for Wind Speed
CMAP4 = 'PuBu' # for Temperature

# save path
SAVE_PATH1 = os.path.join(PDIR, 'kriging_RH.png')
SAVE_PATH2 = os.path.join(PDIR, 'kriging_RF.png')
SAVE_PATH3 = os.path.join(PDIR, 'kriging_WSPD.png')
SAVE_PATH4 = os.path.join(PDIR, 'kriging_ALLTEMP.png')

def read_image(image_path):
    # 由于数据已经是经纬度坐标，所以不需要进行坐标转换
    dataset = gdal.Open(image_path)
    data = dataset.ReadAsArray()
    geotrans = dataset.GetGeoTransform()
    ulx = geotrans[0]
    uly = geotrans[3]
    x_size = dataset.RasterXSize
    y_size = dataset.RasterYSize
    x_res = geotrans[1]
    y_res = geotrans[5]
    x_right = ulx + x_size * x_res
    y_bottom = uly + y_size * y_res
    extent = [ulx, x_right, y_bottom, uly]
    return data, extent

# 添加比例尺（黑白相间，单位为 km）
def add_scalebar(ax, length_deg, location=(0.01, 0.04), linewidth=3):
    """在地图上添加黑白相间的比例尺，单位为公里"""
    # 获取当前坐标轴范围
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()

    # 计算比例尺起止位置
    sb_x = x0 + (x1 - x0) * location[0]
    sb_y = y0 + (y1 - y0) * location[1]

    # 黑白分段
    segments = 4
    segment_length = length_deg / segments

    for i in range(segments):
        start = sb_x + i * segment_length
        end = start + segment_length
        color = 'black' if i % 2 == 0 else 'white'
        ax.plot([start, end], [sb_y, sb_y],
                transform=ccrs.PlateCarree(),
                color=color, linewidth=linewidth, solid_capstyle='butt')

    # 黑色边框
    ax.plot([sb_x, sb_x + length_deg], [sb_y, sb_y],
            transform=ccrs.PlateCarree(), color='k', linewidth=1)

    # 将角度转换为大约公里数（近似）
    length_km = length_deg * 111.32

    # 添加文字（单位为 km）
    ax.text(sb_x + length_deg / 2, sb_y - (y1 - y0) * 0.01,
            f'{int(length_km)} km', transform=ccrs.PlateCarree(),
            ha='center', va='top', fontsize=10)


# 添加指北针（修正方向）
def add_north_arrow(ax, location=(0.9, 0.95), size=30):
    """在地图上添加指北针箭头（修正方向）"""
    # 使用annotate绘制指北针
    ax.annotate('', 
                xy=location, 
                xytext=(location[0], location[1] - 0.03),  # 箭头指向下方
                xycoords='axes fraction',
                arrowprops=dict(facecolor='black', width=4, headwidth=10, headlength=10))

def plot_image(data, extent, CMAP, legendName, SAVE_PATH):
    # 绘制等值线并添加色标
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

    ax.contourf(data, transform=ccrs.PlateCarree(), extent=extent, cmap=CMAP)
    
    # 修改颜色条带为竖放，并置于右上角
    cbar = plt.colorbar(ax.contourf(data, transform=ccrs.PlateCarree(), extent=extent, cmap=CMAP), 
                       ax=ax, orientation='vertical', label=legendName, pad=0.05, fraction=0.05)
    cbar.ax.set_position([0.85, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
    cbar.ax.set_facecolor('white')
    cbar.ax.yaxis.set_major_locator(plt.MaxNLocator(nbins=5))  # Set fewer ticks
    
    # 旋转颜色条带的文字为90°
    for label in cbar.ax.get_yticklabels():
        label.set_rotation(90)

    # 添加等值线 标注出数据的分布
    lon_min, lon_max, lat_min, lat_max = extent[0], extent[1], extent[2], extent[3]
    x = np.linspace(lon_min, lon_max, data.shape[1])
    y = np.linspace(lat_min, lat_max, data.shape[0])
    X, Y = np.meshgrid(x, y)
    contour = ax.contour(X, Y, data, colors='white', linewidths=0.7, levels=10)
    ax.clabel(contour, inline=True, fontsize=12, fmt='%1.1f')
    
    add_scalebar(ax, length_deg=0.11)  # 根据地图范围调节这个值
    add_north_arrow(ax)

    plt.savefig(SAVE_PATH, dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == '__main__':
    data, extent = read_image(PATH1)
    plot_image(data, extent, CMAP1, '14 Years Mean Relative Humidity (%)', SAVE_PATH1)
    print(f'{PATH1} has been plotted.')

    data, extent = read_image(PATH2)
    plot_image(data, extent, CMAP2, '14 Years Mean Rainfall (mm)', SAVE_PATH2)
    print(f'{PATH2} has been plotted.')

    data, extent = read_image(PATH3)
    plot_image(data, extent, CMAP3, '14 Years Mean Wind Speed (m/s)', SAVE_PATH3)
    print(f'{PATH3} has been plotted.')

    data, extent = read_image(PATH4)
    plot_image(data, extent, CMAP4, '14 Years Mean Temperature (°C)', SAVE_PATH4)
    print(f'{PATH4} has been plotted.')