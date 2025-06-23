import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
import tqdm

# 设置字体为 arial
plt.rcParams['font.sans-serif'] = ['Arial']

DIR = os.path.dirname(os.path.abspath(__file__))
PATH = os.path.join(DIR, '..', '..', 'data2', 'files', 'all.csv')

WDIR = os.path.join(DIR, '..', '..', 'data2','weather','output')

PATH_ALLTEMP = os.path.join(WDIR, 'ALLTEMP')
PATH_RF = os.path.join(WDIR, 'RF')
PATH_RH = os.path.join(WDIR, 'RH')
PATH_WSPD = os.path.join(WDIR, 'WSPD')

# 需要首先运行 weather_preprocess(PATH_ALLTEMP) 生成 mean_all_data.csv
PATH_ALLTEMP2 = os.path.join(WDIR, 'ALLTEMP', 'mean_all_data.csv')
PATH_RF2 = os.path.join(WDIR, 'RF', 'mean_all_data.csv')
PATH_RH2 = os.path.join(WDIR, 'RH', 'mean_all_data.csv')
PATH_WSPD2 = os.path.join(WDIR, 'WSPD', 'mean_all_data.csv')

# 气象数据预处理程序
def weather_preprocess(PATH):
       # 首先扫描当前文件夹下所有的csv
       files = os.listdir(PATH)
       # print(files)
       # 检查后缀并 读取所有的csv文件
       dfs = []
       for file in tqdm.tqdm(files):
              if file.endswith('.csv'):
                     df = pd.read_csv(os.path.join(PATH, file))
                     # 若无头 则增加一行 Date, Value

                     dfs.append(df)

       return merge_data(dfs).to_csv(os.path.join(PATH, 'mean_all_data.csv'))

def merge_data(dfs):
       # 合并数据 date 不动 将同一天的所有 value 取平均值 忽略缺失值

       # 1. 合并数据
       df = pd.concat(dfs, ignore_index=True)

       # 2. 转换日期格式
       df['Date'] = pd.to_datetime(df['Date'])

       # 去除所有空值 Trace 替换为空值而后去除
       df['Value'] = pd.to_numeric(df['Value'], errors='coerce')

       # 3. 按照日期分组
       df = df.groupby('Date').mean()

       # 4. 保存数据
       return df

# 按年份可视化气象数据

def yearly_weather_boxplot(df, ylabel):
       # 将日期列转换为日期格式
       df['Date'] = pd.to_datetime(df['Date'])
       
       # 提取年份
       df['Year'] = df['Date'].dt.year

       # print(df)

       # 按照年份分组并绘制 boxplot
       fig, ax = plt.subplots(figsize=(12, 8))

       # 绘制箱型图
       box = ax.boxplot([df[df['Year'] == year]['Value'].values for year in range(2010, 2021)], positions=np.arange(2010, 2021), widths=0.6, patch_artist=True,
                                    showmeans=True, showfliers=False,
                                    medianprops={"color": "orange", "linewidth": 1.5},
                                    meanprops={"marker": "o", "markerfacecolor": "white", "markeredgecolor": "gray", "markersize": 8},
                                    boxprops={"facecolor": "lightblue", "edgecolor": "gray", "linewidth": 1.5},
                                    whiskerprops={"color": "gray", "linewidth": 1.5},
                                    capprops={"color": "gray", "linewidth": 1.5})
       
       # 设置图表的标签
       ax.set(xlabel='Year', ylabel=ylabel, ylim=(df['Value'].min() - 5, df['Value'].max() + 5),
                 xticks=np.arange(2010, 2021))
       
       plt.show()


# 统计2010 - 2020 逐年的火灾数量
def yearly_fire_frequency_boxplot(df):
       # 将日期列转换为日期格式
       df['Date'] = pd.to_datetime(df['Date'])
       
       # 提取年份
       df['Year'] = df['Date'].dt.year
       
       # 转化为 
       # 也即是一年12个月的数据

       yearly_data = [df[df['Year'] == year].groupby(df['Date'].dt.month).size().values for year in range(2010, 2021)]

       # 获取每一年的总火灾数量
       yearly_counts = df.groupby('Year').size()

       # print(yearly_counts)
       # print(yearly_data) # { 2010: [1-12 12 months data], 2011: [1-12 12 months data], ... } 的形式

       # 创建一个图表来展示箱型图
       fig, ax = plt.subplots(figsize=(12, 8))


       # 绘制箱型图
       box = ax.boxplot(yearly_data, positions=np.arange(2010, 2021), widths=0.6, patch_artist=True,
                                    showmeans=True, showfliers=False,
                                    medianprops={"color": "orange", "linewidth": 1.5},
                                    meanprops={"marker": "o", "markerfacecolor": "white", "markeredgecolor": "gray", "markersize": 8},
                                    boxprops={"facecolor": "lightblue", "edgecolor": "gray", "linewidth": 1.5},
                                    whiskerprops={"color": "gray", "linewidth": 1.5},
                                    capprops={"color": "gray", "linewidth": 1.5})
       
       # 叠加浅灰色柱状图
       ax.bar(yearly_counts.index, yearly_counts.values, zorder=0, edgecolor='gray', fill=False)

       # 柱状图顶部显示具体数值
       for i, count in enumerate(yearly_counts):
              ax.text(i + 2010, count + 5, str(count), ha='center', va='bottom', fontsize=FONTSIZE, color='gray')
       
       # 设置图表的标签
       ax.set(xlabel='Year', ylabel='Fire Frequency (n/month)',
                 xticks=np.arange(2010, 2021), ylim=(0, yearly_counts.max() + 50))
       
       plt.show()

def yearly_fire_frequency_boxplot2(df):
       plt.rcParams.update({'font.size': 14})

       # 将日期列转换为日期格式
       df['Date'] = pd.to_datetime(df['Date'])

       # 提取年份
       df['Year'] = df['Date'].dt.year

       # 统计每年的过火面积
       yearly_data = [df[df['Year'] == year].groupby(df['Date'].dt.month)['Area (M2)'].sum().values for year in range(2010, 2025)]
       # 对每一个月的数据进行求和
       yearly_data = [sum(data) for data in yearly_data]
       # 将平方米转化为平方公里
       yearly_data = [m2_to_km2(data) for data in yearly_data]

       # 获取每一年的总火灾数量
       yearly_counts = df.groupby('Year').size()

       # 创建一个图表来展示箱型图
       fig, ax = plt.subplots(figsize=(12, 8))

       # 叠加浅灰色柱状图
       bars = ax.bar(yearly_counts.index, yearly_counts.values, zorder=0, fill=True, color='lightgray', label='Fire Frequency', linewidth=1.5)

       # 柱状图顶部显示具体数值
       for bar in bars:
              height = bar.get_height()
              ax.text(bar.get_x() + bar.get_width() / 2, height + 5, f'{height}', ha='center', va='bottom', fontsize=FONTSIZE, color='black')
       
       # log scale y 放在右边
       ax2 = ax.twinx()

       # 绘制折线图
       ax2.plot(yearly_counts.index, yearly_data, marker='o', markersize=8, linewidth=2, color='tab:blue', label='Total Burnt Area (km²)')
       ax2.set_ylabel('Total Burnt Area(km²)', fontsize=FONTSIZE, color='tab:blue')
       ax2.tick_params(axis='y', labelsize=12, colors='tab:blue')
       ax2.grid(False)

       # 设置图表的标签
       ax.set_xlabel('Year', fontsize=FONTSIZE)
       ax.set_ylabel('Fire Frequency', fontsize=FONTSIZE, color='dimgrey')
       ax.set_xticks(yearly_counts.index)
       ax.set_ylim(0, yearly_counts.max() + 50)
       ax.tick_params(axis='y', labelsize=LABL_FONTSIZE, colors='dimgrey')
       ax.tick_params(axis='x', labelsize=LABL_FONTSIZE)


       # # 调整图例位置
       # lines, labels = ax.get_legend_handles_labels()
       # lines2, labels2 = ax2.get_legend_handles_labels()
       # ax.legend(lines + lines2, labels + labels2, loc='upper left', fontsize=12)

       # 移除多余的网格线
       ax.grid(False)

       plt.tight_layout()
       plt.show()


FONTSIZE= 18
LABL_FONTSIZE = 14


def monthly_fire_frequency_boxplot2(df):
    
       # 将日期列转换为日期格式
       df['Date'] = pd.to_datetime(df['Date'])

       # 提取月份和年份
       df['Month'] = df['Date'].dt.month

       # 统计每个月的过火面积
       monthly_data = [df[df['Month'] == month].groupby(df['Date'].dt.year)['Area (M2)'].sum().values for month in range(1, 13)]
       # 对每一个月的数据进行求和
       monthly_data = [sum(data) for data in monthly_data]
       # 将平方米转化为平方公里
       monthly_data = [m2_to_km2(data) for data in monthly_data]

       # 获取每个月的频率列表
       monthly_counts = df.groupby('Month').size()

       fig, ax = plt.subplots(figsize=(12, 8))

       # 叠加浅灰色柱状图
       bars = ax.bar(np.arange(1, 13), monthly_counts.values, zorder=0, fill=True, color='lightblue', label='Fire Frequency', linewidth=1.5)

       # 柱状图顶部显示具体数值
       for bar in bars:
              height = bar.get_height()
              ax.text(bar.get_x() + bar.get_width() / 2, height + 5, f'{height}', ha='center', va='bottom', fontsize=FONTSIZE, color='black')

       # log scale y 放在右边
       ax2 = ax.twinx()

       # 绘制折线图
       ax2.plot(np.arange(1, 13), monthly_data, marker='o', markersize=8, linewidth=2, color='tab:orange', label='Total Burnt Area(km²)')
       ax2.set_ylabel('Total Burnt Area(km²)', fontsize=FONTSIZE, color='tab:orange')
       ax2.tick_params(axis='y', labelsize=LABL_FONTSIZE, colors='tab:orange')
       ax2.grid(False)

       # 设置图表的标签
       ax.set_xlabel('Month', fontsize=FONTSIZE)
       ax.set_xticks(np.arange(1, 13))
       ax.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'], fontsize=FONTSIZE)

       ax.set_ylabel('Fire Frequency', fontsize=FONTSIZE, color='darkblue')
       ax.set_ylim(0, monthly_counts.max() + 50)

       ax.tick_params(axis='x', labelsize=LABL_FONTSIZE)
       ax.tick_params(axis='y', labelsize=LABL_FONTSIZE, colors='darkblue')

       # 调整图例位置
       #     lines, labels = ax.get_legend_handles_labels()
       #     lines2, labels2 = ax2.get_legend_handles_labels()
       #     ax.legend(lines + lines2, labels + labels2, loc='upper left', fontsize=12)

       # 移除多余的网格线
       ax.grid(False)

       plt.tight_layout()
       plt.show()


def monthly_fire_frequency_boxplot(df):
       # 将日期列转换为日期格式
       df['Date'] = pd.to_datetime(df['Date'])
       
       # 提取月份和年份
       df['Month'] = df['Date'].dt.month
       
       # # 计算每月的火灾发生频率
       # monthly_counts = df.groupby(['Year', 'Month']).size().reset_index(name='Frequency')
       
       # # 获取每个月的频率列表
       # monthly_data = [monthly_counts[monthly_counts['Month'] == month]['Frequency'].values for month in range(1, 13)]

       monthly_data = [df[df['Month'] == month].groupby(df['Date'].dt.year).size().values for month in range(1, 13)]

       # 获取每个月的频率列表
       monthly_counts = df.groupby('Month').size()

       # print(monthly_data) # { Jan: [2010-2020 11 years data], Feb: [2010-2020 11 years data], ... }
       
       # 创建一个图表来展示箱型图
       fig, ax = plt.subplots(figsize=(12, 8))
       
       # 绘制箱型图
       box = ax.boxplot(monthly_data, positions=np.arange(1, 13), widths=0.6, patch_artist=True,
                                    showmeans=True, showfliers=False,
                                    medianprops={"color": "orange", "linewidth": 1.5},
                                    meanprops={"marker": "o", "markerfacecolor": "white", "markeredgecolor": "gray", "markersize": 8},
                                    boxprops={"facecolor": "lightblue", "edgecolor": "gray", "linewidth": 1.5},
                                    whiskerprops={"color": "gray", "linewidth": 1.5},
                                    capprops={"color": "gray", "linewidth": 1.5})
       
       # 叠加浅灰色柱状图
       ax.bar(np.arange(1, 13), monthly_counts.values, zorder=0, edgecolor='gray', fill=False)

       # 柱状图顶部显示具体数值
       for i, count in enumerate(monthly_counts):
              ax.text(i + 1, count + 5, str(count), ha='center', va='bottom', fontsize=FONTSIZE, color='gray')
       
       # 设置图表的标签
       ax.set(xlabel='Month', ylabel='Fire Frequency (n/year)',
                 xticks=np.arange(1, 13), xticklabels=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'],
                 ylim=(0, monthly_counts.max() + 50))
       
       
       
       
       # 局部放大 5-9 月的箱型图
       axins = ax.inset_axes([0.3, 0.3, 0.3, 0.3])

       box = axins.boxplot(monthly_data[4:9], positions=np.arange(5, 10), widths=0.6, patch_artist=True,
                                          showmeans=True, showfliers=False,
                                          medianprops={"color": "orange", "linewidth": 1.5},
                                          meanprops={"marker": "o", "markerfacecolor": "white", "markeredgecolor": "gray", "markersize": 8},
                                          boxprops={"facecolor": "lightblue", "edgecolor": "gray", "linewidth": 1.5},
                                          whiskerprops={"color": "gray", "linewidth": 1.5},
                                          capprops={"color": "gray", "linewidth": 1.5})
       
       # 虚线框标出被放大的区域并指向放大区域
       ax.indicate_inset_zoom(axins, edgecolor="gray", linestyle="--")
       
       plt.show()

# 辅助函数 将平方米转化为 平方公里 即将面积除以 1000000
def m2_to_km2(area):
       return area / 1000000

if __name__ == '__main__':
       # 读取CSV数据
       df = pd.read_csv(PATH)
       monthly_fire_frequency_boxplot2(df)
       yearly_fire_frequency_boxplot2(df)

       # 2. preprocess weather data
       # weather_preprocess(PATH_ALLTEMP)
       # weather_preprocess(PATH_RF)
       # weather_preprocess(PATH_RH)
       # weather_preprocess(PATH_WSPD)

       # df = pd.read_csv(PATH_ALLTEMP2)
       # yearly_weather_boxplot(df, 'Temperature (°C)')

       # df = pd.read_csv(PATH_RF2)
       # yearly_weather_boxplot(df, 'Rainfall (mm)')

       # df = pd.read_csv(PATH_RH2)
       # yearly_weather_boxplot(df, 'Relative Humidity (%)')

       # df = pd.read_csv(PATH_WSPD2)
       # yearly_weather_boxplot(df, 'Wind Speed (m/s)')

