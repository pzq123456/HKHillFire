# 湿度箱线图分析
import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# 设置字体为 arial
plt.rcParams['font.sans-serif'] = ['Arial']

DIR = os.path.dirname(os.path.abspath(__file__))
PATH = os.path.join(DIR, '..', '..', 'data2', 'files', 'all.csv')

FONTSIZE = 18
LABL_FONTSIZE = 14

# preprocess 去除特别离谱的湿度数据 100% 0%
def preprocess(df):
       # 去除湿度为0%和100%的数据
       df = df[(df['Humidity'] > 0) & (df['Humidity'] < 100)]
       return df

def monthly_humidity_boxplot(df):
       # 预处理数据
       df = preprocess(df)
       # 将日期列转换为日期格式
       df['Date'] = pd.to_datetime(df['Date'])
       
       # 提取月份和月份名称
       df['Month'] = df['Date'].dt.month
       df['Month_Name'] = df['Date'].dt.strftime('%b')
       
       # 按月份排序
       month_order = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                     'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
       df['Month_Name'] = pd.Categorical(df['Month_Name'], categories=month_order, ordered=True)
       df = df.sort_values('Month')
       
       # 创建图形
       plt.figure(figsize=(14, 8))
       
       # 使用seaborn绘制箱线图
       boxplot = sns.boxplot(x='Month_Name', y='Humidity', data=df, 
                            palette='Blues', 
                            width=0.6, linewidth=1.5)
       
       # 设置图表标签和样式
       plt.xlabel('Month', fontsize=FONTSIZE)
       plt.ylabel('Relative Humidity (%)', fontsize=FONTSIZE)
       plt.ylim(0, 100)  # 湿度范围0-100%
       
       plt.xticks(fontsize=LABL_FONTSIZE)
       plt.yticks(fontsize=LABL_FONTSIZE)
       
       # 添加网格线
       plt.grid(True, linestyle='--', alpha=0.6, axis='y')
       
       plt.tight_layout()
       plt.show()

if __name__ == '__main__':
    # 读取CSV数据
    df = pd.read_csv(PATH)
    monthly_humidity_boxplot(df)