import pandas as pd
import os
from datetime import datetime

def count_fires_by_year_on_date(file_path, target_month=4, target_day=5):
    """
    统计每年特定日期的火灾发生次数
    
    参数:
        file_path: 数据文件路径
        target_month: 目标月份 (1-12)
        target_day: 目标日期 (1-31)
    
    返回:
        包含年份和火灾次数的DataFrame
    """
    # 读取数据
    df = pd.read_csv(file_path)
    
    # 转换日期列
    df['Date'] = pd.to_datetime(df['Date'])
    
    # 提取年份、月份和日
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Day'] = df['Date'].dt.day
    
    # 筛选目标日期的数据
    target_date_df = df[(df['Month'] == target_month) & (df['Day'] == target_day)]
    
    # 按年份统计
    result = target_date_df.groupby('Year').size().reset_index(name='FireCount')
    
    # 补全年份（确保没有火灾的年份显示为0）
    all_years = pd.DataFrame({'Year': range(df['Year'].min(), df['Year'].max()+1)})
    result = all_years.merge(result, on='Year', how='left').fillna(0)
    
    # 转换火灾次数为整数
    result['FireCount'] = result['FireCount'].astype(int)
    
    return result

# 使用示例
if __name__ == '__main__':
    # 文件路径
    DIR = os.path.dirname(os.path.abspath(__file__))
    PATH = os.path.join(DIR, '..', '..', 'data2', 'files', 'all.csv')
    
    # 统计每年4月5日的火灾次数
    april_5_fires = count_fires_by_year_on_date(PATH, target_month=4, target_day=5)
    
    # 打印结果
    print("每年4月5日火灾统计:")
    print(april_5_fires.to_string(index=False))
    
    # 也可以轻松统计其他日期，例如：
    # december_25_fires = count_fires_by_year_on_date(PATH, target_month=12, target_day=25)

#  Year  FireCount
#  2010          8
#  2011         37
#  2012          4
#  2013          0
#  2014         38
#  2015        114
#  2016          0
#  2017          5
#  2018         78
#  2019         26
#  2020          0
#  2021          5
#  2022         35
#  2023          1
#  2024          3