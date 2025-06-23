import pandas as pd

import random
import time
import requests
import json
import os
import tqdm

DIR = os.path.dirname(os.path.abspath(__file__))

WDIR = os.path.join(DIR, '..', '..', 'data2', 'weather') 
PATH1 = os.path.join(WDIR, 'RH.json') # 相对湿度 低
PATH2 = os.path.join(WDIR, 'RF.json')  # 降雨量 低
PATH3 = os.path.join(WDIR, 'WSPD.json') # 风速 高
PATH4 = os.path.join(WDIR, 'ALLTEMP.json') # 每日均温 高


# DailyMeanTemperature_AllYear_url

SAVE_DIR = os.path.join(WDIR, 'output') 

SAVE_DIR1 = os.path.join(SAVE_DIR, 'RH') 
SAVE_DIR2 = os.path.join(SAVE_DIR, 'RF') 
SAVE_DIR3 = os.path.join(SAVE_DIR, 'WSPD') 
SAVE_DIR4 = os.path.join(SAVE_DIR, 'ALLTEMP')

## 1. ================================================
# def read_geojson_to_df(path):
#     with open(path, 'r', encoding='utf-8') as f:  # 明确指定使用utf-8编码
#         data = json.load(f)
#     df = pd.DataFrame(data['features'])
#     return df, data

def read_geojson_to_df(path):
    with open(path, 'r') as f:  # 明确指定使用utf-8编码
        data = json.load(f)
    df = pd.DataFrame(data['features'])
    return df, data


def update_geojson(path, df, data):
    data['features'] = df.to_dict(orient='records')
    with open(path, 'w') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def download_file(url, save_path):
    response = requests.get(url)
    # 查看是否存在 output 文件夹 若无则创建
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    with open(save_path, 'wb') as f:
        f.write(response.content)

def prepare_data(path=PATH1, save_dir=SAVE_DIR1, key='DailyMeanRelativeHumidity_AllYear_url'):
    df, data = read_geojson_to_df(path)
    # print(df.head())

    # 遍历所有文件 
    for i in tqdm.tqdm(range(len(df))):

        url = df['properties'][i][key]
        # print(url)
        # 下载文件
        save_path = os.path.join(save_dir, f"{i}.csv")
        # 随机休眠 1-3 秒
        time.sleep(random.randint(1, 3))
        download_file(url, save_path)

        df.loc[i]['properties']['save_path'] = os.path.relpath(save_path, save_dir)

    update_geojson(os.path.join(save_dir, 'metadata.json'), df, data) # save to metadata.json

## 2. ================================================
def filter_invalid(DATA_PATH,SAVE_PATH): 
    # 忽略最后五行
    # 读取数据时，跳过前两行 
    df = pd.read_csv(DATA_PATH, skiprows=3, skipfooter=5, engine='python')
    # print(df.head())
    # print(df.columns)

    # 仅仅保留 Completeness 为 C 的数据
    df = df[df['data Completeness/數據完整性/数据完整性'] == 'C']
    # print(df.head())

    # 仅仅保留 Year/年/年 为 2010 - 2024 的数据
    df = df[(df['Year/年/年'] >= 2010) & (df['Year/年/年'] <= 2024)]

    # 保存数据
    df.to_csv(SAVE_PATH, index=False)

def filter_data(path=SAVE_DIR1):
    df, _ = read_geojson_to_df(os.path.join(path, 'metadata.json'))
    for i in tqdm.tqdm(range(len(df))):
        # 读取数据
        save_path = os.path.join(path, df['properties'][i]['save_path'])
        filter_invalid(save_path, save_path)

# 简化数据
# Year/年/年,Month/月/月,Day/日/日,Hour/時/时,Minute/分/分,Second/秒/秒,TimeZone/時區/时区,Value/數值/数值,data Completeness/數據完整性/数据完整性
# 2010-01-01 将 Year/年/年,Month/月/月,Day/日/日 合并为 Date 然后只保留 Date,Value 两列
def simplify_data(DATA_PATH, SAVE_PATH):
    df = pd.read_csv(DATA_PATH)
    # 仅仅保留 Year/年/年,Month/月/月,Day/日/日,Value/數值/数值
    df = df[['Year/年/年', 'Month/月/月', 'Day/日/日', 'Value/數值/数值']]
    # 合并 Year/年/年,Month/月/月,Day/日/日 为 Date
    df['Date'] = df['Year/年/年'].astype(str) + '-' + df['Month/月/月'].astype(str) + '-' + df['Day/日/日'].astype(str)
    # 仅仅保留 Date, Value
    df = df[['Date', 'Value/數值/数值']]
    # 重命名列名 Value/數值/数值 -> Value
    df = df.rename(columns={'Value/數值/数值': 'Value'})
    df.to_csv(SAVE_PATH, index=False)

def simplify_all_data(path=SAVE_DIR1):
    df, _ = read_geojson_to_df(os.path.join(path, 'metadata.json'))
    for i in tqdm.tqdm(range(len(df))):
        # 读取数据
        save_path = os.path.join(path, df['properties'][i]['save_path'])
        simplify_data(save_path, save_path)
        # break

# 计算数据的平均值 并作为属性保存在 metadata.json 中
def calculate_mean(DATA_PATH):
    df = pd.read_csv(DATA_PATH)
    # 若 Value 中存在非数字的数据，将其转换为 NaN 然后去空
    df['Value'] = pd.to_numeric(df['Value'], errors='coerce')
    df = df.dropna(subset=['Value'])
    mean = df['Value'].mean()
    return mean

def calculate_all_mean(path=SAVE_DIR1):
    df, data = read_geojson_to_df(os.path.join(path, 'metadata.json'))
    for i in tqdm.tqdm(range(len(df))):
        # 读取数据
        save_path = os.path.join(path, df['properties'][i]['save_path'])
        mean = calculate_mean(save_path)
        # 保存到 metadata.json
        df.loc[i]['properties']['mean'] = mean
    update_geojson(os.path.join(path, 'metadata.json'), df, data)


def calculate_extreme_mean(path=SAVE_DIR1):
    """
    计算常规均值、极端月份均值及综合极端均值，并保存到metadata.json
    
    参数:
        path: 数据目录路径
    """
    # 读取元数据
    df, data = read_geojson_to_df(os.path.join(path, 'metadata.json'))
    
    # 获取所有站点的月度数据
    all_monthly_means = pd.DataFrame()
    for i in tqdm.tqdm(range(len(df))):
        save_path = os.path.join(path, df['properties'][i]['save_path'])
        try:
            station_data = pd.read_csv(save_path)
            
            # 数据预处理
            station_data['Date'] = pd.to_datetime(station_data['Date'])
            station_data['Month'] = station_data['Date'].dt.month
            station_data['Value'] = pd.to_numeric(station_data['Value'], errors='coerce')
            station_data = station_data.dropna(subset=['Value'])
            
            # 存储月度数据
            monthly_means = station_data.groupby('Month')['Value'].mean().reset_index()
            monthly_means['站点'] = i
            all_monthly_means = pd.concat([all_monthly_means, monthly_means])
        except Exception as e:
            print(f"处理站点 {i} 时出错: {str(e)}")
            continue
    
    # 计算并保存统计量
    for i in tqdm.tqdm(range(len(df))):
        # 获取当前站点的月度数据
        station_data = all_monthly_means[all_monthly_means['站点'] == i]
        
        if not station_data.empty:
            # 计算常规均值
            regular_mean = station_data['Value'].mean()
            
            # 计算极端月份均值
            top3_mean = station_data.nlargest(3, 'Value')['Value'].mean()
            bottom3_mean = station_data.nsmallest(3, 'Value')['Value'].mean()
            
            # 计算综合极端均值（top和bottom的加权平均）
            combined_extreme_mean = (top3_mean + bottom3_mean) / 2
            
            # 保存到properties中
            properties = df.loc[i, 'properties']
            properties['mean'] = regular_mean
            properties['extreme_months'] = {
                'top3_mean': top3_mean,
                'bottom3_mean': bottom3_mean,
                'combined_extreme_mean': combined_extreme_mean,  # 新增的综合指标
                'top_variation': (top3_mean - regular_mean) / regular_mean,
                'bottom_variation': (regular_mean - bottom3_mean) / regular_mean,
                'extreme_variation': (combined_extreme_mean - regular_mean) / regular_mean  # 综合差异率
            }
    
    # 更新元数据文件
    update_geojson(os.path.join(path, 'metadata.json'), df, data)
    print(f"已更新 {len(df)} 个站点的均值数据到 metadata.json")


# ================================================
def monthly_statistics(path=SAVE_DIR1, verbose=True):
    """
    计算并打印每个月的平均数值分布，并识别极端月份
    
    参数:
        path: 数据目录路径
        verbose: 是否打印详细统计信息
        
    返回:
        dict: 包含所有月份统计数据的字典
        pd.DataFrame: 所有站点的月度平均值汇总
    """
    # 读取元数据
    df, _ = read_geojson_to_df(os.path.join(path, 'metadata.json'))
    
    # 初始化结果存储
    all_stats = {}
    all_monthly_means = pd.DataFrame()

    # print(f"\n数据目录: {path}")
    # 由 \ 分割 取最后一个
    print(f"数据目录: {path.split('/')[-1]}")
    
    # 遍历所有站点数据
    for i in range(len(df)):
        save_path = os.path.join(path, df['properties'][i]['save_path'])
        try:
            data = pd.read_csv(save_path)
            
            # 数据预处理
            data['Date'] = pd.to_datetime(data['Date'])
            data['Month'] = data['Date'].dt.month
            data['Value'] = pd.to_numeric(data['Value'], errors='coerce')
            data = data.dropna(subset=['Value'])
            
            # 计算月度统计
            monthly_stats = data.groupby('Month')['Value'].agg(['mean', 'count'])
            monthly_stats.columns = ['平均值', '样本数']
            
            # 存储结果
            all_stats[i] = monthly_stats
            monthly_stats['站点'] = i
            all_monthly_means = pd.concat([all_monthly_means, monthly_stats.reset_index()])
            
            # # 打印站点月度信息
            if verbose:
                print(f"\n站点 {i} 月度平均值:")
                print("="*40)
                print(monthly_stats['平均值'].to_markdown(
                    headers=["月份", "平均值"],
                    floatfmt=".2f",
                    tablefmt="grid"
                ))
                
        except Exception as e:
            print(f"处理站点 {i} 时出错: {str(e)}")
            continue
    
    # 分析所有站点的极端月份
    if not all_monthly_means.empty:
        # 计算各月所有站点的平均值的统计量
        global_month_stats = all_monthly_means.groupby('Month')['平均值'].agg(
            ['mean', 'std', 'count']
        )
        global_month_stats['变异系数'] = global_month_stats['std'] / global_month_stats['mean']
        
        # 识别极端月份 (平均值最高和最低的3个月)
        top_months = global_month_stats['mean'].nlargest(3)
        bottom_months = global_month_stats['mean'].nsmallest(3)
        
        # 打印全局分析结果
        if verbose:
            print("\n" + "="*60)
            print("全局月度分析结果:")
            print("="*60)
            print("\n所有站点的月度平均值统计:")
            print(global_month_stats.to_markdown(floatfmt=".2f"))
        
        print("\n极端月份分析:")
        print(f"平均值最高的3个月份:\n{top_months.to_markdown(floatfmt='.2f')}")
        print(f"\n平均值最低的3个月份:\n{bottom_months.to_markdown(floatfmt='.2f')}")
        
        # 变异系数高的月份 (数据波动大)
        high_var_months = global_month_stats['变异系数'].nlargest(3)
        print(f"\n数据波动最大的3个月份(变异系数):\n{high_var_months.to_markdown(floatfmt='.2f')}")
    
    return all_stats, all_monthly_means

def calculate_extreme_months(path=SAVE_DIR1, verbose=True):
    """
    计算极端月份的均值并与常规均值并列展示
    
    参数:
        path: 数据目录路径
        verbose: 是否打印详细统计信息
        
    返回:
        pd.DataFrame: 包含常规均值和极端月份均值的数据框
    """
    # 先获取月度统计数据
    _, all_monthly_means = monthly_statistics(path, verbose=False)
    
    if all_monthly_means.empty:
        return pd.DataFrame()
    
    # 计算各站点的常规均值
    regular_means = all_monthly_means.groupby('站点')['平均值'].mean().rename('常规均值')
    
    # 计算极端月份均值 (最高和最低的3个月)
    extreme_stats = []
    for station in all_monthly_means['站点'].unique():
        station_data = all_monthly_means[all_monthly_means['站点'] == station]
        
        # 最高3个月均值
        top3_mean = station_data.nlargest(3, '平均值')['平均值'].mean()
        
        # 最低3个月均值
        bottom3_mean = station_data.nsmallest(3, '平均值')['平均值'].mean()
        
        extreme_stats.append({
            '站点': station,
            '高温月均值(前3)': top3_mean,
            '低温月均值(后3)': bottom3_mean
        })
    
    # 合并结果
    extreme_df = pd.DataFrame(extreme_stats)
    result_df = pd.merge(
        regular_means.reset_index(),
        extreme_df,
        on='站点'
    )
    
    # 计算差异比率
    result_df['高温差异率'] = (result_df['高温月均值(前3)'] - result_df['常规均值']) / result_df['常规均值']
    result_df['低温差异率'] = (result_df['常规均值'] - result_df['低温月均值(后3)']) / result_df['常规均值']
    
    # 打印结果
    if verbose:
        print(f"\n{'='*60}")
        print(f"{path.split('/')[-1]} 极端月份分析")
        print(f"{'='*60}")
        print(result_df.to_markdown(
            floatfmt=".2f",
            tablefmt="grid",
            headers=["站点", "常规均值", "高温月均值(前3)", "低温月均值(后3)", "高温差异率", "低温差异率"]
        ))
        
        # 打印全局统计
        print("\n全局统计:")
        global_stats = result_df[['高温月均值(前3)', '低温月均值(后3)']].agg(['mean', 'std'])
        print(global_stats.to_markdown(floatfmt=".2f"))
    
    return result_df

def convert_to_utf8(value):
    """将值转换为UTF-8编码的字符串"""
    if isinstance(value, str):
        try:
            # 尝试解码为UTF-8
            return value.encode('utf-8').decode('utf-8')
        except UnicodeError:
            # 尝试其他常见编码
            for encoding in ['gbk', 'big5', 'latin1', 'cp1252']:
                try:
                    return value.encode(encoding).decode('utf-8')
                except UnicodeError:
                    continue
            # 如果所有编码都失败，返回空字符串
            return ""
    elif isinstance(value, (list, dict)):
        # 递归处理嵌套结构
        if isinstance(value, list):
            return [convert_to_utf8(item) for item in value]
        else:
            return {k: convert_to_utf8(v) for k, v in value.items()}
    else:
        return value

def post_process_metadata(path):
    """后处理metadata.json文件，确保所有字符串为UTF-8编码"""
    df, data = read_geojson_to_df(os.path.join(path, 'metadata.json'))
    
    for i in tqdm.tqdm(range(len(df)), desc="处理站点"):
        properties = df.at[i, 'properties']
        new_properties = {}
        
        for key, value in properties.items():
            try:
                # 转换值为UTF-8编码
                converted_value = convert_to_utf8(value)
                new_properties[key] = converted_value
            except Exception as e:
                print(f"处理字段 {key} 在站点 {i} 时出错: {str(e)}")
                continue
        
        # 更新属性
        df.at[i, 'properties'] = new_properties
    
    # 更新元数据文件
    update_geojson(os.path.join(path, 'metadata.json'), df, data)
    print(f"\n已成功处理 {len(df)} 个站点的元数据，确保所有字段为UTF-8编码")

# 后处理 RH 为每一个csv增加头 Date,Value
def post_process_csv_RH(path=SAVE_DIR1):
    df, data = read_geojson_to_df(os.path.join(path, 'metadata.json'))
    
    for i in tqdm.tqdm(range(len(df)), desc="处理站点"):
        save_path = os.path.join(path, df['properties'][i]['save_path'])
        # 读取数据
        data = pd.read_csv(save_path)
        # 添加头 Date,Value
        data.columns = ['Date', 'Value']
        # 保存数据
        data.to_csv(save_path, index=False)
        # 处理完毕
        print(f"处理完毕 {i}")
    

if __name__ == "__main__":
    # 1. 下载数据至于 data/weather/output 文件夹下，同时输出 metadata.json
    # prepare_data(PATH1, SAVE_DIR1, 'DailyMeanRelativeHumidity_AllYear_url')
    # prepare_data(PATH2, SAVE_DIR2, 'DailyTotalRainfall_AllYear_url')
    # prepare_data(PATH3, SAVE_DIR3, 'DailyMeanWindSpeed_AllYear_url')
    # prepare_data(PATH4, SAVE_DIR4, 'DailyMeanTemperature_AllYear_url')


    # 2. 过滤数据 仅仅保留 Completeness 为 C 的数据 2010 - 2020 的数据
    # filter_data(SAVE_DIR1)
    # filter_data(SAVE_DIR2)
    # filter_data(SAVE_DIR3)
    # filter_data(SAVE_DIR4)

    # 3. 简化数据
    # simplify_all_data()
    # simplify_all_data(SAVE_DIR2)
    # simplify_all_data(SAVE_DIR3)
    # simplify_all_data(SAVE_DIR4)

    # 4. 计算数据的平均值 并作为属性保存在 metadata.json 中
    # calculate_all_mean()
    # calculate_all_mean(SAVE_DIR2)
    # calculate_all_mean(SAVE_DIR3)
    # calculate_all_mean(SAVE_DIR4)

    # 5. 计算每个月的数值分布统计信息 可选
    # monthly_statistics(SAVE_DIR1, verbose=False)
    # monthly_statistics(SAVE_DIR2, verbose=False)
    # monthly_statistics(SAVE_DIR3, verbose=False)
    # monthly_statistics(SAVE_DIR4, verbose=False)

    # 6. 计算极端月份的均值并与常规均值并列展示
    # calculate_extreme_mean(SAVE_DIR1)
    # calculate_extreme_mean(SAVE_DIR2)
    # calculate_extreme_mean(SAVE_DIR3)
    # calculate_extreme_mean(SAVE_DIR4)

    # 7. 针对metadata.json 的后处理，将所有不满足 utf-8 的字段删除
    # post_process_metadata(SAVE_DIR1)
    # post_process_metadata(SAVE_DIR2)
    # post_process_metadata(SAVE_DIR3)
    # post_process_metadata(SAVE_DIR4)

    # 8. 针对每一个csv文件的后处理，增加头 Date,Value
    post_process_csv_RH(SAVE_DIR1)