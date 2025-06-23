import pandas as pd
import os
import tqdm
import re
import requests

URL = "http://localhost:8080/search"

# URL = "https://nominatim.openstreetmap.org/search"


DIR = os.path.dirname(os.path.abspath(__file__))

PATH = os.path.join(DIR, '..', '..', 'data2', 'source', 'Address, Area and TOC of Vegetation Fire in 2021 & 2022.xls') 

SAVE_PATH = os.path.join(DIR, '..', '..', 'data2', 'files')

# 扫描目标文件夹，获取csv文件列表
def get_csv_files(directory):
    csv_files = []
    for file in os.listdir(directory):
        if file.endswith(".csv"):
            csv_files.append(file)
    return csv_files

def split_address(address):
    """将地址按空格进行分词，并返回词汇列表"""
    return address.split()

def sliding_window_combinations(address_parts):
    """生成地址部分的滑动窗口组合"""
    n = len(address_parts)
    for size in range(n, 0, -1):  # 从全地址到单词
        for start in range(n - size + 1):
            yield " ".join(address_parts[start:start + size])  # 每次返回一个滑动窗口组合

# 地理编码函数
def geocode(address):
    url = URL  # 替换为实际的 API URL
    params = {
        "q": address,
        "format": "json",
        "addressdetails": 1,
        "limit": 1  # 设置返回最多5个匹配结果
    }
    
    # 设置 User-Agent 来符合 Nominatim API 的要求
    headers = {
        "User-Agent": "myGeocoderApp/1.0 (myemail@example.com)"  # 替换为你自己的邮箱或应用标识
    }
    
    # 发送请求
    response = requests.get(url, params=params, headers=headers)
    
    # 检查返回状态码是否为200且有返回结果
    if response.status_code == 200:
        data = response.json()
        if data:
            result = data[0]
            lat = float(result["lat"])
            lon = float(result["lon"])
            return lat, lon
        else:
            return None, None
    else:
        return None, None

def match_address(address, parsed_address):
    # 对主地址进行分词
    address_parts = split_address(parsed_address['main_address'])
    
    # 定义地址匹配策略
    strategies = [
        # 第一个策略：使用完整的主地址
        lambda: geocode(parsed_address['main_address']),
        
        # 第二个策略：去掉主地址的第一个单词（去除前缀或无关信息）
        lambda: geocode(' '.join(parsed_address['main_address'].split(' ')[1:])),
        
        # 第三个策略：尝试描述字段，可能包含附加地理信息
        lambda: geocode(parsed_address['description']),
        
        # 第四个策略：去掉描述字段的第一个单词（减少干扰信息）
        lambda: geocode(' '.join(parsed_address['description'].split(' ')[1:])),
        
        # 第五个策略：使用附加信息（如房号、楼层等）
        lambda: geocode(parsed_address['additional_info']),
        
        # 第六个策略：对主地址进行分词并尝试多种组合（滑动窗口）
        lambda: geocode_from_combinations(address_parts),
    ]
    
    # 遍历策略，直到找到匹配
    for strategy in strategies:
        lat, lon = strategy()
        if lat and lon:
            # print(f"Matched address: {strategy.__name__} -> {lat}, {lon}")
            return lat, lon
    return None, None

def geocode_from_combinations(address_parts):
    """尝试对滑动窗口中的每个地址组合进行地理编码"""
    for combination in sliding_window_combinations(address_parts):
        lat, lon = geocode(combination)
        if lat and lon:
            return lat, lon
    return None, None

def process_csv_file(csv_file):
    # 读取 CSV 文件
    df = pd.read_csv(csv_file)

    valid_Count = 0  # 有效地址数量

    # 遍历每行数据
    for index, row in tqdm.tqdm(df.iterrows(), total=df.shape[0], desc=f"Processing {csv_file}"):
        address = row["Eng. Addr."]
        parsed_address = parse_address(address)
        
        lat, lon = match_address(address, parsed_address)

        # 如果成功找到经纬度
        if lat and lon:
            valid_Count += 1
        
        # 将经纬度添加到 DataFrame 中
        df.at[index, "Latitude"] = lat
        df.at[index, "Longitude"] = lon

    print(f"{valid_Count / df.shape[0] * 100:.2f}% valid addresses.")
    return df, valid_Count / df.shape[0] * 100

def parse_address(address):
    # match = re.match(r"(?:(=)?([A-Z]+)\s)?([^,;]+)(?:,([^;]+))?(?:;(.*))?", address)
    # 开头可能是 = 或 / 或 空
    match = re.match(r"(?:(=|\\)?([A-Z]+)\s)?([^,;]+)(?:,([^;]+))?(?:;(.*))?", address)

    if match:
        return {
            "prefix": match.group(2) or "",
            "main_address": match.group(3).strip(),
            "additional_info": match.group(4).strip() if match.group(4) else "",
            "description": match.group(5).strip() if match.group(5) else "",
        }
    return {}

if __name__ == '__main__':

    csv_files = get_csv_files(SAVE_PATH)
    # 仅仅提取 2021 -2024 年的 csv 文件
    csv_files = [file for file in csv_files if re.search(r'202[1-4]', file)]
    # print(csv_files)

    metadata = {}
    # print(csv_files)
    for file in tqdm.tqdm(csv_files, desc="Processing CSV files"):
        df, percentage = process_csv_file(os.path.join(SAVE_PATH, file))
        metadata[file] = percentage
        df.to_csv(os.path.join(SAVE_PATH, file), index=False, encoding="utf-8-sig")

    # save metadata
    with open(os.path.join(SAVE_PATH, 'metadata.txt'), 'w') as f:
        f.write(str(metadata))
    print(metadata)
    print("All CSV files have been processed.")
