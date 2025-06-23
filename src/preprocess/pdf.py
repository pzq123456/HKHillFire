import camelot
import os
import tqdm
import pandas as pd

import re
from collections import defaultdict


DIR = os.path.dirname(os.path.abspath(__file__))

PATH1 = os.path.join(DIR, '..', '..', 'data2', 'source', 'Hill Fire Record 2023.pdf') 
PATH2 = os.path.join(DIR, '..', '..', 'data2', 'source', 'Hill Fire Record 2024.pdf') 

SAVE_PATH = os.path.join(DIR, '..', '..', 'data2', 'files')

def pdf_to_csv(input_pdf_path, output_csv_path):
    # 读取 PDF 文件中的表格
    tables = camelot.read_pdf(input_pdf_path, pages='all', flavor='stream')
    
    all_tables = []  # 用于存储所有表格的数据

    # 解析每个表格
    for i, table in tqdm.tqdm(enumerate(tables), desc="解析进度", total=len(tables)):
        print(f"解析报告（表格 {i+1}）：")
        print(table.parsing_report)
        # 打印表格的形状
        print(f"表格形状：{table.shape}")
        
        # 将每个表格的数据添加到 all_tables 列表中
        all_tables.append(table.df)

        # 保存每个表格为单独的 CSV 文件
        # table.to_csv(insert_index_to_filename(output_csv_path, i))
        print("保存单独的表格为 CSV 完成!")

    # 合并所有表格为一个大的 DataFrame
    combined_df = pd.concat(all_tables, ignore_index=True)

    # 输出合并后的表格到 CSV 文件
    combined_df.to_csv(output_csv_path, index=False)
    print(f"所有表格已合并并保存为：{output_csv_path}")

def postprocess_csv_2023():
    # 读取 CSV 文件
    df = pd.read_csv(os.path.join(SAVE_PATH, '2023.csv'))
    
    # 1. 删除第一行的编号
    df.columns = df.iloc[0]  # 将第一行设置为列名
    df = df[1:]  # 删除第一行
    
    # 2. 删除包含"Data for Vegetation Fire 2023"的行
    df = df[~df.iloc[:, 0].str.contains('Data for Vegetation Fire 2023', na=False)]

    # 删除包含“'No.', 'Month', 'Date', 'Time of Call', 'Incident Location English Address', 'Involved Area'”的行 仅保留第一行
    df = df[~df.iloc[:, 0].isin(['No.', 'Month', 'Date', 'Time of Call', 'Incident Location English Address', 'Involved Area'])]
    
    # 3. 处理地址信息被分成多行的情况
    # 重置索引以便于处理
    df.reset_index(drop=True, inplace=True)
    
    # 创建一个新列用于合并地址
    df['Combined Address'] = ''
    
    # 遍历每一行
    i = 0
    while i < len(df):
        # 检查当前行是否缺少主要信息但包含地址信息
        if pd.isna(df.at[i, 'No.']) and pd.isna(df.at[i, 'Month']) and pd.notna(df.at[i, 'Incident Location English Address']):
            # 找到上一个有效行（包含No.的行）
            j = i - 1
            while j >= 0 and pd.isna(df.at[j, 'No.']):
                j -= 1
            
            if j >= 0:
                # 合并地址信息
                if pd.isna(df.at[j, 'Combined Address']):
                    df.at[j, 'Combined Address'] = df.at[j, 'Incident Location English Address']
                
                # 添加当前行的地址信息
                df.at[j, 'Combined Address'] += ' ' + df.at[i, 'Incident Location English Address']
                
                # 标记当前行为待删除
                df.at[i, 'to_delete'] = True
            else:
                # 如果没有找到有效的前一行，保留当前行
                df.at[i, 'to_delete'] = False
        else:
            # 对于正常行，初始化Combined Address
            if pd.notna(df.at[i, 'Incident Location English Address']):
                df.at[i, 'Combined Address'] = df.at[i, 'Incident Location English Address']
            df.at[i, 'to_delete'] = False
        
        i += 1
    
    # 用合并后的地址替换原始地址
    df['Incident Location English Address'] = df['Combined Address']
    
    # 删除临时列
    df.drop(['Combined Address', 'to_delete'], axis=1, inplace=True)
    
    # 重置索引
    df.reset_index(drop=True, inplace=True)

    # 4. 处理 Involved Area 列 计算面积
    df['Involved Area'] = df['Involved Area'].apply(process_area)

    # No.,Month,Date,Time of Call,Incident Location English Address,Involved Area
    # No.,Date,Time,Eng. Addr.,Involved Area
    df.rename(columns={
        'Time of Call': 'Time',
        'Incident Location English Address': 'Eng. Addr.',
        'Involved Area': 'Involved Area'
    }, inplace=True)
    # 重新排列列的顺序
    df = df[['No.', 'Date', 'Time', 'Eng. Addr.', 'Involved Area']]
    df.dropna(inplace=True)

    df.to_csv(os.path.join(SAVE_PATH, '2023.csv'), index=False)
    print("CSV文件处理完成！")

def postprocess_csv_2024():
    # 读取 CSV 文件
    df = pd.read_csv(os.path.join(SAVE_PATH, '2024.csv'))
    
    # 1. 删除第一行的编号
    df.columns = df.iloc[0]  # 将第一行设置为列名
    df = df[1:]  # 删除第一行
    
    # 3. 处理地址信息被分成多行的情况
    # 重置索引以便于处理
    df.reset_index(drop=True, inplace=True)
    
    # 创建一个新列用于合并地址
    df['Combined Address'] = ''
    
    # 遍历每一行
    i = 0
    while i < len(df):
        # 检查当前行是否缺少主要信息但包含地址信息
        if pd.isna(df.at[i, 'No.']) and pd.isna(df.at[i, 'Month']) and pd.notna(df.at[i, 'Incident Location English Address']):
            # 找到上一个有效行（包含No.的行）
            j = i - 1
            while j >= 0 and pd.isna(df.at[j, 'No.']):
                j -= 1
            
            if j >= 0:
                # 合并地址信息
                if pd.isna(df.at[j, 'Combined Address']):
                    df.at[j, 'Combined Address'] = df.at[j, 'Incident Location English Address']
                
                # 添加当前行的地址信息
                df.at[j, 'Combined Address'] += ' ' + df.at[i, 'Incident Location English Address']
                
                # 标记当前行为待删除
                df.at[i, 'to_delete'] = True
            else:
                # 如果没有找到有效的前一行，保留当前行
                df.at[i, 'to_delete'] = False
        else:
            # 对于正常行，初始化Combined Address
            if pd.notna(df.at[i, 'Incident Location English Address']):
                df.at[i, 'Combined Address'] = df.at[i, 'Incident Location English Address']
            df.at[i, 'to_delete'] = False
        
        i += 1
    
    # 用合并后的地址替换原始地址
    df['Incident Location English Address'] = df['Combined Address']
    
    # 删除临时列
    df.drop(['Combined Address', 'to_delete'], axis=1, inplace=True)
    
    # 重置索引
    df.reset_index(drop=True, inplace=True)
    

    # 4. 处理 Involved Area 列 计算面积
    df['INVOLVED'] = df['INVOLVED'].apply(process_area)
    # 删除 Month 列
    df.drop(['Month'], axis=1, inplace=True)

    # No.,Date,Time,Eng. Addr.,Area (M2),
    # No.,Month,Date,Time of Call,Incident Location English Address,INVOLVED
    df.rename(columns={
        'Time of Call': 'Time',
        'Incident Location English Address': 'Eng. Addr.',
        'INVOLVED': 'Area (M2)'
    }, inplace=True)
    # 重新排列列的顺序
    df = df[['No.', 'Date', 'Time', 'Eng. Addr.', 'Area (M2)']]
    df.dropna(inplace=True)

    df.to_csv(os.path.join(SAVE_PATH, '2024.csv'), index=False)
    print("CSV文件处理完成！")

def analyze_involved_area_patterns(csv_path, Column='Involved Area'):
    # Load the processed CSV file
    df = pd.read_csv(csv_path)
    
    # Initialize a dictionary to store pattern counts
    pattern_counts = defaultdict(int)
    pattern_examples = {}
    
    # Analyze each entry in the Involved Area column
    for area in df['Involved Area']:
        if pd.isna(area):
            pattern = "Empty/NaN"
        else:
            # Standardize whitespace and case for comparison
            standardized = ' '.join(str(area).strip().upper().split())
            
            # Categorize patterns based on common structures
            if 'X' in standardized:
                parts = standardized.split('X')
                if len(parts) == 2 and 'M' in parts[1]:
                    pattern = "Size X SizeM (e.g., 1 X 1M)"
                else:
                    pattern = "Contains X but not standard format"
            elif 'M' in standardized and 'FIRELINE' in standardized:
                pattern = "LengthM FIRELINE (e.g., 15M FIRELINE)"
            elif 'M' in standardized:
                pattern = "Other M measurement"
            else:
                pattern = "Other format"
        
        # Count the pattern and store an example
        pattern_counts[pattern] += 1
        if pattern not in pattern_examples:
            pattern_examples[pattern] = standardized
    
    # Print the results
    print("Involved Area Pattern Analysis:")
    print("=" * 50)
    for pattern, count in pattern_counts.items():
        example = pattern_examples[pattern]
        print(f"Pattern: {pattern}")
        print(f"Count: {count}")
        print(f"Example: {example}")
        print("-" * 50)

# Function to process individual area entries
def process_area(area_str):
    if pd.isna(area_str):
        return None
    
    area_str = str(area_str).strip().upper()
    # pattern 0: "Size M X Size M" (e.g., "1M X 1M")
    if re.match(r'^\d+M\s*X\s*\d+M$', area_str):
        parts = re.split(r'\s*X\s*', area_str.replace('M', ''))
        try:
            length = float(parts[0])
            width = float(parts[1])
            return length * width
        except:
            return None
    
    # Pattern 1: Standard "Size X SizeM" (e.g., "1 X 1M")
    if re.match(r'^\d+\s*X\s*\d+M$', area_str):
        parts = re.split(r'\s*X\s*', area_str.replace('M', ''))
        try:
            length = float(parts[0])
            width = float(parts[1])
            return length * width
        except:
            return None
    
    # Pattern 2: "LengthM FIRELINE" (e.g., "15M FIRELINE")
    elif 'FIRELINE' in area_str:
        num_match = re.search(r'(\d+)M', area_str)
        if num_match:
            try:
                length = float(num_match.group(1))
                # Treat fireline as 1m width (adjust if different)
                return length * 1
            except:
                return None
        return None
    
    # Pattern 3: Contains X but not standard format (e.g., "3 X 3M 1 X 1 M")
    elif 'X' in area_str and 'M' in area_str:
        # Try to sum all X M patterns we can find
        total_area = 0
        matches = re.finditer(r'(\d+)\s*X\s*(\d+)M', area_str)
        for match in matches:
            try:
                length = float(match.group(1))
                width = float(match.group(2))
                total_area += length * width
            except:
                pass
        return total_area if total_area > 0 else None
    
    # Pattern 4: Other M measurement (e.g., "3 FIRE LINE 10M")
    elif 'M' in area_str:
        num_match = re.search(r'(\d+)M', area_str)
        if num_match:
            try:
                # Assume this is a length measurement
                length = float(num_match.group(1))
                # Treat as 1m width (adjust if different)
                return length * 1
            except:
                return None
        return None
    
    # Unknown pattern
    return None

if __name__ == "__main__":
    # 1. 
    # # 处理第一个 PDF 文件
    # pdf_to_csv(PATH1, os.path.join(SAVE_PATH, '2023.csv'))
    # # 处理第二个 PDF 文件
    # pdf_to_csv(PATH2, os.path.join(SAVE_PATH, '2024.csv'))

    # 2. 
    # 处理 CSV 文件
    postprocess_csv_2023()
    postprocess_csv_2024()
    # 辅助： 检查 Involved Area 列的模式
    # analyze_involved_area_patterns(os.path.join(SAVE_PATH, '2023_processed.csv'))
    # analyze_involved_area_patterns(os.path.join(SAVE_PATH, '2024_processed.csv'),"INVOLVED")
