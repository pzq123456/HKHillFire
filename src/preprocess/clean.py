# wild fire 
import pandas as pd
import os

DIR = os.path.dirname(os.path.abspath(__file__))

PATH = os.path.join(DIR, '..', '..', 'data2', 'source', 'Address, Area and TOC of Vegetation Fire in 2021 & 2022.xls') 

SAVE_PATH = os.path.join(DIR, '..', '..', 'data2', 'files')

def excel_to_csv(excel_file, output_directory):
    # 输入 Excel 文件路径
    excel_file = excel_file

    # 输出 CSV 文件的保存目录
    output_directory = output_directory

    # 读取 Excel 文件中的所有 sheet 名称
    excel_data = pd.ExcelFile(excel_file)

    # Install xlrd >= 2.0.1 for xls Excel support Use pip or conda to install xlrd.
    # pip install xlrd>=2.0.1

    # 遍历每个 sheet，将其保存为单独的 CSV 文件
    for sheet_name in excel_data.sheet_names:
        # 读取当前 sheet 数据
        df = excel_data.parse(sheet_name)
        
        # 创建 CSV 文件路径，文件名根据 sheet 名
        csv_file = os.path.join(output_directory, f"{sheet_name}.csv")
        
        # 保存为 CSV 文件
        df.to_csv(csv_file, index=False, encoding="utf-8-sig")
        
        print(f"Saved {sheet_name} to {csv_file}")

    print("All sheets have been exported to CSV files.")

# 扫描目标文件夹，获取csv文件列表
def get_csv_files(directory):
    csv_files = []
    for file in os.listdir(directory):
        if file.endswith(".csv"):
            csv_files.append(file)
    return csv_files

# 指定文件夹下所有csv文件合成一个大的csv文件 all.csv
def merge_csv_files(directory, output_file):
    csv_files = get_csv_files(directory)
    all_data = pd.DataFrame()

    for file in csv_files:
        file_path = os.path.join(directory, file)
        data = pd.read_csv(file_path, encoding="utf-8-sig")
        all_data = pd.concat([all_data, data], ignore_index=True)

    all_data.to_csv(output_file, index=False, encoding="utf-8-sig")
    print(f"All CSV files have been merged into {output_file}")

def postprocess_csv_2022():
    # 删除 Unnamed: 5 列
    df = pd.read_csv(os.path.join(SAVE_PATH, '2022.csv'), encoding="utf-8-sig")
    df.drop(columns=['Unnamed: 5'], inplace=True)
    # save to csv
    df.to_csv(os.path.join(SAVE_PATH, '2022.csv'), index=False, encoding="utf-8-sig")


# 2023, 2024 年的日期格式为 13/1/2024 即 DD/MM/YYYY 需要统一为 YYYY-MM-DD
def postprocess_csv_2023_2024():
    for year in [2023, 2024]:
        df = pd.read_csv(os.path.join(SAVE_PATH, f'{year}.csv'), encoding="utf-8-sig")
        # 将日期格式转换为 YYYY-MM-DD
        df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y').dt.strftime('%Y-%m-%d')
        # save to csv
        df.to_csv(os.path.join(SAVE_PATH, f'{year}.csv'), index=False, encoding="utf-8-sig")

# 重命名 2023 字段 Involved Area
# Involved Area -> Area (M2)
def rename_columns_2023():
    df = pd.read_csv(os.path.join(SAVE_PATH, '2023.csv'), encoding="utf-8-sig")
    df.rename(columns={'Involved Area': 'Area (M2)'}, inplace=True)
    # save to csv
    df.to_csv(os.path.join(SAVE_PATH, '2023.csv'), index=False, encoding="utf-8-sig")

if __name__ == '__main__':
    # excel_to_csv(PATH, SAVE_PATH)
    # postprocess_csv_2022()
    merge_csv_files(SAVE_PATH, os.path.join(SAVE_PATH, 'all.csv'))
    # postprocess_csv_2023_2024()
    # rename_columns_2023()