import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
from osgeo import gdal
import os
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle
import matplotlib as mpl

# 设置字体
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.unicode_minus'] = False

# 设置路径
DIR = os.path.dirname(os.path.abspath(__file__))
WDIR = os.path.join(DIR, '..', '..', 'data2', 'files')
WDIR2 = os.path.join(DIR, '..', '..', 'data2', 'co')

# 栅格路径
TIFF_PATHS = {
    "fire_class": os.path.join(WDIR, "fire_classification2.tiff"),
    "Vegetation_Fraction": os.path.join(WDIR2, "downLUM_end2020_2.tif"),
    "Relative_Humidity": os.path.join(WDIR2, "kriging_RH.tiff"),
    "Rainfall": os.path.join(WDIR2, "kriging_RF.tiff"),
    "Wind_Speed": os.path.join(WDIR2, "kriging_WSPD.tiff"),
    "Temperature": os.path.join(WDIR2, "kriging_ALLTEMP.tiff"),
    "WUI": os.path.join(WDIR2, "Baseline_2020_Intermix_WUI.tif"),
    "PPP": os.path.join(WDIR2, "ppp_2020_1km_Aggregated.tif")
}

# 定义火灾类型标签
FIRE_TYPES = {
    1: "RSD", 2: "RSW", 3: "RLD", 4: "RLW",
    5: "CSD", 6: "CSW", 7: "CLD", 8: "CLW"
}

# 变量分类
ENV_VARS = ["Vegetation_Fraction", "Relative_Humidity", "Rainfall", "Wind_Speed", "Temperature"]
HUMAN_VARS = ["WUI", "PPP"]
ALL_VARS = ENV_VARS + HUMAN_VARS

# 自定义颜色映射
def create_custom_cmap():
    colors = ["#3d72b4", "#f7f7f7", "#c44e52"]  # 更柔和的蓝-白-红
    return LinearSegmentedColormap.from_list("custom_cmap", colors, N=256)

# 读取栅格数据
def load_raster_data(path):
    dataset = gdal.Open(path)
    if dataset is None:
        raise ValueError(f"Failed to load raster: {path}")
    
    array = dataset.ReadAsArray().astype(float)
    nodata_value = dataset.GetRasterBand(1).GetNoDataValue()

    if nodata_value is not None:
        array[array == nodata_value] = np.nan

    dataset = None
    return array.flatten()

# 计算Spearman相关系数及其置信区间
def compute_spearman_with_ci(x, y, alpha=0.05):
    corr, p_value = stats.spearmanr(x, y, nan_policy='omit')
    n = len(x)
    
    if n > 3 and abs(corr) < 1.0:
        z = np.arctanh(corr)
        se = 1.0 / np.sqrt(n - 3)
        z_crit = stats.norm.ppf(1 - alpha/2)
        lo_z, hi_z = z - z_crit*se, z + z_crit*se
        lo, hi = np.tanh((lo_z, hi_z))
    else:
        lo, hi = np.nan, np.nan
    
    return corr, p_value, (lo, hi)

# 计算相关性
def compute_spearman_correlation(df):
    correlation_matrix = pd.DataFrame(index=ALL_VARS, columns=FIRE_TYPES.values(), dtype=float)
    p_value_matrix = pd.DataFrame(index=ALL_VARS, columns=FIRE_TYPES.values(), dtype=float)
    ci_matrix = pd.DataFrame(index=ALL_VARS, columns=FIRE_TYPES.values(), dtype=object)
    sample_size_matrix = pd.DataFrame(index=ALL_VARS, columns=FIRE_TYPES.values(), dtype=int)

    for var in ALL_VARS:
        for f_val, label in FIRE_TYPES.items():
            binary_indicator = (df['fire_class'] == f_val).astype(int)
            valid_mask = ~np.isnan(df[var]) & ~np.isnan(binary_indicator)
            n = valid_mask.sum()
            
            if n > 0:
                corr, p_value, ci = compute_spearman_with_ci(
                    binary_indicator[valid_mask], 
                    df[var][valid_mask]
                )
                correlation_matrix.loc[var, label] = corr
                p_value_matrix.loc[var, label] = p_value
                ci_matrix.loc[var, label] = ci
                sample_size_matrix.loc[var, label] = n
            else:
                correlation_matrix.loc[var, label] = np.nan
                p_value_matrix.loc[var, label] = np.nan
                ci_matrix.loc[var, label] = (np.nan, np.nan)
                sample_size_matrix.loc[var, label] = 0
    
    return correlation_matrix, p_value_matrix, ci_matrix, sample_size_matrix

# 生成标注文本
def format_annotation(corr, p_value, ci, fmt=".2f"):
    if np.isnan(corr):
        return ""
    
    text = f"{corr:{fmt}}"
    
    if p_value < 0.001:
        text += "***"
    elif p_value < 0.01:
        text += "**"
    elif p_value < 0.05:
        text += "*"
    
    return text

# 绘制热图
def plot_correlation_matrix(correlation_matrix, p_value_matrix, ci_matrix, sample_size_matrix):
    # 重新排列列 - 将有缺失值的列移到右侧
    na_counts = correlation_matrix.isna().sum()
    sorted_cols = na_counts.sort_values().index.tolist()
    corr_sorted = correlation_matrix[sorted_cols]
    pval_sorted = p_value_matrix[sorted_cols]
    ci_sorted = ci_matrix[sorted_cols]

    # 重新排列，将最下面两行移到最上面
    corr_sorted = pd.concat([corr_sorted.iloc[-2:], corr_sorted.iloc[:-2]])
    pval_sorted = pd.concat([pval_sorted.iloc[-2:], pval_sorted.iloc[:-2]])
    ci_sorted = pd.concat([ci_sorted.iloc[-2:], ci_sorted.iloc[:-2]])
    
    # 准备标注文本
    annot_matrix = pd.DataFrame(index=ALL_VARS, columns=sorted_cols, dtype=str)
    for var in ALL_VARS:
        for label in sorted_cols:
            corr = corr_sorted.loc[var, label]
            p_value = pval_sorted.loc[var, label]
            ci = ci_sorted.loc[var, label]
            annot_matrix.loc[var, label] = format_annotation(corr, p_value, ci)
    
    # 创建图形
    fig, ax = plt.subplots(figsize=(12, 8))
    fig.subplots_adjust(left=0.15, right=0.85, bottom=0.15, top=0.9)
    
    # 创建自定义颜色映射
    cmap = create_custom_cmap()
    
    # 绘制热图
    sns.heatmap(
        corr_sorted,
        annot=annot_matrix,
        cmap=cmap,
        center=0,
        fmt="",
        annot_kws={"size": 12, "va": "center", "color": "#333333"},
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"label": "Spearman Correlation", "shrink": 0.6, "pad": 0.02},
        ax=ax
    )
    
    # 添加斜线填充缺失值 - 简化版
    for i, var in enumerate(ALL_VARS):
        for j, col in enumerate(sorted_cols):
            if np.isnan(corr_sorted.loc[var, col]):
                ax.add_patch(Rectangle((j, i), 1, 1, fill=True, 
                            color='#f5f5f5', linewidth=0))
                ax.add_patch(Rectangle((j, i), 1, 1, fill=False, 
                            hatch='////', linewidth=0.5, edgecolor='#cccccc'))
    
    # # 添加变量类型分隔线
    # ax.axhline(len(ENV_VARS), color='gray', linestyle='--', linewidth=1, alpha=0.7)
    num_samples = sample_size_matrix.max().max()
    # 添加标题和标签
    fig.text(0.5, 0.02, f"Correlation Matrix with 95% Confidence Intervals (*p<0.05, **p<0.01, ***p<0.001) Sample Size: {num_samples}",
             ha='center', fontsize=14, fontweight='bold')
    
    ax.set_xlabel('Fire Type', fontsize=12, labelpad=12, fontweight='bold')
    ax.set_ylabel('Variable', fontsize=12, labelpad=12, fontweight='bold')
    
    # 调整坐标轴标签
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=12)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=12)
    
    # # 添加变量类型标签
    # ax.text(-0.5, len(ENV_VARS)/2, "Environmental\nVariables", 
    #         rotation=90, va='center', ha='center', fontsize=12)
    # ax.text(-0.5, len(ENV_VARS) + len(HUMAN_VARS)/2, "Human\nVariables", 
    #         rotation=90, va='center', ha='center', fontsize=12)
    
    # 添加样本量信息
    # min_samples = sample_size_matrix.min().min()
    # max_samples = sample_size_matrix.max().max()
    # fig.text(0.95, 0.03, f"Sample sizes: {min_samples}-{max_samples}",
    #          ha='right', fontsize=12, alpha=0.7)
    
    # 调整颜色条位置
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=12)
    cbar.set_label("Correlation", fontsize=14)
    
    plt.tight_layout()
    plt.show()

def main():
    # 加载所有栅格数据
    data = {key: load_raster_data(path) for key, path in TIFF_PATHS.items()}
    df = pd.DataFrame(data).dropna()

    # 计算相关性
    correlation_matrix, p_value_matrix, ci_matrix, sample_size_matrix = compute_spearman_correlation(df)

    # 可视化
    plot_correlation_matrix(correlation_matrix, p_value_matrix, ci_matrix, sample_size_matrix)

if __name__ == '__main__':
    main()