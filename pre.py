import rasterio
import numpy as np
import os
import cv2
import sys

# --- 0. 配置与路径 (请确保路径正确) ---

T1_DAT_PATH = '第一时相文件'
T2_DAT_PATH = '第二时相文件'

OUTPUT_ROOT = '结果文件夹'
OUTPUT_IM1_DIR = os.path.join(OUTPUT_ROOT, 'im1')
OUTPUT_IM2_DIR = os.path.join(OUTPUT_ROOT, 'im2')

TILE_SIZE = 512

# 关键配置 1: 波段映射
BAND_MAP_T1 = [4, 3, 2]#根据实际情况
BAND_MAP_T2 = [4, 3, 2]

# 创建输出目录
os.makedirs(OUTPUT_IM1_DIR, exist_ok=True)
os.makedirs(OUTPUT_IM2_DIR, exist_ok=True)


# --- 辅助函数：计算影像的实际最大值 ---
def calculate_actual_max_value(input_path, band_map):
    if not os.path.exists(input_path):
        return 0.0, 'N/A'

    try:
        with rasterio.open(input_path) as src:
            # 读取所有感兴趣的波段数据
            # 注意：如果影像非常大，这可能需要时间。
            image_data = src.read(band_map)

            # 计算所有波段中的全局最大值
            actual_max = np.max(image_data)
            data_type = str(src.dtypes[0])

            return float(actual_max), data_type

    except Exception as e:
        print(f"❌ 错误：计算 {os.path.basename(input_path)} 最大值失败。详细错误: {e}")
        return 0.0, 'N/A'


# --- 1. 转换和分块核心函数 (更新了 max_value 的来源) ---

# 函数签名现在接受 max_value 作为归一化分母
def process_and_tile_image(input_path, output_dir, band_map, max_value, tile_size):
    if not os.path.exists(input_path):
        print(f"❌ 错误：输入文件不存在: {input_path}")
        return

    # 检查全局最大值是否有效
    if max_value <= 0:
        print("❌ 错误：归一化分母为零或负数，无法处理。")
        return

    print(f"\n--- 正在处理文件: {os.path.basename(input_path)} ---")
    print(f"    归一化分母 (全局最大值): {max_value}")

    try:
        with rasterio.open(input_path) as src:
            height, width = src.height, src.width
            num_rows = height // tile_size
            num_cols = width // tile_size

            print(f"原图尺寸: {height}x{width}，将生成 {num_rows * num_cols} 个 {tile_size}x{tile_size} 块。")

            for i in range(num_rows):
                for j in range(num_cols):
                    window = rasterio.windows.Window(j * tile_size, i * tile_size, tile_size, tile_size)
                    tile_data = src.read(band_map, window=window)
                    tile_data = np.transpose(tile_data, (1, 2, 0))

                    # 归一化：使用全局最大值作为分母
                    normalized_data = tile_data / max_value

                    # 确保数值在 0.0 到 1.0 之间 (防止计算误差导致溢出)
                    normalized_data = np.clip(normalized_data, 0.0, 1.0)

                    # 放大到 0-255 并转换为 8 位整数
                    tile_data_norm_8bit = (normalized_data * 255).astype(np.uint8)

                    filename = os.path.join(output_dir, f'tile_{i:04d}_{j:04d}.png')
                    cv2.imwrite(filename, cv2.cvtColor(tile_data_norm_8bit, cv2.COLOR_RGB2BGR))

            print(f"✅ 文件 {os.path.basename(input_path)} 分块完成，保存到 {output_dir}")

    except Exception as e:
        print(f"❌ 处理文件时发生错误: {e}")
        print(f"请检查文件路径、或 BAND_MAP ({band_map}) 配置是否正确。详细错误: {e}")


# --- 2. 主执行逻辑 (整合全局最大值计算和归一化) ---
if __name__ == '__main__':
    print("======== 步骤 1: 检查并计算全局最大值 ========")

    # 计算 T1 实际最大值
    max_t1, type_t1 = calculate_actual_max_value(T1_DAT_PATH, BAND_MAP_T1)
    print(f"T1 (2017) 实际最大值: {max_t1} ({type_t1})")

    # 计算 T2 实际最大值
    max_t2, type_t2 = calculate_actual_max_value(T2_DAT_PATH, BAND_MAP_T2)
    print(f"T2 (2021) 实际最大值: {max_t2} ({type_t2})")

    # 确定全局最大值，用于统一归一化
    GLOBAL_MAX_VALUE = max(max_t1, max_t2)

    if GLOBAL_MAX_VALUE <= 0:
        print("❌ 致命错误：无法确定有效的全局最大值。请检查影像文件是否存在或是否包含有效数据。")
        sys.exit(1)

    print(f"\n🎉 确定全局最大值 (归一化分母): {GLOBAL_MAX_VALUE}")

    print("\n======== 步骤 2: 开始分块和转换 ========")

    # 处理 T1 图像 (传入 GLOBAL_MAX_VALUE)
    process_and_tile_image(T1_DAT_PATH, OUTPUT_IM1_DIR, BAND_MAP_T1, GLOBAL_MAX_VALUE, TILE_SIZE)

    # 处理 T2 图像 (传入 GLOBAL_MAX_VALUE)
    process_and_tile_image(T2_DAT_PATH, OUTPUT_IM2_DIR, BAND_MAP_T2, GLOBAL_MAX_VALUE, TILE_SIZE)

    print("\n--- 分块完成 ---")

    print("现在您可以运行之前提供的『本地 CPU 预测脚本』了。")
