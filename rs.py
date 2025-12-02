import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
import os
import cv2
import glob
import matplotlib.pyplot as plt
import re
from tqdm import tqdm  # 引入进度条库
from matplotlib.colors import ListedColormap, BoundaryNorm  # 显式导入

# --- Matplotlib 中文配置 ---
# ⚠️ 确保您的系统中安装了 SimHei 字体或其他中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
# ---------------------------

# --- 0. 全局配置与本地路径 (‼️ 请修改为您的本地路径) ---
IMG_HEIGHT = 512
IMG_WIDTH = 512
NUM_BANDS = 3
INPUT_SHAPE = (IMG_HEIGHT, IMG_WIDTH, NUM_BANDS)
NUM_CHANGE_CLASSES = 4
BATCH_SIZE = 16  # 性能优化：每次送入模型的切片数量

# 🚨 替换为您本地 SECOND_data 文件夹的绝对路径或相对路径
SECOND_DATA_ROOT = './hnresults'
# 🚨 替换为您本地保存的权重文件路径
CD_WEIGHTS_PATH = 'best_cd_finetune_weights.h5'

IM1_DIR = os.path.join(SECOND_DATA_ROOT, 'im1')
IM2_DIR = os.path.join(SECOND_DATA_ROOT, 'im2')

# 确保 TensorFlow 使用 CPU
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
print("✅ TensorFlow 已配置为使用 CPU。")


# --- 1. 模型结构重新定义 (保持不变) ---

def conv_block(input_tensor, num_filters, kernel_size=(3, 3), name_suffix=''):
    x = layers.Conv2D(num_filters, kernel_size, activation='relu', padding='same',
                      name=f'conv_{num_filters}_a{name_suffix}')(input_tensor)
    x = layers.BatchNormalization(name=f'bn_{num_filters}_a{name_suffix}')(x)
    x = layers.Conv2D(num_filters, kernel_size, activation='relu', padding='same',
                      name=f'conv_{num_filters}_b{name_suffix}')(x)
    x = layers.BatchNormalization(name=f'bn_{num_filters}_b{name_suffix}')(x)
    return x


def encoder_block(input_tensor, num_filters, name_prefix):
    x = conv_block(input_tensor, num_filters, name_suffix=f'_{name_prefix}')
    p = layers.MaxPooling2D((2, 2), name=f'pool_{num_filters}_{name_prefix}')(x)
    return x, p


def decoder_block(input_tensor, skip_tensor, num_filters):
    x = layers.Conv2DTranspose(num_filters, (2, 2), strides=(2, 2), padding='same')(input_tensor)
    x = layers.concatenate([x, skip_tensor])
    x = conv_block(x, num_filters)
    return x


def build_pseudo_siamese_unet(input_shape, num_change_classes):
    input_t1 = layers.Input(input_shape, name='input_t1')
    input_t2 = layers.Input(input_shape, name='input_t2')

    c1_t1, p1_t1 = encoder_block(input_t1, 32, 't1')
    c2_t1, p2_t1 = encoder_block(p1_t1, 64, 't1')
    c3_t1, p3_t1 = encoder_block(p2_t1, 128, 't1')
    c4_t1, p4_t1 = encoder_block(p3_t1, 256, 't1')

    c1_t2, p1_t2 = encoder_block(input_t2, 32, 't2')
    c2_t2, p2_t2 = encoder_block(p1_t2, 64, 't2')
    c3_t2, p3_t2 = encoder_block(p2_t2, 128, 't2')
    c4_t2, p4_t2 = encoder_block(p3_t2, 256, 't2')

    b_t1 = conv_block(p4_t1, 512, name_suffix='_bottleneck_t1')
    b_t2 = conv_block(p4_t2, 512, name_suffix='_bottleneck_t2')

    bottleneck_diff = layers.Subtract(name='bottleneck_diff')([b_t1, b_t2])

    diff_c4 = layers.Subtract(name='skip_diff_c4')([c4_t1, c4_t2])
    diff_c3 = layers.Subtract(name='skip_diff_c3')([c3_t1, c3_t2])
    diff_c2 = layers.Subtract(name='skip_diff_c2')([c2_t1, c2_t2])
    diff_c1 = layers.Subtract(name='skip_diff_c1')([c1_t1, c1_t2])

    u4 = decoder_block(bottleneck_diff, diff_c4, 256)
    u3 = decoder_block(u4, diff_c3, 128)
    u2 = decoder_block(u3, diff_c2, 64)
    u1 = decoder_block(u2, diff_c1, 32)

    outputs = layers.Conv2D(num_change_classes, (1, 1), activation='softmax', name='change_output')(u1)
    model = Model(inputs=[input_t1, input_t2], outputs=outputs, name='Pseudo_Siamese_CD')
    return model


# --- 2. 加载模型和权重 (保持不变) ---

def load_best_model():
    model = build_pseudo_siamese_unet(INPUT_SHAPE, NUM_CHANGE_CLASSES)

    if os.path.exists(CD_WEIGHTS_PATH):
        try:
            model.load_weights(CD_WEIGHTS_PATH)
            print(f"✅ 成功加载最佳变化检测权重: {CD_WEIGHTS_PATH}")
            return model
        except Exception as e:
            print(f"❌ 错误：加载权重失败，请检查文件是否损坏或路径是否正确: {e}")
            return None
    else:
        print(f"❌ 错误：未找到权重文件，请检查 CD_WEIGHTS_PATH 是否正确: {CD_WEIGHTS_PATH}")
        return None


# --- 3. 辅助函数：加载和预处理切片 (保持不变) ---

def load_and_preprocess_tile(im1_path, im2_path):
    # 读取 T1/T2 图像并归一化 (BGR -> RGB, 0-255 -> 0.0-1.0)
    im1 = cv2.cvtColor(cv2.imread(im1_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    im2 = cv2.cvtColor(cv2.imread(im2_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

    return im1, im2


# --- 4. 预测和拼接完整影像函数 (修复接缝问题) ---

# 定义裁剪的像素数。我们将从预测结果的每个边界裁剪掉 CROP 像素。
CROP_PIXELS = 1
TILE_EFFECTIVE_SIZE = IMG_HEIGHT - 2 * CROP_PIXELS  # 512 - 2*1 = 510


def predict_and_stitch_full_image(model):
    all_files_t1 = glob.glob(os.path.join(IM1_DIR, 'tile_*.png'))

    if not all_files_t1:
        print(f"❌ 错误：在 {IM1_DIR} 中未找到任何 .png 图像切片。")
        return

    # 1. 确定原始影像的尺寸 (行数和列数) 和文件映射
    max_row, max_col = 0, 0
    file_metadata = []

    for path_t1 in all_files_t1:
        file_id = os.path.basename(path_t1)
        match = re.search(r'tile_(\d+)_(\d+)\.png', file_id)
        if match:
            row_idx = int(match.group(1))
            col_idx = int(match.group(2))

            max_row = max(max_row, row_idx)
            max_col = max(max_col, col_idx)

            file_metadata.append({'r': row_idx, 'c': col_idx, 'id': file_id})

    total_tiles = len(file_metadata)

    # 根据裁剪后的有效尺寸计算最终图像大小
    new_total_rows = (max_row + 1) * TILE_EFFECTIVE_SIZE
    new_total_cols = (max_col + 1) * TILE_EFFECTIVE_SIZE

    # 初始化完整的预测结果数组
    full_prediction_map = np.zeros((new_total_rows, new_total_cols), dtype=np.int8)

    print(f"✅ 找到 {total_tiles} 个切片。最终预测图尺寸将调整为 {new_total_rows}x{new_total_cols}。")
    print(f"🚀 将使用 BATCH_SIZE={BATCH_SIZE} 进行预测，并裁剪 {CROP_PIXELS} 像素边缘以消除接缝...")

    # 2. 批处理预测和填充

    file_metadata.sort(key=lambda x: (x['r'], x['c']))

    current_idx = 0

    with tqdm(total=total_tiles, desc="预测切片进度") as pbar:
        while current_idx < total_tiles:
            batch_end = min(current_idx + BATCH_SIZE, total_tiles)
            batch_metadata = file_metadata[current_idx:batch_end]
            batch_size_actual = len(batch_metadata)

            batch_im1 = np.zeros((batch_size_actual, IMG_HEIGHT, IMG_WIDTH, NUM_BANDS), dtype=np.float32)
            batch_im2 = np.zeros((batch_size_actual, IMG_HEIGHT, IMG_WIDTH, NUM_BANDS), dtype=np.float32)

            for i, meta in enumerate(batch_metadata):
                path_t1 = os.path.join(IM1_DIR, meta['id'])
                path_t2 = os.path.join(IM2_DIR, meta['id'])

                if not os.path.exists(path_t2): continue

                im1, im2 = load_and_preprocess_tile(path_t1, path_t2)

                batch_im1[i] = im1
                batch_im2[i] = im2

            prediction_raw = model.predict([batch_im1, batch_im2], verbose=0)

            # 将预测结果解包并填充到大图中
            for i, pred_tile_raw in enumerate(prediction_raw):
                meta = batch_metadata[i]

                pred_tile = np.argmax(pred_tile_raw, axis=-1).astype(np.int8)

                # --- 核心修改：裁剪预测结果以消除接缝 ---
                pred_tile_cropped = pred_tile[
                                    CROP_PIXELS: IMG_HEIGHT - CROP_PIXELS,
                                    CROP_PIXELS: IMG_WIDTH - CROP_PIXELS
                                    ]
                # ----------------------------------------

                # 计算在最终图中的起始和结束坐标，使用有效尺寸 (TILE_EFFECTIVE_SIZE)
                start_row = meta['r'] * TILE_EFFECTIVE_SIZE
                end_row = start_row + TILE_EFFECTIVE_SIZE
                start_col = meta['c'] * TILE_EFFECTIVE_SIZE
                end_col = start_col + TILE_EFFECTIVE_SIZE

                # 粘贴裁剪后的结果
                full_prediction_map[start_row:end_row, start_col:end_col] = pred_tile_cropped

            pbar.update(batch_size_actual)
            current_idx = batch_end

    # 3. 可视化完整的变化图

    # 定义颜色映射 (与之前相同)
    colors = ['lightgray', 'blue', 'red', '#FFFFF7']
    cmap = ListedColormap(colors)
    bounds = [-0.5, 0.5, 1.5, 2.5, 3.5]
    norm = BoundaryNorm(bounds, cmap.N)

    # 修复 Matplotlib 错误：使用 plt.subplots()
    fig, ax = plt.subplots(figsize=(10, 10))

    # 绘制图像到 Axes 对象上
    im = ax.imshow(full_prediction_map, cmap=cmap, norm=norm)

    ax.set_title("完整的变化监测预测结果(河南县)")
    ax.axis('off')

    # 修正：将 im 和 ax 传递给 colorbar
    cbar = fig.colorbar(im, ax=ax, ticks=[0, 1, 2, 3])

    cbar.set_ticklabels(['0: 未变化 (灰)', '1: 洪涝风险 (蓝)', '2: 草甸破坏 (红)', '3: 其他变化 (黄)'])

    # 确保输出目录存在
    if not os.path.exists(SECOND_DATA_ROOT):
        os.makedirs(SECOND_DATA_ROOT)

    output_filename = os.path.join(SECOND_DATA_ROOT, 'full_change_map_cropped.png')
    plt.savefig(output_filename, bbox_inches='tight', dpi=300)
    print(f"\n🎉 完整的变化监测图已保存到: {output_filename}")

    plt.show()


# --- 5. 主执行逻辑 ---

if __name__ == '__main__':
    loaded_model = load_best_model()
    if loaded_model:
        predict_and_stitch_full_image(loaded_model)