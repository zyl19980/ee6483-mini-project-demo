import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import pandas as pd
import os

# --- 1. 定义常量 (与训练时保持一致) ---
IMG_SIZE = (128, 128)
BATCH_SIZE = 32  # 批量大小可以调大，预测时不需要太小
TEST_DIR = 'datasets/tests' # 指向包装文件夹

# --- 2. 加载模型 ---
print("Loading saved model...")
model = load_model('best_dog_vs_cat_model.keras')
print("Model loaded.")

# --- 3. 准备测试数据生成器 ---
# 测试集只需要归一化
test_datagen = ImageDataGenerator(rescale=1./255)

print("Preparing test data generator...")
test_generator = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode=None,      # 测试集没有标签
    shuffle=False         # !!! 关键：绝对不能打乱顺序 !!!
)

# --- 4. 执行预测 ---
print("Generating predictions on test set...")
predictions = model.predict(test_generator)

# --- 5. 格式化并保存 submission.csv ---
# predictions 是 (0, 1) 之间的概率值
# 你的模型中 cat=0, dog=1。项目要求 0=cat, 1=dog 
# 所以 > 0.5 判定为 1 (dog)
predicted_labels = (predictions > 0.5).astype(int).flatten()

# 获取文件名作为 ID
# test_generator.filenames 会返回 'test/1.jpg', 'test/10.jpg' ...
# 我们需要提取文件名 (不含 .jpg) 作为 ID
try:
    ids = [os.path.splitext(os.path.basename(f))[0] for f in test_generator.filenames]
    
    # 创建 DataFrame
    submission_df = pd.DataFrame({
        'ID': ids,
        'label': predicted_labels
    })

    # 按照ID (文件名) 排序，确保顺序正确 (可选，但推荐)
    # 假设 ID 是 '1', '2', ... '500'
    submission_df['ID'] = submission_df['ID'].astype(int)
    submission_df = submission_df.sort_values(by='ID')

    # 保存为 CSV
    submission_df.to_csv('submission.csv', index=False)
    print("submission.csv file created successfully.")
    print(submission_df.head())

except ValueError:
    print("\nError: Could not convert filenames to integer IDs for sorting.")
    print("This may happen if filenames are not simple numbers (e.g., 'a_1.jpg').")
    print("Saving file without sorting...")
    
    # 如果ID不是纯数字，按文件名原始顺序保存
    ids_raw = [os.path.splitext(os.path.basename(f))[0] for f in test_generator.filenames]
    submission_df_raw = pd.DataFrame({
        'ID': ids_raw,
        'label': predicted_labels
    })
    submission_df_raw.to_csv('submission.csv', index=False)
    print("submission.csv file (unsorted) created successfully.")