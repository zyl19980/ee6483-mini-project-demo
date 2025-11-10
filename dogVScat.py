import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import os
import random
import numpy as np

# 固定随机种子
os.environ['PYTHONHASHSEED'] = '42'
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

# --- 1. 定义常量 (与上一步相同) ---

# !! 修改为你本地的实际路径 !!
TRAIN_DIR = 'datasets/train'
VAL_DIR = 'datasets/val'

IMG_SIZE = (128, 128)  # 目标图像尺寸
BATCH_SIZE = 32
# 我们先训练 10 个周期 (epochs) 看看效果
# 一个 epoch = 模型看完了所有训练数据一次
EPOCHS = 100 


# --- 2. 准备数据 (与上一步相同) ---

# 训练数据生成器：
# 添加随机旋转、平移、缩放和翻转来进行数据增强
# 数据增强
train_datagen = ImageDataGenerator(
    rescale=1./255,          # 归一化 (必须)
    rotation_range=20,       # 随机旋转 -20 到 +20 度
    width_shift_range=0.1,   # 随机水平平移 10%
    height_shift_range=0.1,  # 随机垂直平移 10%
    zoom_range=0.1,          # 随机缩放 10%
    horizontal_flip=True,    # 随机水平翻转
    fill_mode='nearest'      # 填充因变换产生的空白区域
)
# 验证数据生成器：
# 验证集也必须做同样的归一化
val_datagen = ImageDataGenerator(rescale=1./255)

print("Preparing training data generator...")
train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary'       # 二分类
)

print("Preparing validation data generator...")
val_generator = val_datagen.flow_from_directory(
    VAL_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary'
)

print(f"Class indices: {train_generator.class_indices}")
# 确认 {'cat': 0, 'dog': 1}

# --- 3. 搭建你的基线CNN模型 (全新内容) ---
# 这就是项目报告 c) 部分要求你描述的模型 

print("\nBuilding simple CNN model...")
model = models.Sequential()

# Keras 2.9+ 推荐使用 Input 层显式声明输入形状
# 我们的输入是 128x128 像素的彩色 (3通道) 图片
model.add(layers.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3)))

# 第 1 组: 卷积 + 池化
# Conv2D: 32个滤镜, 每个滤镜 3x3 大小。它会学习识别边缘、纹理等。
model.add(layers.Conv2D(32, (3, 3), activation='relu'))
# MaxPooling2D: 将特征图缩小一半，保留最显著的特征
model.add(layers.MaxPooling2D((2, 2)))

# 第 2 组: 卷积 + 池化
# 增加滤镜数量 (64)，学习更复杂的组合特征 (比如 "眼睛" 或 "鼻子")
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

# 第 3 组: 卷积 + 池化
# 再增加滤镜数量 (128)，学习更高级的特征
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

# "压扁"：将 2D 的特征图转换成 1D 向量，送入"大脑"
model.add(layers.Flatten())

# "大脑" (全连接层)
# Dense: 128个神经元，它会分析所有特征并进行"思考"
model.add(layers.Dense(128, activation='relu'))

# "输出层" (关键！)
# Dense: 只有 1 个神经元
# activation='sigmoid': 这是二分类的核心。
# Sigmoid 函数会输出一个 0 到 1 之间的概率值
# < 0.5 代表 'cat' (标签0), > 0.5 代表 'dog' (标签1)
model.add(layers.Dense(1, activation='sigmoid'))

# 打印模型结构，检查参数量
model.summary()

# --- 4. 编译模型 (全新内容) ---
# 这是项目报告 c) 部分要求的损失函数和训练策略 [cite: 47]

print("\nCompiling model...")
model.compile(
    # 优化器: 'adam' 是目前最常用、效果最好的优化器之一
    # learning_rate: 学习率，这是你需要调优的超参数 [cite: 50]
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    
    # 损失函数: 'binary_crossentropy'
    # 这是二分类 (0 vs 1) 问题的“标准计分板”
    loss='binary_crossentropy',
    
    # 评估指标: 我们最关心的是 'accuracy' (准确率)
    metrics=['accuracy']
)

# --- 5. 训练模型 (全新内容) ---
# 这就是 "fit" 过程，模型开始从数据中学习

# 5.1. 定义 Callbacks
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
# 监控 'val_loss'，如果 7 个周期 (patience=7) 都没有改善，就停止训练
early_stopper = EarlyStopping(
    monitor='val_loss',
    patience=7,
    verbose=1,
    restore_best_weights=True  # 自动恢复到最佳权重点 (推荐)
)

# 监控 'val_loss'，只保存 (save_best_only) 表现最好的模型
# 这是你最终应该用来提交预测的模型
model_checkpoint = ModelCheckpoint(
    'best_dog_vs_cat_model.keras', # 保存到新的文件名
    monitor='val_loss',
    save_best_only=True,
    verbose=1
)

print("\nStarting model training...")
# .fit() 会返回一个 'history' 对象，包含了训练过程中的所有指标
history = model.fit(
    train_generator,  # 我们的训练数据
    
    # Keras 需要知道 ImageDataGenerator 一共有多少步
    # 总样本数 // 批大小
    steps_per_epoch=train_generator.samples // BATCH_SIZE,
    
    epochs=EPOCHS, # 训练 10 个周期
    
    validation_data=val_generator, # 我们的验证数据
    
    # 验证集有多少步
    validation_steps=val_generator.samples // BATCH_SIZE,

    # 3. 传入 Callbacks
    callbacks=[early_stopper, model_checkpoint]
)

print("Training complete.")

# --- 6. (可选) 可视化训练结果 ---
# 我们可以画出训练和验证的准确率，看看模型学得怎么样

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(len(acc))

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')

plt.savefig('training_plots.png')
print("Training plot saved to training_plots.png")

# --- 7. (可选) 保存模型 ---
# 训练好的模型可以保存下来，以便后续用于预测 [cite: 32]
# model.save('my_baseline_cnn_model_after_dataArgument.keras')
# print("Model saved as 'my_baseline_cnn_model.keras'")