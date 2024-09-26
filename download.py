import tensorflow as tf
from PIL import Image
import numpy as np
import os

# 下载 MNIST 数据集
mnist = tf.keras.datasets.mnist.load_data()
(train_images, train_labels), (test_images, test_labels) = mnist

# 创建输出目录
output_dirs = {
    'training': 'mnist_images/training',
    'testing': 'mnist_images/testing'
}
for folder in output_dirs.values():
    if not os.path.exists(folder):
        os.makedirs(folder)

# 确保每个数字（0-9）只保存一张图像
digit_indices = {i: None for i in range(10)}

for i in range(len(train_labels)):
    digit = train_labels[i]
    if digit_indices[digit] is None:
        digit_indices[digit] = i
    if all(idx is not None for idx in digit_indices.values()):
        break

# 保存每个数字的训练图像
for digit, idx in digit_indices.items():
    if idx is not None:
        img = Image.fromarray(train_images[idx], 'L')
        img = img.resize((20, 20))
        img_path = os.path.join(output_dirs['training'], f'{digit}.png')
        img.save(img_path)
        print(f"保存了数字 {digit} 的训练图像")

# 保存一些测试图像为 PNG 文件
num_test_samples = 10
for i in range(num_test_samples):
    img = Image.fromarray(test_images[i], 'L')
    img = img.resize((20, 20))
    img_path = os.path.join(output_dirs['testing'], f'{i}.png')
    img.save(img_path)
    print(f"保存了测试图像 {i} 张")
