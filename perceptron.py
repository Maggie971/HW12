import numpy as np
from PIL import Image
import os
from sklearn.preprocessing import LabelBinarizer

class Perceptron:
    def __init__(self, input_size, output_size, learning_rate=0.01, activation_function='sigmoid'):
        self.input_size = input_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.activation_function = activation_function
        
        self.weights = np.random.randn(input_size, output_size)
        self.bias = np.zeros((1, output_size))
    
    def activate(self, x):
        if self.activation_function == 'sigmoid':
            return 1 / (1 + np.exp(-x))
        elif self.activation_function == 'relu':
            return np.maximum(0, x)
        else:
            raise ValueError("Unsupported activation function")
    
    def activate_derivative(self, x):
        if self.activation_function == 'sigmoid':
            return x * (1 - x)
        elif self.activation_function == 'relu':
            return np.where(x > 0, 1, 0)
        else:
            raise ValueError("Unsupported activation function")
    
    def forward(self, x):
        self.z = np.dot(x, self.weights) + self.bias
        self.a = self.activate(self.z)
        return self.a
    
    def backward(self, x, y):
        m = x.shape[0]
        self.error = self.a - y
        self.d_weights = np.dot(x.T, self.error * self.activate_derivative(self.a)) / m
        self.d_bias = np.sum(self.error * self.activate_derivative(self.a), axis=0, keepdims=True) / m
    
    def update_parameters(self):
        self.weights -= self.learning_rate * self.d_weights
        self.bias -= self.learning_rate * self.d_bias
    
    def train(self, x, y, epochs=1000):
        for epoch in range(epochs):
            self.forward(x)
            self.backward(x, y)
            self.update_parameters()
            if epoch % 100 == 0:
                loss = np.mean(np.square(self.error))
                print(f'Epoch {epoch}, Loss: {loss}')
    
    def predict(self, x):
        predictions = self.forward(x)
        return np.argmax(predictions, axis=1)

def load_images_from_folder(folder, size=(20, 20)):
    images = []
    for filename in os.listdir(folder):
        img = Image.open(os.path.join(folder, filename)).convert('L')
        img = img.resize(size)
        img = np.array(img).flatten() / 255.0
        images.append(img)
    return np.array(images)

def load_labels_from_folder(folder):
    labels = []
    for filename in os.listdir(folder):
        label = int(filename.split('.')[0])  # 从文件名中提取标签
        labels.append(label)
    return np.array(labels)

# 数据路径
training_images_path = 'mnist_images/training'
testing_images_path = 'mnist_images/testing'

# 加载数据
train_images = load_images_from_folder(training_images_path)
train_labels = load_labels_from_folder(training_images_path)
test_images = load_images_from_folder(testing_images_path)

# 数据预处理
lb = LabelBinarizer()
train_labels = lb.fit_transform(train_labels)  # 将标签转换为 one-hot 编码

# 初始化感知器
input_size = train_images.shape[1]  # 每张图像的像素数
output_size = train_labels.shape[1]  # 标签的类别数
perceptron = Perceptron(input_size=input_size, output_size=output_size)

# 训练感知器
perceptron.train(train_images, train_labels)

# 预测
test_predictions = perceptron.predict(test_images)
print("测试预测结果:", test_predictions)
