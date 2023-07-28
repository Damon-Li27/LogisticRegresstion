import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


class LogisticRegression:

    def __init__(self, iterations=100, learning_rate=0.01):
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.bias = 0

    def fit(self, x, y):
        """
        梯度下降
        :param x: 输入
        :param y: 真实输出值
        :return:
        """
        # 初始化权重
        self.weight = np.zeros(x.shape[1])
        # 获取训练样本个数
        m = len(x)
        # 迭代指定次数
        for i in range(self.iterations):
            # z = wx + b
            z = np.dot(x, self.weight) + self.bias
            # 使用sigmoid函数归一化
            a = sigmoid(z)
            # 计算损失函数并求均值
            loss_sum = -(y * np.log(a) + (1 - y) * np.log(1 - a))
            loss = 1 / m * np.sum(loss_sum)
            if i % 10 == 0:
                print(f"loss after iteration{i}:{loss}")
            # 损失函数对w和b求编偏导，其中db是一个值
            dz = a - y
            dw = 1 / m * np.dot(dz, x)
            db = 1 / m * (np.sum(dz))
            # 调整参数
            self.weight -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict(self, x):
        y_hat = sigmoid(np.dot(x, self.weight) + self.bias)
        y_hat[y_hat >= 0.5] = 1
        y_hat[y_hat < 0.5] = 0
        return y_hat

    def score(self, y_pred, y):
        flag = y_pred == y
        accuracy = flag.sum() / len(y)
        return accuracy


# 导入数据
iris = load_iris()
x = iris.data[:, :2]  # 花萼长度、花萼宽度
y = iris.target != 0  # 将类别归类为0与非0。  训练集中，花萼species有：0、1、2，分别是山鸢尾（setosa）、变色鸢尾（versicolor）、维吉尼亚鸢尾（virginica）

# train_test_split方法能够将数据集按照用户的需要指定划分为训练集和测试集
# x：所要划分的样本特征集
# y：所要划分的样本结果
# test_size：测试集样本数目与原始样本数目之比，如果是整数的话就是样本的数量
# random_state：是随机数的种子。
# 随机数种子：其实就是该组随机数的编号，在需要重复试验的时候，保证得到一组一样的随机数。比如每次都填1，其他参数一样的情况下你得到的随机数组是一样的。但填0或不填，每次都会不一样。
# stratify是为了保持split前类的分布，例如训练集和测试集数量的比例是 A：B= 4：1，等同于split前的比例（80：20）。通常在这种类分布不平衡的情况下会用到stratify。
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0, stratify=y)

model = LogisticRegression(iterations=2000, learning_rate=0.02)
model.fit(x, y)
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

accuracy_train = model.score(y_train_pred, y_train)
accuracy_test = model.score(y_test_pred, y_test)

print("训练集Accuracy：", accuracy_train)
print("测试集Accuracy：", accuracy_test)
print(f"w value is {model.weight} after iterations")
print(f"b value is {model.bias} after iterations")
