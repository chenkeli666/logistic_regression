import numpy as np
import json
import matplotlib.pyplot as plt

trainSet_path = "trainset.json"
devSet_path = "devset.json"
testSet_path = "devset.json"


def read_data(Set_path):
    """
        读取训练集或验证集
    """
    with open(Set_path) as f:
        Set = json.load(f)
    X_List = list()
    Y_List = list()
    for XY_List in Set:
        X_List.append(XY_List[0])
        Y_List.append(XY_List[1])
    Set_X = np.array(X_List)
    Set_Y = np.array(Y_List)
    return Set_X, Set_Y


def write_result(Set_path, w, b):
    """
        测试集结果并写入 testset.json
    """
    with open(Set_path) as f:
        Set = json.load(f)
    X_List = list()
    Y_List = list()
    for X in Set:
        X_List.append(X[0])
    Set_X = np.array(X_List)
    Set_Y = np.dot(w, Set_X.T) + b
    for Y in Set_Y:
        if Y>=0.5:
            Y_List.append(1)
        else:
            Y_List.append(0)
    output = []
    for i in range(len(Y_List)):
        output.append([X_List[i], Y_List[i]])
    #print(output)
    with open('testset_output.json', 'w') as f:
        json.dump(output, f, indent=2)


class Logistic_Regression:
    """
        learningRate：学习速率
        max_iter：最大迭代次数
    """
    def __init__(self, learningRate=0.005, max_iter=50000):
        self.learningRate = learningRate
        self.max_iter = max_iter
        self.w = np.random.random(9) * 0.1
        self.b = 0

    def train(self, TX, TY, DX, DY):
        """
        梯度下降法训练
        参数说明：
            TX：训练集数据矩阵
            TY：训练集标签矩阵
            DX：验证集训练矩阵
            DY：验证集标签矩阵
            w：权值
            b：偏置
            loss_T:训练集损失值
            loss_T：验证集损失值
        """
        M_T, N_T = TX.shape
        M_D, N_D = DX.shape
        x_point = list()
        y_point_T = list()
        y_point_D = list()
        for i in range(self.max_iter):
            # 计算损失函数微分，迭代更新
            TZ = np.dot(self.w, TX.T) + self.b
            DZ = np.dot(self.w, DX.T) + self.b
            dz = self.sigmod(TZ)-TY   # dz:1*1000
            dw = 1/M_T * np.dot(dz, TX)
            db = 1/M_T * np.sum(dz)
            self.w = self.w - self.learningRate * dw  # 更新w
            self.b = self.b - self.learningRate * db  # 更新b
            # loss
            loss_T = -1 / M_T * np.sum(TY*np.log(self.sigmod(TZ)) + (1-TY)*np.log(1-self.sigmod(TZ)))
            loss_D = -1 / M_D * np.sum(DY*np.log(self.sigmod(DZ)) + (1-DY)*np.log(1-self.sigmod(DZ)))
            x_point.append(i)
            y_point_T.append(loss_T)
            y_point_D.append(loss_D)
        # matplotlib
        self.plot(x_point, y_point_T, y_point_D)
        return self.w, self.b

    def print_accuracy(self, X, Y):
        """
            输出训练集或验证集的精度
        """
        z = np.dot(self.w, X.T) + self.b
        T = 0
        num = len(Y)
        for i in range(num):
            if z[i]>0.5 and Y[i]==1:
                T = T + 1
            if z[i]<0.5 and Y[i]==0:
                T = T + 1
        accuracy = T/num
        print(accuracy)

    def plot(self, x, y_T, y_D):
        plt.plot(x, y_T, color='red', label='train')
        plt.plot(x, y_D, color='blue',label='valid')
        plt.legend()
        plt.xlabel('iterations')
        plt.ylabel('loss')
        plt.show()

    def sigmod(self, z):
        return 1/(1 + np.exp(-z))

if __name__ == '__main__':
    trainSet_data, trainSet_label = read_data(trainSet_path)    #读取训练集
    devSet_data, devSet_label = read_data(devSet_path)  # 读取验证集
    model = Logistic_Regression()
    w, b = model.train(trainSet_data, trainSet_label, devSet_data, devSet_label)    # 训练并验证
    print("Associate rate of trainSet:")
    model.print_accuracy(trainSet_data, trainSet_label)    # 训练集精度
    print("Associate rate of devSet:")
    model.print_accuracy(devSet_data, devSet_label)    # 验证集精度
    write_result(testSet_path, w, b)    # 测试集
