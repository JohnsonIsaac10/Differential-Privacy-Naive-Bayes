from sklearn.model_selection import train_test_split

import NaiveBayesLDP
from NaiveBayesLDP import DataAggregator, Individuals, NaiveBayesLDP, normalize, Bayes_Model

import pandas as pd
import numpy as np
# 用于编码的类
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler



train_data_path = "./data/weatherAUS.csv"
test_data_path = "./data/weatherAUS.csv"
# eps_dis = 1
# p_dis = np.exp(eps_dis/2)/(np.exp(eps_dis/2) + 1)
# q_dis = 1 / (np.exp(eps_dis/2) + 1)


def get_data():
    train_data_df = pd.read_csv(train_data_path)
    test_data_df = pd.read_csv(test_data_path)
    return train_data_df, test_data_df


def process_train_data(train_data_df):

    train_data_df.drop(["Date"], inplace=True, axis=1)
    train_data_df.dropna(inplace=True)
    train_data_df.index = range(len(train_data_df))


    # train_data_df.to_csv("./data/test_data.csv")

    train_data = train_data_df.values

    # 连续属性的索引
    continuous_index = []

    # 离散属性的索引
    discrete_index = []

    # 标签编码器列表
    label_encoder = []

    # 存放编码后的数据，先指定为空的二维数组，再进行赋值
    train_data_encode = np.empty(train_data.shape)

    # index是data每一个元素的下标，item是data每一个元素的内容
    for index, item in enumerate(train_data[0]):
        # 若该列数据是是数字，直接赋值，否则先进行编码
        if (type(item) == int or type(item) == float or type(item) == np.float64) and index != 16 and index != 17:
            temp = normalize(train_data[:, index])
            train_data_encode[:, index] = temp
            # print("continuous: {}".format(index))
            continuous_index.append(index)
        else:
            # 对于每一个非数字的列，分别创建一个标签编码器，便于后期用来预测样本
            # 每一个标签编码器分别使用各自列的数据进行编码，这样预测数据时可以不用再训练标签分类器，直接对需要预测的样本数据进行编码
            label_encoder.append(LabelEncoder())
            train_data_encode[:, index] = label_encoder[-1].fit_transform(train_data[:, index])
            # print("discrete: {}".format(index))
            discrete_index.append(index)

    discrete_index.pop()        # 最后一列是类别，不属于属性

    # train_data_encode = np.delete(train_data_encode, [2, 10, 11], axis=1)
    # discrete = train_data[:, discrete_index]
    # print(discrete)

    X_train = train_data_encode[:, :-1]
    Y_train = train_data_encode[:, -1].astype(int)

    return X_train, Y_train, continuous_index, discrete_index



if __name__ == '__main__':
    eps_dis = 2
    eps_con = 2
    train_data_df, test_data_df = get_data()
    X_train, Y_train, continuous_index, discrete_index = process_train_data(train_data_df)
    import matplotlib.pyplot as plt

    eps_dis_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.8, 1, 1.5, 2, 2.5, 3, 4]
    eps_con_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.8, 1, 1.5, 2, 2.5, 3, 4]
    acc_Bayes_list_mean = []
    acc_LDP_list_mean = []
    test_times = 50
    for i in range(test_times):
        acc_LDP_list = []
        acc_Bayes_list = []
        for eps_dis, eps_con in zip(eps_dis_list, eps_con_list):
            _, X_test, _, Y_test = train_test_split(X_train, Y_train, test_size=0.2)
            naiveBayesLDP = NaiveBayesLDP(eps_con=eps_con, eps_dis=eps_dis)
            naiveBayesLDP.fit(X_train=X_train, Y_train=Y_train, continuous_index=continuous_index)
            acc_LDP = naiveBayesLDP.predict(X_test=X_test, Y_test=Y_test, continuous_index=continuous_index)
            acc_Bayes = Bayes_Model(X_train, Y_train, X_test, Y_test)
            acc_LDP_list.append(acc_LDP)
            acc_Bayes_list.append(acc_Bayes)
        acc_Bayes_list_mean.append(acc_Bayes_list)
        acc_LDP_list_mean.append(acc_LDP_list)

    acc_Bayes_list_mean = np.mean(acc_Bayes_list_mean, axis=0)
    acc_LDP_list_mean = np.mean(acc_LDP_list_mean, axis=0)

    plt.ylim(0, 1)
    plt.grid(linestyle="--")
    plt.plot(eps_dis_list, acc_Bayes_list_mean, label="Bayes", marker="o")
    plt.plot(eps_dis_list, acc_LDP_list_mean, label="Bayes_LDP", marker="s")
    plt.xlabel("ε")
    plt.ylabel("Accuracy")
    plt.legend(loc="lower right")
    plt.savefig("./savefigure2/weatherAUS.png")
    plt.show()