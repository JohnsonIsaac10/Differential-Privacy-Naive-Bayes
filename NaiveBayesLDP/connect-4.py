import NaiveBayesLDP
from NaiveBayesLDP import DataAggregator, Individuals, NaiveBayesLDP, normalize, Bayes_Model
import pandas as pd
import numpy as np
# 用于编码的类
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from pandas.core.frame import DataFrame

train_data_path = "./data/connect-4.csv"
test_data_path = "./data/connect-4.csv"


def process_test_data(test_data_df):
    test_data = test_data_df.values

    # 标签编码器列表
    label_encoder = []

    # 存放编码后的数据，先指定为空的二维数组，再进行赋值
    test_data_encode = np.empty(test_data.shape)

    # index是data每一个元素的下标，item是data每一个元素的内容
    for index, item in enumerate(test_data[0]):
        # 若该列数据是是数字，直接赋值，否则先进行编码
        if type(item) == int or type(item) == np.float64 or type(item) == float:
            temp = normalize(test_data[:, index])
            test_data_encode[:, index] = temp
        else:
            # 对于每一个非数字的列，分别创建一个标签编码器，便于后期用来预测样本
            # 每一个标签编码器分别使用各自列的数据进行编码，这样预测数据时可以不用再训练标签分类器，直接对需要预测的样本数据进行编码
            label_encoder.append(LabelEncoder())
            test_data_encode[:, index] = label_encoder[-1].fit_transform(test_data[:, index])

    X_test = test_data_encode[:, :-1]
    Y_test = test_data_encode[:, -1].astype(int)

    return X_test, Y_test


def process_train_data(train_data_df):
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
        if type(item) == int:
            temp = normalize(train_data[:, index])
            train_data_encode[:, index] = temp
            continuous_index.append(index)
        else:
            # 对于每一个非数字的列，分别创建一个标签编码器，便于后期用来预测样本
            # 每一个标签编码器分别使用各自列的数据进行编码，这样预测数据时可以不用再训练标签分类器，直接对需要预测的样本数据进行编码
            label_encoder.append(LabelEncoder())
            train_data_encode[:, index] = label_encoder[-1].fit_transform(train_data[:, index])
            discrete_index.append(index)

    if len(discrete_index) != 0:
        discrete_index.pop()  # 最后一列是类别，不属于属性

    X_train = train_data_encode[:, :-1]
    Y_train = train_data_encode[:, -1].astype(int)

    return X_train, Y_train, train_data_encode, continuous_index, discrete_index


def split_test(data, test_size=0.2):
    X_train = data[:, :-1]
    Y_train = data[:, -1].astype(int)
    X_train, X_test, Y_train, Y_test = train_test_split(X_train, Y_train, test_size=test_size)
    return X_test, Y_test


def process_data(train_data_path, test_data_path="", test_size=0.2):
    if test_data_path == "":
        train_df = pd.read_csv(train_data_path)
        X_train, Y_train, train_data_encode, continuous_index, discrete_index = process_train_data(train_df)
        X_test, Y_test = split_test(data=train_data_encode, test_size=test_size)

        return X_train, Y_train, X_test, Y_test, continuous_index
    elif test_data_path != "":
        train_df = pd.read_csv(train_data_path)
        test_df = pd.read_csv(test_data_path)
        X_train, Y_train, train_data_encode, continuous_index, discrete_index = process_train_data(train_df)
        num_class = len(set(Y_train))
        X_test, Y_test = process_test_data(test_df)

        return X_train, Y_train, X_test, Y_test, continuous_index


if __name__ == '__main__':
    eps_dis = 5
    eps_con = 2
    X_train, Y_train, X_test, Y_test, continuous_index = process_data(train_data_path="./data/connect-4.csv",
                                                                      test_data_path="./data/connect-4.csv")

    import matplotlib.pyplot as plt

    eps_dis_list = [0.1, 0.2, 0.4, 0.6, 0.8, 1, 1.2, 1.4, 1.6, 1.8, 2, 2.5, 3, 3.5, 4, 4.5, 5, 6, 7]

    acc_Bayes_list_mean = []
    precision_Bayes_list_mean = []
    recall_Bayes_list_mean = []

    acc_LDP_list_mean = []
    precision_LDP_list_mean = []
    recall_LDP_list_mean = []

    test_times = 50

    for i in range(test_times):
        acc_LDP_list = []
        precision_LDP_list = []
        recall_LDP_list = []

        acc_Bayes_list = []
        precision_Bayes_list = []
        recall_Bayes_list = []

        for eps_dis in eps_dis_list:
            _, X_test, _, Y_test = train_test_split(X_train, Y_train, test_size=0.2)
            naiveBayesLDP = NaiveBayesLDP(eps_con=eps_con, eps_dis=eps_dis)
            naiveBayesLDP.fit(X_train=X_train, Y_train=Y_train, continuous_index=continuous_index)
            acc_LDP, _, precision_LDP, recall_LDP = naiveBayesLDP.predict(X_test=X_test, Y_test=Y_test,
                                                                          continuous_index=continuous_index)
            acc_Bayes, precision_Bayes, recall_Bayes = Bayes_Model(X_train, Y_train, X_test, Y_test)

            acc_LDP_list.append(acc_LDP)
            precision_LDP_list.append(acc_LDP)
            recall_LDP_list.append(recall_LDP)

            acc_Bayes_list.append(acc_Bayes)
            precision_Bayes_list.append(precision_Bayes)
            recall_Bayes_list.append(recall_Bayes)

        acc_Bayes_list_mean.append(acc_Bayes_list)
        precision_Bayes_list_mean.append(precision_Bayes_list)
        recall_Bayes_list_mean.append(recall_Bayes_list)

        acc_LDP_list_mean.append(acc_LDP_list)
        precision_LDP_list_mean.append(precision_LDP_list)
        recall_LDP_list_mean.append(recall_LDP_list)

    acc_Bayes_list_mean = np.mean(acc_Bayes_list_mean, axis=0)
    precision_Bayes_list_mean = np.mean(precision_Bayes_list_mean, axis=0)
    recall_Bayes_list_mean = np.mean(recall_Bayes_list_mean, axis=0)

    acc_LDP_list_mean = np.mean(acc_LDP_list_mean, axis=0)
    precision_LDP_list_mean = np.mean(precision_LDP_list_mean, axis=0)
    recall_LDP_list_mean = np.mean(recall_LDP_list_mean, axis=0)
    plt.style.use(['science', 'ieee', 'no-latex'])

    plt.ylim(0, 1)
    plt.grid(linestyle="--")
    plt.plot(eps_dis_list, acc_Bayes_list_mean, label="Bayes", linestyle='-')
    plt.plot(eps_dis_list, acc_LDP_list_mean, label="LDP_Bayes", linestyle='-')
    plt.xlabel("ε")
    plt.ylabel("Accuracy")
    plt.legend(loc="lower right")
    plt.savefig("./ieee_fig_new/connect-4_ieee_acc.png")
    plt.show()

    plt.ylim(0, 1)
    plt.grid(linestyle="--")
    plt.plot(eps_dis_list, precision_Bayes_list_mean, label="Bayes", linestyle='-')
    plt.plot(eps_dis_list, precision_LDP_list_mean, label="LDP_Bayes", linestyle='-')
    plt.xlabel("ε")
    plt.ylabel("Precision")
    plt.legend(loc="lower right")
    plt.savefig("./ieee_fig_new/connect-4_ieee_precision.png")
    plt.show()

    plt.ylim(0, 1)
    plt.grid(linestyle="--")
    plt.plot(eps_dis_list, recall_Bayes_list_mean, label="Bayes", linestyle='-')
    plt.plot(eps_dis_list, recall_LDP_list_mean, label="LDP_Bayes", linestyle='-')
    plt.xlabel("ε")
    plt.ylabel("Recall")
    plt.legend(loc="lower right")
    plt.savefig("./ieee_fig_new/connect-4_ieee_recall.png")
    plt.show()
