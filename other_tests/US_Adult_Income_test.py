import pandas as pd
import numpy as np
# 用于编码的类
from sklearn.preprocessing import LabelEncoder

# 朴素贝叶斯类
from sklearn.naive_bayes import GaussianNB

# 交叉验证方法
from sklearn.model_selection import cross_val_score

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.naive_bayes import MultinomialNB

def normalize(data):
    """
    normalize to [-1, 1]
    :param x:
    :param data:
    :return:
    """
    # mean = np.mean(x)
    Xmax = np.max(data)
    Xmin = np.min(data)
    return 2 * (data - Xmin) / (Xmax - Xmin) - 1

train_data_path = "./data/diabetes.csv"
test_data_path = "./data/diabetes.csv"
eps = 0.5
p = np.exp(eps/2)/(np.exp(eps/2) + 1)
q = 1 / (np.exp(eps/2) + 1)


def get_data():
    train_data_df = pd.read_csv(train_data_path)
    test_data_df = pd.read_csv(test_data_path)
    return train_data_df, test_data_df

def process_data_diabetes(train_data_df):

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
        if index == 8:
            # 对于每一个非数字的列，分别创建一个标签编码器，便于后期用来预测样本
            # 每一个标签编码器分别使用各自列的数据进行编码，这样预测数据时可以不用再训练标签分类器，直接对需要预测的样本数据进行编码
            label_encoder.append(LabelEncoder())
            train_data_encode[:, index] = label_encoder[-1].fit_transform(train_data[:, index])
            # print("discrete: {}".format(index))
            discrete_index.append(index)
        elif type(item) == int or type(item) == np.float64:
            # temp = normalize(train_data[:, index])
            train_data_encode[:, index] = train_data[:, index]
            # print("continuous: {}".format(index))
            continuous_index.append(index)


    if len(discrete_index) != 0:
        discrete_index.pop()        # 最后一列是类别，不属于属性


    X_train = train_data_encode[:, :-1]
    Y_train = train_data_encode[:, -1].astype(int)

    num_class = len(set(Y_train))

    for i in discrete_index:
        attr = X_train[:, i]
        attr_mapping = [(num_class * attr_value + label + 1) for attr_value, label in zip(attr, Y_train)]
        X_train[:, i] = attr_mapping

    return X_train, Y_train, train_data_encode, continuous_index, discrete_index


def process_data(train_data_df):
    col_labels = ['age', 'workclass', 'fnlwgt', 'education', 'education_num', 'marital_status',
                  'occupation', 'relationship', 'race', 'sex', 'capital_gain', 'capital_loss',
                  'hours_per_week', 'native_country', 'wage_class']
    train_data_df.columns = col_labels

    str_cols = [1, 3, 5, 6, 7, 8, 9, 13, 14]
    for col in str_cols:
        train_data_df.iloc[:, col] = train_data_df.iloc[:, col].map(lambda x: x.strip())

    train_data_df = train_data_df.replace("?", np.nan).dropna()

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
        if type(item) == int and index != 4:
            train_data_encode[:, index] = train_data[:, index]
            # print("continuous: {}".format(index))
            continuous_index.append(index)
        else:
            # 对于每一个非数字的列，分别创建一个标签编码器，便于后期用来预测样本
            # 每一个标签编码器分别使用各自列的数据进行编码，这样预测数据时可以不用再训练标签分类器，直接对需要预测的样本数据进行编码
            label_encoder.append(LabelEncoder())
            train_data_encode[:, index] = label_encoder[-1].fit_transform(train_data[:, index])
            # print("discrete: {}".format(index))
            discrete_index.append(index)

    # train_data_encode = np.delete(train_data_encode, [2, 10, 11], axis=1)
    # discrete = train_data[:, discrete_index]
    # print(discrete)

    X_train = train_data_encode[:, :-1].astype(int)
    Y_train = train_data_encode[:, -1].astype(int)

    return X_train, Y_train, train_data_encode, continuous_index, discrete_index


def SUE_method(data, length, num_data):
    """
    这是没有经过扰动的SUE
    :param data: 列向量，要编码的数据
    :param length: 转化的二进制向量长度
    :param num_data: 列向量数据数量
    :return:
    """
    # 编码
    # 对应位设置为1
    binary_vec_pert = np.zeros((num_data, length))
    for i in range(num_data):
        value = data[i]
        binary_vec_pert[i][value] = 1

    # 每个位的计数
    num_of_bits = sum(binary_vec_pert).astype(int)

    return num_of_bits



def SUE_encode(data, length, num_data):
    """
    这是加了扰动的SUE
    :param data: 列向量，要编码的数据
    :param length: 转化的二进制向量长度
    :param num_data: 列向量数据数量
    :return:
    """
    # 编码
    # 对应位设置为1
    binary_vec_pert = np.zeros((num_data, length))
    for i in range(num_data):
        value = data[i]
        binary_vec_pert[i][value] = 1


    # 扰动
    for i in range(len(binary_vec_pert)):
        for j in range(length):
            x = np.random.rand()
            if x < q:
                if binary_vec_pert[i][j] == 1:
                    binary_vec_pert[i][j] = 0
                elif binary_vec_pert[i][j] == 0:
                    binary_vec_pert[i][j] = 1

    # 每个位的计数
    num_of_bits = sum(binary_vec_pert).astype(int)

    # 每个位的估计次数，就是不同值的次数
    estimate_num = []
    for item in num_of_bits:
        p_c = (item - num_data * q) / (p - q)
        estimate_num.append(p_c.astype(int))
    return estimate_num

# def SUE_for_class_test(Y_data):
#     length = len(set(Y_data))
#     num_data = Y_data.shape[0]
#     estimate_num_test = SUE_encode(data=Y_data, length=length, num_data=num_data)
#     print("estimate_num_test: ", estimate_num_test)
#     estimate_num_orig = SUE_method(data=Y_train, length=length, num_data=num_data)
#     print("estimate_num_orig: ", estimate_num_orig)


def SUE_for_class(Y_data):
    """
    类别的SUE
    :param Y_data: 类别信息
    :return: 添加扰动后计算的各个类别的估计次数和先验概率
    """
    num_class = len(set(Y_data))
    print(num_class)
    num_data = Y_data.shape[0]

    # 扰动后的估计量
    estimate_sum_class = SUE_encode(data=Y_data, length=num_class, num_data=num_data)

    conditional_prob_pert = []

    for i in range(len(estimate_sum_class)):
        conditional_prob_pert.append(estimate_sum_class[i]/sum(estimate_sum_class))

    print("conditional_prob_pert: ", conditional_prob_pert)

    return estimate_sum_class, conditional_prob_pert





def SUE_map_discrete_attr(X_data, Y_data, discrete_index):
    num_class = len(set(Y_data))
    num_data = Y_data.shape[0]
    Y_data = Y_data.reshape((Y_data.shape[0], 1))
    data = np.concatenate((X_data, Y_data), axis=1)         # (30162,11)
    data_pert = np.zeros(data.shape).astype(int)
    # for col in range(data.shape[1]-1):
    #     for row in range(num_data):
    #
    # num_of_all_attr = []
    # # 扰动后的估计量
    # for i in range(X_data.shape[1]):
    #     if i in discrete_index:
    #
    #         estimate_sum_attr = SUE_encode(data=X_data[:,i], length=num_class, num_data=num_data)
    #     num_of_all_attr.append(estimate_sum_attr)




def Bayes_Model(X_train, Y_train, X_test, Y_test):
    # 建立朴素贝叶斯分类器模型
    gaussianNB = GaussianNB()
    gaussianNB.fit(X_train, Y_train)

    # 2 用交叉验证来检验模型的准确性，只是在test set上验证准确性
    num_validations = 5
    accuracy = cross_val_score(gaussianNB, X_test, Y_test,
                               scoring='accuracy', cv=num_validations)
    print('准确率：{:.2f}%'.format(accuracy.mean() * 100))
    precision = cross_val_score(gaussianNB, X_test, Y_test,
                                scoring='precision_weighted', cv=num_validations)
    print('精确度：{:.2f}%'.format(precision.mean() * 100))
    recall = cross_val_score(gaussianNB, X_test, Y_test,
                             scoring='recall_weighted', cv=num_validations)
    print('召回率：{:.2f}%'.format(recall.mean() * 100))
    f1 = cross_val_score(gaussianNB, X_test, Y_test,
                         scoring='f1_weighted', cv=num_validations)
    print('F1  值：{:.2f}%'.format(f1.mean() * 100))

    # 3 打印性能报告
    y_pred = gaussianNB.predict(X_test)
    confusion_mat = confusion_matrix(Y_test, y_pred)
    print(confusion_mat)  # 混淆矩阵

    # 直接使用sklearn打印精度，召回率和F1值
    target_names = ['0', '1']
    print(classification_report(Y_test, y_pred, target_names=target_names))


# def Bayes_Model_MultinomialNB(X_train, Y_train, X_test, Y_test):
#     # 建立朴素贝叶斯分类器模型
#
#     multinomialNB = MultinomialNB()
#     multinomialNB.fit(X_train, Y_train)
#
#     # 2 用交叉验证来检验模型的准确性，只是在test set上验证准确性
#     num_validations = 5
#     accuracy = cross_val_score(multinomialNB, X_test, Y_test,
#                                scoring='accuracy', cv=num_validations)
#     print('准确率：{:.2f}%'.format(accuracy.mean() * 100))
#     precision = cross_val_score(multinomialNB, X_test, Y_test,
#                                 scoring='precision_weighted', cv=num_validations)
#     print('精确度：{:.2f}%'.format(precision.mean() * 100))
#     recall = cross_val_score(multinomialNB, X_test, Y_test,
#                              scoring='recall_weighted', cv=num_validations)
#     print('召回率：{:.2f}%'.format(recall.mean() * 100))
#     f1 = cross_val_score(multinomialNB, X_test, Y_test,
#                          scoring='f1_weighted', cv=num_validations)
#     print('F1  值：{:.2f}%'.format(f1.mean() * 100))
#
#     # 3 打印性能报告
#     y_pred = multinomialNB.predict(X_test)
#     confusion_mat = confusion_matrix(Y_test, y_pred)
#     print(confusion_mat)  # 混淆矩阵
#
#     # 直接使用sklearn打印精度，召回率和F1值
#     target_names = ['<=50K', '>50K']
#     print(classification_report(Y_test, y_pred, target_names=target_names))



if __name__ == '__main__':
    train_data_df, test_data_df = get_data()
    X_train, Y_train, train_data_encode, continuous_index, discrete_index = process_data_diabetes(train_data_df)
    X_test, Y_test, test_data_encode, _, _ = process_data_diabetes(test_data_df)
    Bayes_Model(X_train, Y_train, X_test, Y_test)

    # estimate_sum_class, conditional_pro_pert = SUE_for_class(Y_train)



    # SUE_for_class_test(Y_train)
    # conditional_prob_test = SUE_for_class(Y_test)
    #
    # print(discrete_index)
    # SUE_map_discrete_attr(train_data_encode[:, discrete_index], Y_train)












