import pandas as pd
import numpy as np
# 用于编码的类
from sklearn.metrics import confusion_matrix, precision_score, recall_score
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB, CategoricalNB
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
import random

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


def Gaussian_Distribution(value, mean, variance):

    return 1/(np.sqrt(2*np.pi*variance))*np.exp(-((value - mean) ** 2)/(2*variance))


train_data_path = "./data/adult-training.csv"
test_data_path = "./data/adult-test.csv"
eps_dis = 1.5
eps_con = 2.5
# p_dis = np.exp(eps_dis/2)/(np.exp(eps_dis/2) + 1)
# q_dis = 1 / (np.exp(eps_dis/2) + 1)


def get_data():
    train_data_df = pd.read_csv(train_data_path, header=None)
    test_data_df = pd.read_csv(test_data_path, header=None)
    return train_data_df, test_data_df


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

    num_class = len(set(Y_train))

    for i in discrete_index:
        attr = X_train[:, i]
        attr_mapping = [(num_class * attr_value + label + 1) for attr_value, label in zip(attr, Y_train)]
        X_train[:, i] = attr_mapping

    return X_train, Y_train, train_data_encode, continuous_index, discrete_index

class Individuals:
    def __init__(self, eps_dis = 0.5, eps_con = 0.5):
        self.num_class = 0
        self.num_data = 0
        self.eps_dis = eps_dis
        self.p_dis = np.exp(eps_dis / 2) / (np.exp(eps_dis / 2) + 1)
        self.q_dis = 1 / (np.exp(eps_dis / 2) + 1)
        self.eps_con = eps_con
        # print("Individuals")
        # print("eps_dis: ", eps_dis)
        # print("eps_con: ", eps_con)
        # print("self.p_dis: ", self.p_dis)
        # print("self.q_dis: ", self.q_dis)


    def SUE_encode(self, data, length, num_data):
        """
        这是加了扰动的SUE
        :param data: 列向量，要编码的数据
        :param length: 转化的二进制向量长度
        :param num_data: 列向量数据数量
        :return:SUE扰动编码向量
        """
        # 编码
        # 对应位设置为1
        binary_vec_pert = np.zeros((num_data, length))
        binary_vec = np.zeros((num_data, length))
        for i in range(num_data):
            value = data[i]
            # if value == 159:
            #     print("here")
            binary_vec_pert[i][value] = 1
            binary_vec[i][value] = 1
        # 扰动
        for i in range(binary_vec_pert.shape[0]):
            for j in range(length):
                x = np.random.rand()
                if x < self.q_dis:
                    if binary_vec_pert[i][j] == 1:
                        binary_vec_pert[i][j] = 0
                    elif binary_vec_pert[i][j] == 0:
                        binary_vec_pert[i][j] = 1

        return binary_vec_pert, binary_vec

    def SUE_for_class(self, Y_data):
        """
        类别的SUE
        :param Y_data: 类别信息
        :return: 添加扰动后计算的各个类别的估计次数和先验概率
        """
        self.num_class = len(set(Y_data))
        self.num_data = Y_data.shape[0]

        # 扰动后的估计量
        vec_pert_class, _ = self.SUE_encode(data=Y_data, length=self.num_class, num_data=self.num_data)
        return vec_pert_class

    def SUE_map_discrete_attr(self, attr, Y_data):
        self.num_class = len(set(Y_data))

        attr_count = len(set(attr))         # 某一属性可能的取值数

        length = self.num_class * attr_count+1
        # length = max(attr) + 1
        self.num_data = Y_data.shape[0]
        Y_data = Y_data.reshape((Y_data.shape[0], 1))

        # 属性映射
        attr_mapping = [(self.num_class * attr_value + label + 1) for attr_value, label in zip(attr, Y_data)]
        # if attr[68] ==

        vec_pert_attr, binary_vec = self.SUE_encode(data=attr_mapping, length=length, num_data=self.num_data)

        return vec_pert_attr, attr_count, binary_vec

    def PM_continuous_attr(self, attr):
        """
        每个属性值用PM添加扰动，返回扰动向量，属性值应归一化normalize
        :param attr:
        :return: 属性扰动值，属性平方后的扰动值
        """
        self.num_data = len(attr)
        attr_square = np.square(attr)
        attr_pert = np.zeros(attr.shape)
        attr_pert_square = np.zeros(attr.shape)

        for i in range(len(attr)):
            attr_pert[i] = self.PM_one_dim(attr[i], self.eps_con)
            attr_pert_square[i] = self.PM_one_dim(attr_square[i], self.eps_con)

        return attr_pert, attr_pert_square

    def PM_continuous_attr1(self, attr, Y_data):
        length = len(set(Y_data))
        self.num_data = len(attr)
        attr_square = np.square(attr)

        attr_pert = np.zeros((self.num_data, length))
        attr_pert_square = np.zeros((self.num_data, length))

        for i in range(self.num_data):
            attr_pert[i][Y_data[i]] = attr[i]
            attr_pert_square[i][Y_data[i]] = attr_square[i]
            for j in range(length):
                attr_pert[i][j] = self.PM_one_dim(attr_pert[i][j], self.eps_con)
                attr_pert_square[i][j] = self.PM_one_dim(attr_pert_square[i][j], self.eps_con)
        # attr_pert_square = np.square(attr_pert)

        attr_test = np.zeros((self.num_data, length))
        attr_test_square = np.zeros((self.num_data, length))
        for i in range(self.num_data):
            attr_test[i][Y_data[i]] = attr[i]
            attr_test_square[i][Y_data[i]] = attr_square[i]

        # print("here")

        return attr_pert, attr_pert_square, attr_test, attr_test_square


    def Multi_PM_continuous(self, sample, label, length):
        rows = len(sample)
        sample_square = np.square(sample)

        sample_pert = np.zeros((rows, length))
        sample_pert_square = np.zeros((rows, length))

        # for i in range(self.num_data):

        sample_pert[:, label] = sample
        sample_pert_square[:, label] = sample_square
        for j in range(length):
            sample_pert[:, j] = self.PM_multi_dim(sample_pert[:, j])
            sample_pert_square[:, j] = self.PM_multi_dim(sample_pert_square[:, j])

        return sample_pert, sample_pert_square

    def Laplace_continuous_attr(self, attr, Y_data):
        length = len(set(Y_data))
        self.num_data = len(attr)
        attr_square = np.square(attr)

        attr_pert = np.zeros((self.num_data, length))
        attr_pert_square = np.zeros((self.num_data, length))

        for i in range(self.num_data):
            attr_pert[i][Y_data[i]] = attr[i]
            attr_pert_square[i][Y_data[i]] = attr_square[i]
            for j in range(length):
                attr_pert[i][j] = self.Laplace(attr_pert[i][j])
                attr_pert_square[i][j] = self.Laplace(attr_pert_square[i][j])
        # attr_pert_square = np.square(attr_pert)

        attr_test = np.zeros((self.num_data, length))
        attr_test_square = np.zeros((self.num_data, length))
        for i in range(self.num_data):
            attr_test[i][Y_data[i]] = attr[i]
            attr_test_square[i][Y_data[i]] = attr_square[i]

        return attr_pert, attr_pert_square

    def Laplace(self, value):
        delta = 2 / self.eps_con

        lap = np.random.laplace(0, delta)
        t_ = value + lap

        return t_

    def PM_one_dim(self, value, eps_con):
        """
        一维PM
        :param value: input value
        :return: perturbed value
        """
        # eps = 2
        C = (np.exp(eps_con / 2) + 1) / (np.exp(eps_con / 2) - 1)

        p = np.exp(eps_con / 2) / (np.exp(eps_con / 2) + 1)

        lt = value * (C + 1) / 2 - (C - 1) / 2
        rt = lt + C - 1
        # print('lt: {}, rt: {}'.format(lt, rt))
        x = np.random.uniform(0, 1)
        if x < p:
            t_ = np.random.uniform(lt, rt)
        else:
            t_1 = np.random.uniform(-C, lt)
            t_2 = np.random.uniform(rt, C)
            t_ = np.random.choice([t_1, t_2])

        return t_

    def PM_multi_dim(self, sample):
        """
        多维PM
        :param sample:
        :return:
        """
        # print("PM_multi_dim")
        sample_pert = np.zeros(sample.shape)
        d = len(sample)
        k = max(1, min(d, int(self.eps_con/2.5)))
        data = np.arange(0, d, 1)

        select_data = []
        # for i in range(k):
        #     item = np.random.choice(data)
        #     select_data.append(item)
        #     idx = np.where(data == item)
        #
        #     data = np.delete(data, item)
        select_data = random.sample(list(data), k)

        for value in select_data:
            sample_pert[value] = (d/k) * self.PM_one_dim(sample[value], self.eps_con/k)

        return sample_pert


class DataAggregator:
    def __init__(self, eps_dis = 0.5):
        self.num_class = 0
        self.num_data = 0
        self.num_of_all_attr = []       # 每个属性下，不同取值的个数
        self.prior_pert_proba = []      # 先验概率
        self.estimate_sum_class = []    # 每个类的估计次数
        self.attr_counts = []            # 离散属性，每个属性可能的取值数
        self.mean_class = []            # 每个属性下，在不同类中的属性均值
        self.variance_class = []        # 每个属性下，在不同类中的属性方差
        self.eps_dis = eps_dis
        self.p_dis = np.exp(eps_dis / 2) / (np.exp(eps_dis / 2) + 1)
        self.q_dis = 1 / (np.exp(eps_dis / 2) + 1)
        self.num_of_all_attr_orig = []       # 测试用
        # print("DataAggregator")
        # print("eps_dis: ", eps_dis)
        # print("self.p_dis: ", self.p_dis)
        # print("self.q_dis: ", self.q_dis)
        self.mean_class_lap = []
        self.variance_class_lap = []
        self.mean_class_multi = []
        self.variance_class_multi = []

        self.mean_test_class = []
        self.variance_test_class = []

        self.mean_test_square_class = []
        self.mean_square_class = []

    def calc_prior_proba(self, vec_pert_class):
        # 每个位的计数
        num_of_bits = sum(vec_pert_class).astype(int)
        self.num_data = len(vec_pert_class)
        self.num_class = vec_pert_class.shape[1]

        # 每个位的估计次数，就是不同值的次数
        # estimate_sum_class = []
        for item in num_of_bits:
            p_c = (item - self.num_data * self.q_dis) / (self.p_dis - self.q_dis)
            # laplace smoothing
            if p_c < 0:
                p_c = -p_c
            self.estimate_sum_class.append(p_c.astype(int))

        # prior_pert_proba = []

        for i in range(len(self.estimate_sum_class)):
            # self.prior_pert_proba.append((self.estimate_sum_class[i] + 1) / (sum(self.estimate_sum_class))+self.num_class)
            temp = self.estimate_sum_class[i] + 1
            temp2 = sum(self.estimate_sum_class)
            temp3 = temp/temp2
            self.prior_pert_proba.append(temp3)

        return self.prior_pert_proba

    def calc_num_of_attr(self, vec_pert_attr, attr_count, binary_vec):
        """
        计算每个属性下不同值的个数
        :param vec_pert_attr: 某一属性下，每个映射值的扰动向量组
        :param attr_count: 属性可能取值数
        :return:
        """
        num_of_bits = np.sum(vec_pert_attr, axis=0)
        num_data = len(vec_pert_attr)
        estimate_sum_attr = []
        num_of_bits_orig = np.sum(binary_vec, axis=0).astype(int)

        for item in num_of_bits:
            p_c = (item - num_data * self.q_dis) / (self.p_dis - self.q_dis)
            # laplace smoothing
            if p_c < 0:
                p_c = -p_c
            estimate_sum_attr.append(p_c.astype(int))

        self.num_of_all_attr.append(estimate_sum_attr)
        self.num_of_all_attr_orig.append(num_of_bits_orig)

        # self.num_of_all_attr.append(num_of_bits)

        self.attr_counts.append(attr_count)
        self.mean_class.append([])
        self.variance_class.append([])

    def classify_discrete(self, sample):
        """
        输入一个映射好的样本，在之前计算好的每个属性下不同取值的个数的列表中索引到对应取值的个数，除以类别个数，得到条件概率
        在乘类别先验概率，得到总的概率，最后选出概率最大的类别标签
        :param sample: mapped sample
        :return: 概率最大的标签
        """
        probabilities = []
        for label in range(self.num_class):
            proba_in_label = 1
            # 某一属性下不同属性值的个数类别、映射属性值、属性可能取值数
            for num_of_attr, attr_value, attr_count in zip(self.num_of_all_attr, sample[:-1], self.attr_counts):
                # 计算条件概率
                proba_in_label *= num_of_attr[attr_value]/(self.estimate_sum_class[label]+attr_count)
            proba_in_label *= self.prior_pert_proba[label]
            probabilities.append(proba_in_label)

        return probabilities.index(max(probabilities))

    def classify_mixed(self, sample, continuous_index):
        """

        :param sample:
        :param continuous_index:
        :return:
        """
        probabilities = []
        for label in range(self.num_class):
            proba_in_label = 1
            # sample_class = sample[-1].astype(int)
            for idx in range(len(sample)):
                attr_value = sample[idx]
                if idx in continuous_index:
                    # 连续属性
                    # PM
                    mean_in_class = self.mean_class[idx][label]
                    variance_in_class = self.variance_class[idx][label]

                    # Laplace
                    # mean_in_class = self.mean_class_lap[idx][label]
                    # variance_in_class = self.variance_class_lap[idx][label]
                    # temp = Gaussian_Distribution(attr_value, mean_in_class, variance_in_class)
                    proba_in_label *= Gaussian_Distribution(attr_value, mean_in_class, variance_in_class)
                else:
                    # 离散属性
                    attr_value = attr_value.astype(int)
                    mapped_value = attr_value * self.num_class + label + 1
                    num_of_attr = self.num_of_all_attr[idx]
                    proba_in_label *= num_of_attr[mapped_value] / (self.estimate_sum_class[label])

            proba_in_label *= self.prior_pert_proba[label]
            probabilities.append(proba_in_label)

        return probabilities.index(max(probabilities))

    def classify_mixed_lap(self, sample, continuous_index):
        """

        :param sample:
        :param continuous_index:
        :return:
        """
        probabilities = []
        for label in range(self.num_class):
            proba_in_label = 1
            # sample_class = sample[-1].astype(int)
            for idx in range(len(sample)):
                attr_value = sample[idx]
                if idx in continuous_index:
                    # 连续属性
                    # PM
                    # mean_in_class = self.mean_class[idx][label]
                    # variance_in_class = self.variance_class[idx][label]

                    # Laplace
                    mean_in_class = self.mean_class_lap[idx][label]
                    variance_in_class = self.variance_class_lap[idx][label]
                    # temp = Gaussian_Distribution(attr_value, mean_in_class, variance_in_class)
                    proba_in_label *= Gaussian_Distribution(attr_value, mean_in_class, variance_in_class)
                else:
                    # 离散属性
                    attr_value = attr_value.astype(int)
                    mapped_value = attr_value * self.num_class + label + 1
                    num_of_attr = self.num_of_all_attr[idx]
                    proba_in_label *= num_of_attr[mapped_value] / (self.estimate_sum_class[label])

            proba_in_label *= self.prior_pert_proba[label]
            probabilities.append(proba_in_label)

        return probabilities.index(max(probabilities))



    def calc_mean_variance_lap(self, attr_pert, attr_pert_square):
        mean = np.mean(attr_pert, axis=0)
        mean_class = [mean_ / p_class for mean_, p_class in zip(mean, self.prior_pert_proba)]

        mean_square = np.mean(attr_pert_square, axis=0)
        mean_square_class = [mean_square_ / p_class for mean_square_, p_class in zip(mean_square, self.prior_pert_proba)]

        variance_class = [squared_mean - np.square(mean_in_class)
                          for squared_mean, mean_in_class in zip(mean_square_class, mean_class)]

        for i in range(len(variance_class)):
            if variance_class[i] < 0:
                variance_class[i] = -variance_class[i]

        self.mean_class_lap.append(mean_class)
        self.variance_class_lap.append(variance_class)
        # self.num_of_all_attr.append([])
        # self.attr_counts.append([])



    def calc_mean_variance(self, attr_pert, attr_pert_square, attr_test, attr_test_square):
        """

        :param attr_pert:
        :param attr_pert_square:
        :return: 每个属性中，不同类别的属性均值、方差的列表
                两个维度：第一维度数属性索引，第二维度是类别索引
        """
        mean = np.mean(attr_pert, axis=0)
        mean_class = [mean_ / p_class for mean_, p_class in zip(mean, self.prior_pert_proba)]

        mean_square = np.mean(attr_pert_square, axis=0)
        mean_square_class = [mean_square_ / p_class for mean_square_, p_class in zip(mean_square, self.prior_pert_proba)]

        variance_class = [squared_mean - np.square(mean_in_class)
                          for squared_mean, mean_in_class in zip(mean_square_class, mean_class)]

        mean_test = np.mean(attr_test, axis=0)
        mean_test_class = [mean_test_in_class / p_class for mean_test_in_class, p_class in zip(mean_test, self.prior_pert_proba)]

        mean_test_square = np.mean(attr_test_square, axis=0)
        mean_test_square_class = [mean_square_ / p_class for mean_square_, p_class in zip(mean_test_square, self.prior_pert_proba)]
        variance_test_class = [squared_mean - np.square(mean_in_class)
                          for squared_mean, mean_in_class in zip(mean_test_square_class, mean_test_class)]


        self.mean_test_class.append(mean_test_class)
        self.variance_test_class.append(variance_test_class)
        self.mean_test_square_class.append(mean_test_square_class)
        self.mean_square_class.append(mean_square_class)


        for i in range(len(variance_class)):
            if variance_class[i] < 0:
                variance_class[i] = -variance_class[i]

        self.mean_class.append(mean_class)
        self.variance_class.append(variance_class)
        self.num_of_all_attr.append([])
        self.attr_counts.append([])


    def calc_mean_variance_multi(self, sample_collect, sample_square_collect):
        labels = sample_collect.shape[1]
        mean = sample_collect / self.num_data
        mean_square = sample_square_collect / self.num_data


        for i in range(sample_collect.shape[0]):
            mean_class = [mean_ / p_class for mean_, p_class in zip(mean[i, :], self.prior_pert_proba)]
            mean_square_class = [mean_square_ / p_class for mean_square_, p_class in
                                 zip(mean_square[i, :], self.prior_pert_proba)]
            variance_class = [squared_mean - np.square(mean_in_class)
                          for squared_mean, mean_in_class in zip(mean_square_class, mean_class)]

            for i in range(len(variance_class)):
                if variance_class[i] < 0:
                    variance_class[i] = -variance_class[i]

            self.mean_class_multi.append(mean_class)
            self.variance_class_multi.append(variance_class)

    def classify_continuous_multi(self, sample):

        probabilities = []
        for label in range(self.num_class):
            proba_in_label = 1
            for idx in range(len(sample)):
                attr_value = sample[idx]
                mean_in_class = self.mean_class_multi[idx][label]
                variance_in_class = self.variance_class_multi[idx][label]
                proba_in_label *= Gaussian_Distribution(attr_value, mean_in_class, variance_in_class)
            proba_in_label *= self.prior_pert_proba[label]
            probabilities.append(proba_in_label)

        return probabilities.index(max(probabilities))


class NaiveBayesLDP:
    def __init__(self, eps_dis, eps_con):
        self.individuals = Individuals(eps_dis=eps_dis, eps_con=eps_con)
        self.dataAggregator = DataAggregator(eps_dis=eps_dis)


    def fit_multi_continuous(self, X_train, Y_train):
        # 先验概率计算
        # vec_pert_class = self.individuals.SUE_for_class(Y_data=Y_train)
        # prior_pert_proba = self.dataAggregator.calc_prior_proba(vec_pert_class=vec_pert_class)
        rows = X_train.shape[1]
        length = len(set(Y_train))

        sample_collect = np.zeros((rows, length))
        sample_square_collect = np.zeros((rows, length))

        for i in range(X_train.shape[0]):
            sample = X_train[i, :]
            sample_pert, sample_pert_square = self.individuals.Multi_PM_continuous(sample=sample, label=Y_train[i], length=length)
            sample_collect += sample_pert
            sample_square_collect += sample_pert_square

        self.dataAggregator.calc_mean_variance_multi(sample_collect, sample_square_collect)


    def fit(self, X_train, Y_train, continuous_index):

        # 先验概率计算
        vec_pert_class = self.individuals.SUE_for_class(Y_data=Y_train)
        prior_pert_proba = self.dataAggregator.calc_prior_proba(vec_pert_class=vec_pert_class)

        # 条件概率计算
        attr_num = X_train.shape[1]
        for i in range(attr_num):
            attr = X_train[:, i]

            if i in continuous_index:
                # 连续属性
                # if i == 3:
                #     print("here")
                # attr = normalize(attr)
                attr_pert, attr_pert_square, attr_test, attr_test_square = self.individuals.PM_continuous_attr1(attr, Y_train)
                self.dataAggregator.calc_mean_variance(attr_pert=attr_pert, attr_pert_square=attr_pert_square,
                                                       attr_test=attr_test, attr_test_square=attr_test_square)

                # laplace
                attr_pert, attr_pert_square = self.individuals.Laplace_continuous_attr(attr, Y_train)
                self.dataAggregator.calc_mean_variance_lap(attr_pert=attr_pert, attr_pert_square=attr_pert_square)

            else:
                # if i == 3:
                #     print("here")
                # 离散属性
                attr = attr.astype(int)
                attr_count = len(set(attr))  # 某一属性可能的取值数
                vec_pert_attr, attr_count, binary_vec = self.individuals.SUE_map_discrete_attr(attr=attr, Y_data=Y_train)
                self.dataAggregator.calc_num_of_attr(vec_pert_attr=vec_pert_attr, attr_count=attr_count, binary_vec=binary_vec)

    def predict(self, X_test, Y_test, continuous_index):
        pred_result = np.zeros(Y_test.shape)
        pred_result_lap = np.zeros(Y_test.shape)
        pred_result_multi = np.zeros(Y_test.shape)
        acc = 0
        acc_lap = 0
        acc_multi = 0

        for i in range(X_test.shape[0]):
            sample = X_test[i, :]
            pred_result[i] = self.dataAggregator.classify_mixed(sample=sample, continuous_index=continuous_index)
            # pred_result_lap[i] = self.dataAggregator.classify_mixed_lap(sample=sample, continuous_index=continuous_index)

            # for multi_PM
            # pred_result_multi[i] = self.dataAggregator.classify_continuous_multi(sample=sample)

            if pred_result[i] == Y_test[i]:
                acc += 1
            #
            if pred_result_multi[i] == Y_test[i]:
                acc_multi += 1

        TP = len([x for x, y in zip(Y_test, pred_result) if x == 1 and y == 1])
        FP = len([x for x, y in zip(Y_test, pred_result) if x == 0 and y == 1])
        TN = len([x for x, y in zip(Y_test, pred_result) if x == 1 and y == 0])
        FN = len([x for x, y in zip(Y_test, pred_result) if x == 0 and y == 0])

        # precision = TP / (TP + FP + 1)
        precision = precision_score(Y_test, pred_result, average='macro')
        # accuracy_test = (TP + TN) / (TP + FP + TN + FN + 1)
        # recall = TP / (TP + FN + 1)
        recall = recall_score(Y_test, pred_result, average='macro')
        accuracy = acc / Y_test.shape[0]
        accuracy_lap = acc_lap / Y_test.shape[0]
        accuracy_multi = acc_multi / Y_test.shape[0]
        print("accuracy: {:.2f}%".format(float(accuracy)*100))
        print("precision: {:.2f}%".format(float(precision)*100))
        print("recall: {:.2f}%".format(float(recall)*100))

        # print("accuracy_multi: {:.2f}%".format(float(accuracy_multi) * 100))
        return accuracy, accuracy_multi, precision, recall


def Bayes_Model(X_train, Y_train, X_test, Y_test):
    # 建立朴素贝叶斯分类器模型
    gaussianNB = CategoricalNB()
    gaussianNB.fit(X_train, Y_train)

    # 2 用交叉验证来检验模型的准确性，只是在test set上验证准确性
    # num_validations = 5
    #
    # accuracy = cross_val_score(gaussianNB, X_test, Y_test,
    #                            scoring='accuracy', cv=num_validations)
    #
    # print('准确率：{:.2f}%'.format(accuracy.mean() * 100))
    # precision = cross_val_score(gaussianNB, X_test, Y_test,
    #                             scoring='precision_weighted', cv=num_validations)
    # print('精确度：{:.2f}%'.format(precision.mean() * 100))
    # recall = cross_val_score(gaussianNB, X_test, Y_test,
    #                          scoring='recall_weighted', cv=num_validations)
    # print('召回率：{:.2f}%'.format(recall.mean() * 100))
    # f1 = cross_val_score(gaussianNB, X_test, Y_test,
    #                      scoring='f1_weighted', cv=num_validations)
    # print('F1  值：{:.2f}%'.format(f1.mean() * 100))

    # 3 打印性能报告
    acc = 0
    y_pred = gaussianNB.predict(X_test)
    for y, label in zip(y_pred, Y_test):
        if y == label:
            acc+=1

    TP = len([x for x, y in zip(Y_test, y_pred) if x == 1 and y == 1])
    FP = len([x for x, y in zip(Y_test, y_pred) if x == 0 and y == 1])
    TN = len([x for x, y in zip(Y_test, y_pred) if x == 1 and y == 0])
    FN = len([x for x, y in zip(Y_test, y_pred) if x == 0 and y == 0])

    # precision = TP / (TP + FP)
    precision = precision_score(Y_test, y_pred, average='macro')
    # accuracy_test = (TP + TN) / (TP + FP + TN + FN)
    # recall = TP / (TP + FN)
    recall = recall_score(Y_test, y_pred, average='macro')

    accuracy = acc / len(y_pred)
    print("baseline accuracy: {:.2f}%".format(float(accuracy) * 100))
    print("baseline precision: {:.2f}%".format(float(precision) * 100))
    print("baseline recall: {:.2f}%".format(float(recall) * 100))

    print('\n')
    # confusion_mat = confusion_matrix(Y_test, y_pred)
    # print(confusion_mat)  # 混淆矩阵
    return accuracy, precision, recall

    # 直接使用sklearn打印精度，召回率和F1值
    # target_names = ['0', '1']
    # print(classification_report(Y_test, y_pred, target_names=target_names))
# if __name__ == '__main__':
#     naiveBayesLDP = NaiveBayesLDP(eps_dis=eps_dis, eps_con=eps_con)
#     train_data_df, test_data_df = get_data()
#     X_train, Y_train, train_data_encode, continuous_index, discrete_index = process_data(train_data_df)
#     X_test, Y_test, test_data_encode, _, _ = process_data(test_data_df)
#     naiveBayesLDP.fit(X_train=X_train, Y_train=Y_train, continuous_index=continuous_index)
#     naiveBayesLDP.predict(X_test=X_test, Y_test=Y_test, continuous_index=continuous_index)
