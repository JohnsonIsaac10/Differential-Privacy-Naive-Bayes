import numpy as np
from intervals import IntInterval
import intervals
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def PM(num, data):
    eps = 2.5
    C = (np.exp(eps / 2) + 1) / (np.exp(eps / 2) - 1)

    p = np.exp(eps / 2) / (np.exp(eps / 2) + 1)

    sum1 = 0
    sum2 = 0
    for t in data:

        # t = np.random.uniform(-1, 1)
        # t = 0.5
        lt = t * (C + 1) / 2 - (C - 1) / 2
        rt = lt + C - 1
        # print('lt: {}, rt: {}'.format(lt, rt))
        x = np.random.uniform(0, 1)
        if x < p:
            t_ = np.random.uniform(lt, rt)
        else:
            t_1 = np.random.uniform(-C, lt)
            t_2 = np.random.uniform(rt, C)
            t_ = np.random.choice([t_1, t_2])
            # if np.random.random() > 0.5:
            #     t_ = t_1
            # else:t_ = t_2
        sum1 += t_
        # print('t_: {}，t: {}'.format(t_, t))
        sum2 += t
    return sum1/num, sum2/num


def normalize(x):
    mean = np.mean(x)
    Xmax = np.max(test_data)
    Xmin = np.min(test_data)
    return 2 * (x - Xmin) / (Xmax - Xmin) - 1



df = pd.read_csv('./data/weatherAUS.csv')

full_data = df.values
# print(df['MaxTemp'].values[0:250])

data1 = df[['Rainfall', 'RainTomorrow']].dropna().values
num = data1.shape[0]
print(data1)
test_data = data1[:, 0]
test_label = data1[:, 1]

yes_label_idx = np.where(test_label == 'Yes')
no_label_idx = np.where(test_label == 'No')




yes_data = test_data[yes_label_idx]
num_yes = yes_data.shape[0]
P_yes = num_yes/num

no_data = test_data[no_label_idx]
num_no = no_data.shape[0]
P_no = num_no/num


zero_yes = np.zeros(no_data.shape[0])
# test_zeor_norm = normalize(test_zero)
zero_mean_pm, zero_mean_true = PM(num_no, zero_yes)

# yes_data_all = np.concatenate((yes_data, np.zeros(no_data.shape[0])), axis=0)


# test_data = [0.697, 0.774, 0.634, 0.608, 0.556, 0.403, 0.481, 0.437]
# test_data = np.random.randint(0, 500, num)
# test_data = np.zeros(num)

# Xmax = np.max(test_data)
# Xmin = np.min(test_data)
# Y = a + (b-a)/(Xmax-Xmin)*(test_data-Xmin)


Y = normalize(test_data)

yes_data_norm = normalize(yes_data)      # 正类数据标准化
no_data_norm = normalize(no_data)       # 反类数据标准化

# yes_data_norm_all = normalize(yes_data_all)
yes_data_all = np.concatenate((yes_data_norm, np.zeros(no_data.shape[0])), axis=0)
# yes_data_norm_all = normalize(yes_data_all)

print('mean of test data: ', sum(test_data)/num)
# print(Y)
print('mean of test data after normalizing: ', sum(Y)/num)
pm_test_data_mean, _ = PM(num=num, data=Y)
print('扰动后test data的均值：', pm_test_data_mean)
print('-------------------------------')


# new_data = np.zeros(num)

mean_pm, true_mean_pm = PM(num, data=Y)
# print('C:{}, -C:{}'.format(C, -C))

# print('estimate mean with PM: ', mean_pm)
# print('true mean: ', true_mean_pm)
# print('---------------------------')

yes_mean_pm_all, yes_mean_true_all = PM(num, data=yes_data_all)

yes_mean_pm, yes_mean_true = PM(num_yes, data=yes_data_norm)

# no_mean_pm, no_mean_true = PM(num, data=no_data_norm)

print('estimate mean of all yes data with PM: ', yes_mean_pm_all) # 添加噪声后，正类的数据总均值
print('true mean of all yes: ', yes_mean_true_all)  # 实际正类总均值，加了反类0数据
print('P_yes: ', P_yes)   # 正类比例
# print('calculate: ', mean_pm/P_yes)  # 计算正类均值
print('---------------------------')


print('estimate mean of yes data with PM: ', yes_mean_pm) # 添加噪声后，正类的数据均值
print('true mean of yes: ', yes_mean_true)  # 实际正类均值
print('P_yes: ', P_yes)   # 正类比例
print('calculate: ', yes_mean_pm_all/P_yes)  # 计算正类均值
print('---------------------------')

#
# test_zero = np.zeros(no_data.shape[0])
# # test_zeor_norm = normalize(test_zero)
# zero_mean_pm, zero_mean_true = PM(num_no, test_zero)


# print('zero_mean_pm: ', zero_mean_pm)
# print('zero_mean_true: ', zero_mean_true)


# print('estimate mean of no data with PM: ', no_mean_pm) # 添加噪声后，反类的数据均值
# print('true mean of no: ', no_mean_true)  # 实际反类均值
# print('P_no: ', P_no)   # 反类比例
# print('calculate: ', mean_pm/P_yes)  # 计算反类均值
# print('---------------------------')


def Laplace(num, data):
    eps = 2.5

    delta = 2/eps
    sum_temp1 = 0
    sum_temp2 = 0
    sum_lap = 0

    for t in data:
        # t = np.random.uniform(-1, 1)
        # t = 0.5
        lap = np.random.laplace(0, delta)
        # print(lap)
        sum_lap += lap
        t_ = t + lap
        # print('laplace noise: ',lap)
        # print('noisy data with laplace: ',t_)
        sum_temp1 += t_
        sum_temp2 += t
    # print('mean of lap: ', sum_lap/num)
    # print('mean of laplace noise: ', sum_lap/num)
    return sum_temp1/num, sum_temp2/num

mean_noise, mean_orig = Laplace(num, data=Y)
# print('estimate mean with laplace: ', mean_noise)
# print('true mean: ', mean_orig)
# print('---------------------')




# temp = np.random.uniform(-3,3)
# xx = intervals.FloatInterval.closed_open(-1, 2)
# yy = intervals.FloatInterval.open_closed(3, 5)
# x1 = np.array(xx)
# y1 = np.array(yy)
# print(x1)

