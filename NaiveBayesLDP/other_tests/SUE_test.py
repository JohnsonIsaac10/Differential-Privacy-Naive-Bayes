import numpy as np
from intervals import IntInterval
import intervals
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

num_k = 2
df = pd.read_csv('./data/weatherAUS.csv')
data1 = df[['Rainfall', 'RainToday']].dropna().values
num = data1.shape[0]
print(data1)
test_data = data1[:, 0]
test_label = data1[:, 1]

yes_label_idx = np.where(test_label == 'Yes')
no_label_idx = np.where(test_label == 'No')

M_class = np.zeros((num, num_k))

M_class_pert = np.zeros((num, num_k))


idx = yes_idx = no_idx = 0
num_yes = yes_label_idx[0].shape[0]  # 正例个数
num_no = no_label_idx[0].shape[0]   # 反例个数

# SUE编号，yes = 0，no = 1
for idx in range(num):
    if yes_idx < yes_label_idx[0].shape[0] and idx == yes_label_idx[0][yes_idx]:
        M_class[idx][0] = 1
        M_class_pert[idx][0] = 1
        yes_idx += 1
    elif no_idx < no_label_idx[0].shape[0] and idx == no_label_idx[0][no_idx]:
        no_idx += 1
        M_class_pert[idx][1] = 1
        M_class[idx][1] = 1



eps = 0.5
p = np.exp(eps/2)/(np.exp(eps/2) + 1)
q = 1 / (np.exp(eps/2) + 1)

# 添加扰动
# for i in range(len(M_class_pert)):
#     for j in range(2):
#         # print(M_c[i][j])
#         if M_class_pert[i][j] == 1:
#             x = np.random.rand()
#             if x < q:
#                 M_class_pert[i][j] = 0
#
#         elif M_class_pert[i][j] == 0:
#             x = np.random.rand()
#             if x < q:
#                 M_class_pert[i][j] = 1

# 添加扰动
for i in range(len(M_class_pert)):
    for j in range(2):
        x = np.random.rand()
        if x < q:
            if M_class_pert[i][j] == 1:
                M_class_pert[i][j] = 0
            elif M_class_pert[i][j] == 0:
                M_class_pert[i][j] = 1


M_sum_pert = []   # 扰动向量统计

for j in range(2):
    sum_c = 0
    for i in range(len(M_class_pert)):
        sum_c += M_class_pert[i][j]
    M_sum_pert.append(sum_c)

print('扰动向量中对应位统计', M_sum_pert)


M_sum = []
for j in range(2):
    sum_c = 0
    for i in range(len(M_class)):
        sum_c += M_class[i][j]
    M_sum.append(sum_c)

print('原始向量中对应位统计', M_sum)

num_c_esti = []

for item in M_sum_pert:
    p_c = (item - num*q)/(p-q)
    num_c_esti.append(p_c)
    print("估计：",p_c)

for item in num_c_esti:
    print(item/sum(num_c_esti))

print(num_yes/num)
print(num_no/num)



# M = [[0, 0, 0, 0, 1, 0],
#      [0, 1, 0, 0, 0, 0],
#      [0, 0, 0, 1, 0, 0],
#      [1, 0, 0, 0, 0, 0],
#      [0, 0, 0, 0, 0, 1],
#      [0, 1, 0, 0, 0, 0]]
#
# M_c = [[1, 0],
#        [0, 1],
#        [0, 1],
#        [1, 0],
#        [0, 1],
#        [0, 1]]
#
#
#
#
#
# M_c_pert = M_c
# # print(M_c)
# # print(M_c_pert)
#
#
# for i in range(len(M_c)):
#     for j in range(2):
#         x = np.random.uniform(0, 1)
#         # print(M_c[i][j])
#         if M_c[i][j] == 1:
#             if x > p:
#                 M_c[i][j] = 0
#         elif M_c[i][j] == 0:
#             if x < q:
#                 M_c[i][j] = 1
#
# # print(M_c)
#
# M_sum = []
#
# for j in range(2):
#     sum_c = 0
#     for i in range(len(M_c)):
#         sum_c += M_c[i][j]
#     M_sum.append(sum_c)
#
# # print(M_sum)
#
# for item in M_sum:
#     p_c = (item - 2*q)/(p-q)
#     print(int(p_c))
#


# print(M_c_pert)

