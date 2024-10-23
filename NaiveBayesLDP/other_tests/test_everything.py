import matplotlib
import numpy as np
import random
list_all = []
import matplotlib.pyplot as plt
list1 = [1,2,3,4,5,6]
list2 = [1,2,3,4,5]
list3 = [2,6,5,6,7,8]
list4 = [7,6,3,9,4]
list_all.append(list1)
list_all.append(list2)
list_all.append(list3)
list_all.append(list4)

plt.style.use(['science','ieee','no-latex'])
plt.grid(linestyle="--")
plt.xlabel("ε")
plt.ylabel("Accuracy")
plt.plot(list2, list4,label='test1')
plt.plot(list1,list3,label='test2')
plt.legend(loc='lower right')
plt.savefig('./test_fig/test1.png')
plt.show()



# idx = [0,1,2,2]
# test = []
# for list_, index in zip(list_all, idx):
#     print(list_[index])
#     test.append(list_[index])
#
# print(test.index(max(test)))


test = np.array([[1,2,3,4],
                [5,5,6,7],
                 [1,3,4,5]])

print(test/10)

list_ = [1,2,3,4,5,6,7,8,9]

print(random.sample(list_, 2))