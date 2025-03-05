import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import numpy as np

# 示例数据
data = {
    'category': ['cat', 'dog', 'bird', 'cat', 'dog'],
    'value': [1, 2, 3, 4, 5]
}
df = pd.DataFrame(data)

# 特征和目标分离
X = df[['category', 'value']]
y = None  # 这里没有目标变量，仅为示例

# 创建预处理步骤：对类别数据进行独热编码，对数值数据进行标准化
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), ['value']),  # 对value列进行标准化
        ('cat', OneHotEncoder(), ['category'])  # 对category列进行独热编码
    ]
)

test_data = preprocessor.fit_transform(X)

# 创建PCA模型
pca = PCA(n_components=None)  # 例如，我们只想要一个主成分

# 创建管道，先进行预处理，然后应用PCA
pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('pca', pca)])

# 拟合和转换数据
pipeline.fit(X)
X_transformed = pipeline.transform(X)
print(X_transformed)
print('保留的主成分个数：', pca.n_components_)
print('保留的特征向量：\n', pca.components_)
print('保留的n个主成分各自方差：\n', pca.explained_variance_)

print('保留的n个主成分对原始数据信息累计解释的贡献率：\n', np.cumsum(pca.explained_variance_ratio_))
print('here')