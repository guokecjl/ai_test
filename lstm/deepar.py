# coding: utf-8
'''
deepar算法实现,依赖包　mxnet mxnet-mkl gluon gluonts,pathlib
'''

import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from gluonts.model import deepar
from gluonts.dataset import common
from gluonts.dataset.util import to_pandas
from gluonts.model.predictor import Predictor

# 数据加载
df = pd.read_csv('/home/zhouxi/pig.csv', header=0, index_col=0)
data = common.ListDataset([{"start": df.index[100],
   "target": df.price[:"2018-12-05 00:00:00"]}], freq="D")
train_data = data

# 模型训练
estimator = deepar.DeepAREstimator(freq="D", prediction_length=10)
predictor = estimator.train(training_data=data)

# 模型预测和绘图
for test_entry, forecast in zip(train_data, predictor.predict(train_data)):
    to_pandas(test_entry)[-60:].plot(linewidth=2)
    forecast.plot(color='g', prediction_intervals=[50.0, 90.0])
plt.grid(which='both')
plt.show()

prediction = next(predictor.predict(train_data))
print(prediction.mean)
prediction.plot(output_file='graph.png')

# 模型存储
predictor.serialize(Path("/home/zhouxi/my_product/ai_test/lstm/model/"))

# 模型加载
predictor = Predictor.deserialize(Path("/home/zhouxi/my_product/ai_test/lstm/model/"))

# 模型预测
prediction = next(predictor.predict(train_data))
print(prediction.mean.tolist())
prediction.plot(output_file='graph.png')
