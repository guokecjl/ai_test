from sklearn import datasets
from sklearn.model_selection import train_test_split
import xgboost as xgb
from matplotlib import pyplot
from sklearn import metrics
import pandas as pd
df = pd.read_csv('../data/xijing.csv')
feature_name = list(df.columns)[:-1]

data = df[feature_name].to_numpy()
# 标签
label = df.label.to_numpy()

# 提取训练集和测试集
train_x, test_x, train_y, test_y = train_test_split(data, label, test_size=0.8, random_state=0)

dtrain = xgb.DMatrix(train_x, label=train_y, feature_names=feature_name)
dtest = xgb.DMatrix(test_x, label=test_y, feature_names=feature_name)

params = {
    'max_depth': 20,
    'eta': 0.1,
    'silent': 1,
    'objective': 'multi:softmax',
    'num_class': 2
}

watchlist = [(dtrain,'train'), (dtest, 'eval')]
model = xgb.train(params,dtrain,num_boost_round=10, evals=watchlist)

xgb.plot_importance(model)
pyplot.show()

ypred=model.predict(dtest)

import numpy as np

from sklearn.metrics import f1_score, accuracy_score, recall_score, \
    precision_score, roc_curve, auc, confusion_matrix, classification_report,\
    precision_recall_fscore_support, roc_auc_score

def evaluate_2_classify(label, prediction, pred_score=None):
    """
    评估二分类模型
    :param label: 真实标签列
    :param prediction: 评估结果
    :param pred_score: 为正样本概率
    :return:
    """
    labels = list(set(label.tolist()))
    tmp_dict = {}
    tmp_dict['acc_score'] = accuracy_score(label, prediction)
    tmp_dict['precision_score'] = precision_score(label, prediction)
    tmp_dict['recall_score'] = recall_score(label, prediction)

    if pred_score:
        tmp_dict['roc_auc_score'] = roc_auc_score(labels, pred_score)
        fpr, tpr, threshold = roc_curve(label, pred_score)
        tmp_dict['roc_curve'] = {'x': fpr.tolist(), 'y': tpr.tolist()}

        tmp_dict['ks_score'] = max(tpr - fpr)
        tmp_dict['ks_curve'] = {'x': threshold.tolist(), 'y': (tpr - fpr).tolist()}

    tmp_dict['f1_score'] = f1_score(label, prediction)
    tmp_dict['confusion_matrix'] = confusion_matrix(label, prediction,
                                                    labels=labels).tolist()
    return tmp_dict

import torch
print(evaluate_2_classify(test_y, torch.tensor(ypred).int()))
