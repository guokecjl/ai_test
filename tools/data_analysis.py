# coding: utf-8

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def show_miss_value(df):
    """
    输出对空值的统计
    """
    variable_name = []
    total_value = []
    total_miss_value = []
    missing_value_rate = []
    unique_value_list = []
    total_unique_value = []
    data_type = []
    for col in df.columns:
        variable_name.append(col)
        data_type.append(df[col].dtype)
        total_value.append(df[col].size)
        total_miss_value.append(df[col].isnull().sum())
        missing_value_rate.append(
            round(total_miss_value[-1] / total_value[-1], 3))
        unique_value_list.append(df[col].unique())
        total_unique_value.append(len(df[col].unique()))
    missing_data = pd.DataFrame({
        'variable_name': variable_name,
        'total_value': total_value,
        'total_miss_value': total_miss_value,
        'missing_value_rate': missing_value_rate,
        'unique_value_list': unique_value_list,
        'total_unique_value': total_unique_value,
        'data_type': data_type
    })
    return missing_data.sort_values('missing_value_rate', ascending=False)


def caculate_corr(df):
    df_corr = df.corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    mask = np.triu(np.ones_like(df_corr, dtype=np.bool))
    sns.heatmap(df_corr, mask=mask, annot=True, fmt=".2f",
                cmap=sns.diverging_palette(240, 10, n=19),
                vmin=-1, vmax=1, cbar_kws={"shrink": .8})
    plt.show()
