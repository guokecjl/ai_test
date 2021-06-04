# coding: utf-8
"""
通过svd智能推荐电影
"""

import pandas as pd
import numpy as np

rating_info = pd.read_table('data/ratings.txt')
rating_info = rating_info.pivot(index='UserId', columns='MovieId',values=['Rating'])
rating_info = rating_info.fillna(0)

data = rating_info.to_numpy()
v, s, u = np.linalg.svd(data, 500)
print(s.shape)