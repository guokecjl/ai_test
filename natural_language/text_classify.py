'''
文本分类算法
'''

import fasttext

train_data = '/home/zhouxi/text_classify.csv'

classifier = fasttext.train_supervised(
    input = train_data,
    label = 'cat',
    dim = 256,
    epoch = 50,
    lr = 1,
    lr_update_rate = 50,
    min_count = 3,
    loss = 'softmax',
    word_ngrams = 2,
    bucket = 1000000
)
