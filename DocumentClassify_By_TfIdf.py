import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 读取数据
data = pd.read_csv('./abcnews-date-text(1).csv',error_bad_lines=False,usecols=['headline_text'])
data.head()
data = data.head(10000)

# print(data.info())

# 删除重复数据
a = data['headline_text'].duplicated(keep=False)
print(data[a])
#print(data[data['headline_text'].duplicated(keep=False)])