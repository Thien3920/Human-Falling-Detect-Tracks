import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter

Path = '../Data/test.csv'
class_names = ['Standing', 'Walking', 'Sitting', 'Lying Down', 'Stand up', 'Sit down', 'Fall Down']
class_num = np.zeros(7)

annot = pd.read_csv(Path)
video = annot['0'].unique()
for vid in video:
    print('Process on: ' + vid)
    labels = annot[annot['0'] == vid].drop(columns='0').reset_index(drop=True)
    labels = np.array(labels)
    labels = labels.argmax(axis=1)
    s = Counter(labels)
    for p in s:
        class_num[p] += s[p]

idx_class = range(7)
plt.bar(idx_class, class_num)
plt.xticks(idx_class, class_names)

for x, y in zip(idx_class, class_num):
    plt.text(x+0.02, y+0.05, '%d' % y, ha='center', va='bottom')

plt.show()