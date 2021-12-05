import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
# from collections import Counter

Path = '../Data/AllStep2.csv'
thread = 20

main_parts = ['LShoulder_x', 'LShoulder_y', 'RShoulder_x', 'RShoulder_y', 'LHip_x', 'LHip_y',
              'RHip_x', 'RHip_y']
class_names = ['Standing', 'Walking', 'Sitting', 'Lying Down', 'Stand up', 'Sit down', 'Fall Down']
class_num = np.zeros(7)

annot = pd.read_csv(Path)
idx = annot.iloc[:, 2:-1][main_parts].isna().sum(1) > 0
idx = np.where(idx)[0]
annot = annot.drop(idx)

def countLabel(labels):
    curClass = labels[0]
    result = dict()
    result[curClass] = 0
    for i in labels:
        if curClass == i:
            result[curClass] += 1
        else:
            # print(str(curClass) + '   ' + str(result[curClass]))
            if result[curClass] < thread:
                result[curClass] = 0
            else:
                class_num[curClass] += 1
                result[curClass] = 0
            curClass = i
            result[curClass] = 1
    # print(str(curClass) + '   ' + str(result[curClass]))
    if result[curClass] >= thread:
        class_num[curClass] += 1


video = annot['video'].unique()
for vid in video:
    print('Process on: ' + vid)
    labels = annot[annot['video'] == vid]['label'].reset_index(drop=True)
    countLabel(labels)
    # for i in labels:
    #     if labels[i] >= thread:
    #         class_num[i] += 1
    # print(Counter(labels))
    print(class_num)
idx_class = range(7)
plt.bar(idx_class, class_num)
plt.xticks(idx_class, class_names)

for x, y in zip(idx_class, class_num):
    plt.text(x+0.02, y+0.05, '%d' % y, ha='center', va='bottom')

plt.show()
