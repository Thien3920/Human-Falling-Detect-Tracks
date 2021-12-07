"""Copy all files from step 2 to new file"""
import numpy as np
import pandas as pd
from glob import glob
import os

Path = ['../Data/Coffee_room_01_2_pose_and_score.csv',
        '../Data/Coffee_room_02_2_pose_and_score.csv',
        '../Data/Home_01_2_pose_and_score.csv',
        '../Data/Home_02_2_pose_and_score.csv',
        '../Data/Lecture_room_pose_and_score.csv',
        '../Data/Office_pose_and_score.csv']
# Path = '../Data/*.csv'
# allfile = glob(Path)
savePath = '../Data/AllStep2.csv'

columns = ['video', 'frame', 'Nose_x', 'Nose_y', 'Nose_s', 'LShoulder_x', 'LShoulder_y', 'LShoulder_s',
           'RShoulder_x', 'RShoulder_y', 'RShoulder_s', 'LElbow_x', 'LElbow_y', 'LElbow_s', 'RElbow_x',
           'RElbow_y', 'RElbow_s', 'LWrist_x', 'LWrist_y', 'LWrist_s', 'RWrist_x', 'RWrist_y', 'RWrist_s',
           'LHip_x', 'LHip_y', 'LHip_s', 'RHip_x', 'RHip_y', 'RHip_s', 'LKnee_x', 'LKnee_y', 'LKnee_s',
           'RKnee_x', 'RKnee_y', 'RKnee_s', 'LAnkle_x', 'LAnkle_y', 'LAnkle_s', 'RAnkle_x', 'RAnkle_y',
           'RAnkle_s', 'label']

df = pd.DataFrame(columns=columns)
start = 1
currentVid = None
for path in Path:
    print('Processing on: ' + os.path.basename(path))
    df2 = pd.read_csv(path)
    # df2 = df2.values()
    video = df2['video']
    setVid = video.unique()
    video = np.array(video)
    for i in setVid:
        video = np.where(video == i, 'video'+str(start), video)
        start += 1
    df2['video'] = video
    df = df.append(df2, ignore_index=True)
df.to_csv(savePath, mode='w', index=False)
