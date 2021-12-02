"""
This script to create .csv videos frames action annotation file.

- It will play a video frame by frame control the flow by [a] and [d]
 to play previos or next frame.
- Open the annot_file (.csv) and label each frame of video with number
 of action class.
"""
#unset SESSION_MANAGER
import os
import cv2
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#class_names = ['Walking', 'Fall Down']  # label.
class_names = ['Standing', 'Walking', 'Sitting', 'Lying Down','Stand up', 'Sit down', 'Fall Down']

video_folder = '/home/minhhuy/Desktop/Python/Human-Falling-Detect-Tracks/Data/falldata/Home/Videos'
annot_file = '../Data/Home_new.csv'
annot_file_2 = '../Data/Home_new_2.csv'


def create_csv(folder):
    list_file = sorted(os.listdir(folder))
    cols = ['video', 'frame', 'label']
    df = pd.DataFrame(columns=cols)
    for fil in list_file:
        cap = cv2.VideoCapture(os.path.join(folder, fil))
        frames_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video = np.array([fil] * frames_count)
        frame = np.arange(1, frames_count + 1)
        label = np.array([0] * frames_count)
        rows = np.stack([video, frame, label], axis=1)
        df = df.append(pd.DataFrame(rows, columns=cols),
                       ignore_index=True)
        cap.release()
    df.to_csv(annot_file, index=False)


if not os.path.exists(annot_file):
    create_csv(video_folder)

annot = pd.read_csv(annot_file)
video_list = annot.iloc[:, 0].unique()

cols = ['video', 'frame', 'label']
df = pd.DataFrame(columns=cols)

index_video_to_play = 0
backVideo = False
step = 0
while index_video_to_play < len(video_list):
    video_file = os.path.join(video_folder, video_list[index_video_to_play])
    print(os.path.basename(video_file))

    annot_2 = annot[annot['video'] == video_list[index_video_to_play]].reset_index(drop=True)
    frames_idx = annot_2.iloc[:, 1].tolist()

    cap = cv2.VideoCapture(video_file)
    frames_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    assert frames_count == len(frames_idx), 'frame count not equal! {} and {}'.format(
        len(frames_idx), frames_count)

    video = np.array([video_list[index_video_to_play]] * frames_count)
    frame_1 = np.arange(1, frames_count + 1)
    label = np.array([0] * frames_count)
    if backVideo is True and step == 1:
        step = 2
    k = 0
    i = 0
    ipre = -1
    while True:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if i > ipre and i > 0:
            label[i - 1] = k
            ipre = i
        if ret:
            cls_name = class_names[label[i - 1]]
            frame = cv2.resize(frame, (0, 0), fx=3.5, fy=3.5)
            frame = cv2.putText(frame, 'Videos: {}     Total_frames: {}        Frame: {}       Pose: {} '.format(
                video_list[index_video_to_play], frames_count, i + 1, cls_name),
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 0, 0), 2)

            frame = cv2.putText(frame, 'Back:  a', (10, 270), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            frame = cv2.putText(frame, 'Standing:   0', (10, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            frame = cv2.putText(frame, 'Walking:    1', (10, 330), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            frame = cv2.putText(frame, 'Sitting:    2', (10, 360), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            frame = cv2.putText(frame, 'Lying Down: 3', (10, 390), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            frame = cv2.putText(frame, 'Stand Up:   4', (10, 420), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            frame = cv2.putText(frame, 'Sit Down:   5', (10, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            frame = cv2.putText(frame, 'Fall Down:  6', (10, 480), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            cv2.imshow('frame', frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('0'):  # standing
                i += 1
                k = 0
                continue
            elif key == ord('1'):  # walking
                i += 1
                k = 1
                continue
            elif key == ord('2'):  # sitting
                i += 1
                k = 2
            elif key == ord('3'):  # lying down
                i += 1
                k = 3
                continue
            elif key == ord('4'):  # stand up
                i += 1
                k = 4
            elif key == ord('5'):  # sit down
                i += 1
                k = 5
                continue
            elif key == ord('6'):  # fall down
                i += 1
                k = 6
                continue
            elif key == ord('a'):  # back
                i -= 1
                continue
            elif key == ord('b'):  # back video
                backVideo = True
                step = 1
                index_video_to_play -= 1
                break
        else:
            break
    if backVideo == False:
        rows = np.stack([video, frame_1, label], axis=1)
        df = df.append(pd.DataFrame(rows, columns=cols), ignore_index=True)
        print(len(df))
        index_video_to_play += 1
    elif backVideo is True and step == 2:
        rows = np.stack([video, frame_1, label], axis=1)
        df.iloc[len(df)-frames_count:len(df), :] = rows
        print(len(df))
        index_video_to_play += 1
        backVideo = False
        step = 0

df.to_csv(annot_file_2, index=False)
cap.release()
cv2.destroyAllWindows()
