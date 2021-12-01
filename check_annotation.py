import os
import cv2
import pandas as pd


video_folder = '/home/thien/Desktop/Human-Falling-Detect-Tracks/videos'
annot_folder = '/home/thien/Desktop/Human-Falling-Detect-Tracks/annot'  # bounding box annotation for each frame.


class_names = ['Standing', 'Walking', 'Sitting', 'Lying Down', 'Stand up', 'Sit down', 'Fall Down']
# with score.


vid_list = sorted(os.listdir(video_folder))
for vid in vid_list:

    cap = cv2.VideoCapture(os.path.join(video_folder, vid))
    frames_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                  int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    # Bounding Boxs Labels.
    annot_file_2 = os.path.join(annot_folder, vid.split('.')[0])
    annot_file_2 = annot_file_2+'.txt'
    annot_2 = []
    if os.path.exists(annot_file_2):
        annot_2 = pd.read_csv(annot_file_2, header=None,
                                  names=['frame_idx', 'class', 'xmin', 'ymin', 'xmax', 'ymax'])
        annot_2 = annot_2.dropna().reset_index(drop=True)

        if frames_count != len(annot_2):
            print(f'Process on: {vid}')
            print('frame count not equal! {} and {}'.format(frames_count, len(annot_2)))
    cap.release()
    cv2.destroyAllWindows()