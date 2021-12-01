import shutil
from glob import glob


video_folder_source = '/home/minhhuy/Downloads/Home_01/Videos'
video_folder_destination = '/home/minhhuy/Desktop/Python/Human-Falling-Detect-Tracks/Data/falldata/Home/Video'
annotation_folder_source = '/home/minhhuy/Downloads/Home_01/Annotation_files'
annotation_folder_destination = '/home/minhhuy/Desktop/Python/Human-Falling-Detect-Tracks/Data/falldata/Home/Annotation_files'

def getStart():
    max = None
    pathVideoDes = glob(video_folder_destination + '/*')
    for i in range(len(pathVideoDes)):
        idx = pathVideoDes[i].split('/')[-1].split('.')[0].split('(')[-1].split(')')[0]
        if max == None or max < idx:
            max = idx
    return max

pathVideo = glob(video_folder_source + '/*')
pathVideo.sort()
pathAnnot = glob(annotation_folder_source + '/*')
pathAnnot.sort()

if getStart() == None:
    start = 1
else:
    start = getStart()

for i in range(len(pathVideo)):
    codec = pathVideo[i].split('/')[-1].split('.')[-1]
    newNameVideo = video_folder_destination + '/' + 'video' + '({})'.format(start) + '.{}'.format(codec)
    newNameAnnot = annotation_folder_destination + '/' + 'video' + '({})'.format(start) + '.txt'
    shutil.copy(pathVideo[i], newNameVideo)
    shutil.copy(pathAnnot[i], newNameAnnot)
    start += 1
