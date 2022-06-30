import cv2
import numpy as np
import glob
import os

"""
This script is merely for covering the animal in the frames
"""


def getFrames(path='/data/develop/extract_mkv_k4a/output/26_04'):
    cameras = sorted(os.listdir('/data/develop/extract_mkv_k4a/output/26_04'))
    res = []
    for camera in cameras:
        path_to_cam = os.path.join(path, camera)
        res.append(glob.glob(path_to_cam + '/*.jpg'))

        os.makedirs(os.path.join(path+'_covered', camera), exist_ok=True)

    return res

def coverImages(frames, path='/data/develop/extract_mkv_k4a/output/26_04_covered/'):
    positions = {
        # x, y bottom left
        # x, y top right
        # -> top left = 0, 3
        # -> bottom right = 2, 1
        0 : [846, 970, 1311, 766],
        1 : [835, 1070, 1380, 464],
        2 : [819, 1044, 1247, 781],
        3 : [874, 1233, 1147, 869]
    }

    print(frames)

    for i, camera in enumerate(frames):
        for frame in camera:
            pos = positions[i]
            img = cv2.imread(frame)
            cv2.rectangle(img, (pos[0], pos[3]), (pos[2], pos[1]), (0, 0, 0), -1)
            base_name = os.path.basename(frame)
            path_to_save = os.path.join(path, 'cn0'+str(i + 1), base_name)
            cv2.imwrite(path_to_save, img)


def main():
    frames = getFrames()
    coverImages(frames=frames)

if __name__=='__main__':
    main()





