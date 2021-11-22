import ffmpeg

import os
from glob import glob

import cv2

PATH = "/media/gtzelepis/DATA/real_data/more_videos/videos/white_large_videos/one_hand_side/"
image_name = "white_large_one_hand_side_%d.png"
# EXT = "*.mp4"
# img_EXT = "*.png"

video_count = 100000

for path, subdir, files in os.walk(PATH):
        for file in files:
            video_path = os.path.join(PATH, file)
            # print(video_path)
            # ffmpeg.input(video_path).filter('fps', fps='1/60').output('PATH/test-%d.png',
            #                                                    start_number=0).overwrite_output().run(quiet=True)
            vidcap = cv2.VideoCapture(video_path)
            success, image = vidcap.read()
            count = video_count
            while success:
                cv2.imwrite(PATH + image_name % count, image)
                success, image = vidcap.read()
                print('Read a new frame: ', success)
                count += 1
            video_count += 100000

