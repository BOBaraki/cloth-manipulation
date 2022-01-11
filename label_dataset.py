import os
from glob import glob

import pandas as pd
import numpy as np

import csv

import pdb

PATH = "/media/gtzelepis/DATA/real_data/more_videos/videos/white_medium/two_hand_middle/"
EXT = "*.csv"
img_EXT = "*.png"
fn = 'white_medium_two_hand_middle_'

header = ['filename', 'cloth_state', 'false_labeling']

# cloth_state = ['flat', 'semi-lifted-onehand', 'diagonally Folded']
# cloth_state = ['flat', 'semi-lifted-twohands', 'folded']
# cloth_state = ['flat', 'semi-lifted-onehand', 'folded']
cloth_state = ['flat', 'semi-lifted-twohands-middle']


flat = True
semi = False
folded = False


# def stateidentification(filename, flat, semi, folded):
#     if filename == fn+'100001' or filename == fn+'100119':
#         flat = True
#         semi = False
#         folded = False
#         # lifted = False
#     elif name == fn+'100026' or name == fn+'200038':
#         flat = False
#         semi = True
#         folded = False
#         # lifted = False
#     # elif name == 'one_hand_lifting_100089' or name == 'one_hand_lifting_200098':
#     #     flat = False
#     #     semi = False
#     #     semi_crampled = True
#     #     lifted = False
#     elif name == fn+'100074' or name == fn+'200092':
#         flat = False
#         semi = False
#         folded = True
#         # lifted = True
#     return flat, semi, folded


def stateidentification(filename, flat, semi, folded):
    if filename == fn+'100001' or filename == fn+'100122' or filename == fn+'200124':
        flat = True
        semi = False
        folded = False
        # lifted = False
    elif name == fn+'100035' or name == fn+'200042' or name == fn+'300054':
        flat = False
        semi = True
        folded = False
        # lifted = False
    # elif name == 'one_hand_lifting_100089' or name == 'one_hand_lifting_200098':
    #     flat = False
    #     semi = False
    #     semi_crampled = True
    #     lifted = False
    elif name == fn+'10008700' or name == fn+'200090000' or name == fn+'3000910000':
        flat = False
        semi = False
        folded = True
        # lifted = True
    return flat, semi, folded

filenames = next(os.walk(PATH), (None, None, []))[2]

# pdb.set_trace()
flist = []
for file in filenames:
    flist.append(os.path.splitext(file)[0])


flist.sort()
data = []
temp_cloth_state = cloth_state[0]
for name in flist:
    data.append([name, temp_cloth_state])
    flat, semi, folded = stateidentification(name, flat, semi, folded)
    if flat:
        temp_cloth_state = cloth_state[0]
    elif semi:
        temp_cloth_state = cloth_state[1]
    elif folded:
        temp_cloth_state = cloth_state[2]
    # else:
    #     temp_cloth_state = cloth_state[3]

# print(data)


with open(PATH + 'data.csv', 'w', encoding='UTF8', newline='') as f:
    writer = csv.writer(f)

    # write the header
    writer.writerow(header)

    # write multiple rows
    writer.writerows(data)