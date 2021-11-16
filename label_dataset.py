import os
from glob import glob

import pandas as pd
import numpy as np

import csv

import pdb

PATH = "/media/gtzelepis/DATA/real_data/videos/data/two_hands_side/"
EXT = "*.csv"
img_EXT = "*.png"

header = ['filename', 'cloth_state']

cloth_state = ['flat', 'semi-lifted-twohands', 'folded']

flat = True
semi = False
folded = False


def stateidentification(filename, flat, semi, folded):
    if filename == 'two_hands_side_100227' or filename == 'two_hands_side_100001':
        flat = True
        semi = False
        folded = False
        # lifted = False
    elif name == 'two_hands_side_100074' or name == 'two_hands_side_200072':
        flat = False
        semi = True
        folded = False
        # lifted = False
    # elif name == 'one_hand_lifting_100089' or name == 'one_hand_lifting_200098':
    #     flat = False
    #     semi = False
    #     semi_crampled = True
    #     lifted = False
    elif name == 'two_hands_side_100142' or name == 'two_hands_side_200132':
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