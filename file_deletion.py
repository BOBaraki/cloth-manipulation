import os
import pandas as pd

import pdb

DIR = "/home/gtzelepis/Data/cloth_manipulation/one_hand_lowering/cloth_yellow_table_white/"

path = DIR + "RGB/"

path_depth = DIR + "depth/"

path_points = DIR + "points/"

flist = pd.read_csv(DIR + "data.csv")



file_name = flist["filename"].tolist()

# pdb.set_trace()

for filename in os.listdir(path):
    # print(filename)
    tempfile = filename.replace('.png', '')
    if tempfile not in file_name:
        # pdb.set_trace()
        print(filename)
        os.remove(path + filename)


for filename in os.listdir(path_depth):
    # print(filename)
    tempfile = filename.replace('.tif', '')
    if tempfile not in file_name:
        # pdb.set_trace()
        print(filename)
        os.remove(path_depth + filename)


for filename in os.listdir(path_points):
    # print(filename)
    tempfile = filename.replace('.csv', '')
    if tempfile not in file_name:
        # pdb.set_trace()
        print(filename)
        os.remove(path_points + filename)