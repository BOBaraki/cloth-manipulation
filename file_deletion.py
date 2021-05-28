import os
import pandas as pd

import pdb

path = "/home/gtzelepis/Data/cloth_manipulation/RGB/"

flist = pd.read_csv("/home/gtzelepis/Data/cloth_manipulation/data.csv")



file_name = flist["filename"].tolist()

# pdb.set_trace()

for filename in os.listdir(path):
    # print(filename)
    tempfile = filename.replace('.png', '')
    if tempfile not in file_name:
        # pdb.set_trace()
        print(filename)
        os.remove(path + filename)