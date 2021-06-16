import os
from glob import glob

import pandas as pd

import csv

import pdb

'''Helper function to customize/group the data before feeding them into pytorch'''

PATH = "/home/gtzelepis/Data/cloth_manipulation/small_dataset/two_hands_sideways/"
EXT = "*.csv"
img_EXT = "*.png"



# all_csv_files = [file
#                  for path, subdir, files in os.walk(PATH)
#                  for file in glob(os.path.join(path, EXT))]

all_csv_files = []
all_img_files = []
for path, subdir, files in os.walk(PATH):
    for file in glob(os.path.join(path, EXT)):
        if file.endswith('data.csv'):
            your_value = 'semi-lifted-twohands'  # value that you want to replace with
            with open(file, 'r') as infile, open('output.csv', 'w') as outfile:
                reader = csv.reader(infile)
                writer = csv.writer(outfile)
                df = pd.read_csv(file, index_col=None, header=0)
                df.replace('semi-lifted', 'semi-lifted-twohands', inplace=True)
                df.to_csv(file)
                # pdb.set_trace()
#             all_csv_files.append(file)
#     for file in glob(os.path.join(path, img_EXT)):
#         all_img_files.append(file)
#
#
# li = []
#
# for filename in all_csv_files:
#     df = pd.read_csv(filename, index_col=None, header=0)
#     li.append(df)
#
# frame = pd.concat(li, axis=0, ignore_index=True)


