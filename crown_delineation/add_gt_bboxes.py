import cv2
import numpy as np
import pandas as pd
from os.path import join
from glob import glob
from os.path import join
from os.path import basename

path = "./"
img_fname = "out1.png"
ann_fname = "/home/iharmon1/data/DeepForest-pytorch/evaluation4/NIWO-test.csv"
original_img_name = 'NIWO_001_2018.tif'

def draw_rect(raster, df):
    for idx, row in df.iterrows():
        cv2.rectangle(raster, (row['xmin'], row['ymin']), (row['xmax'], row['ymax']), (0, 255, 0), 1)
        #cv2.putText(raster, "{:.2f}".format(row[color]), (np.int((row['xmin'] + row['xmax'])/2) - 13, row['ymin'] + 10), font, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
        #cv2.putText(raster, "{:.2f}".format(row[-1]), (np.int((row[1] + row[3]) / 2) - 12, np.int((row[2] + row[4]) / 2) + 6), font, 0.6, (0, 0, 0), 1,cv2.LINE_AA)
    return raster


img_bgr = cv2.imread(join(path, img_fname))
test_df = pd.read_csv(ann_fname)
test_df_filtered = test_df.loc[test_df.image_path == original_img_name, :]

final_img = draw_rect(img_bgr, test_df_filtered)
cv2.imwrite("out2.png", final_img)
