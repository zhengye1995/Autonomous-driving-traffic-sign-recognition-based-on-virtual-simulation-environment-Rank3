import pandas as pd
import cv2
import os.path as osp
import numpy as np
import json
import os

import psutil
from multiprocessing import Pool
import time


def crop_img(img_name, xmin, xmax, ymin, ymax, crop_w, crop_h, img_root, patch_img_root,csv_file):
    path = img_root + img_name
    img = cv2.imread(path)
    assert img is not None
    height, width, _ = img.shape
    bbox_xmin = int(np.round(xmin))
    bbox_xmax = int(np.round(xmax))
    bbox_ymin = int(np.round(ymin))
    bbox_ymax = int(np.round(ymax))

    if bbox_xmin == 0 and bbox_xmax == 0:
        return csv_file, img+","+str(0)+","+str(0)+"\n"
    bbox_w = bbox_xmax - bbox_xmin
    bbox_h = bbox_ymax - bbox_ymin
    # 512*512
    if bbox_xmin + int(bbox_w / 2) >= (crop_w / 2) and (3200 - bbox_xmax) + int(bbox_w / 2) >= (crop_w / 2):
        x_start = bbox_xmin - (int(crop_w / 2) - int(bbox_w / 2) - 1)
    elif bbox_xmin < (crop_w / 2) - int(bbox_w / 2) and (3200 - bbox_xmax) + int(bbox_w / 2) >= (crop_w / 2):
        x_start = 0
    elif bbox_xmin >= (crop_w / 2) - int(bbox_w / 2) and (3200 - bbox_xmax) + int(bbox_w / 2) < (crop_w / 2):
        x_start = 3200 - crop_w
    else:
        return csv_file, img + "," + str(0) + "," + str(0) + "\n"

    if bbox_ymin > (crop_h / 2) - int(bbox_h / 2) and (1800 - bbox_ymax) + int(bbox_h / 2) > (crop_h / 2):
        y_start = bbox_ymin - (int(crop_h / 2) - int(bbox_h / 2) - 1)
    elif bbox_ymin < (crop_h / 2) - int(bbox_h / 2) and (1800 - bbox_ymax) + int(bbox_h / 2) > (crop_h / 2):
        y_start = 0
    elif bbox_ymin > (crop_h / 2) - int(bbox_h / 2) and (1800 - bbox_ymax) + int(bbox_h / 2) < (crop_h / 2):
        y_start = 1800 - crop_h
    else:
        return csv_file, img + "," + str(0) + "," + str(0) + "\n"

    img_patch = img[y_start: y_start + crop_h, x_start: x_start + crop_w, :]
    assert img_patch.shape == (crop_h, crop_w, 3)

    target_path = osp.join(patch_img_root, img_name)
    cv2.imwrite(target_path, img_patch)
    return [csv_file, img_name + "," + str(x_start) + "," + str(y_start) + "\n"]


def mycallback(x):
    with open(x[0], 'a+') as f:
        f.write(x[1])


if __name__ == '__main__':
    print("generate crop test img, please waiting a few minutes.")

    test_csv = "test_crop.csv"
    img_root = '../mmdetection/data/jiaotongbiaozhi/test/'
    crop_w = 224
    crop_h = 224
    patch_img_root = '../mmdetection/data/jiaotongbiaozhi/test_patch_'+str(crop_w)+'_'+str(crop_h)
    df = pd.read_csv(test_csv)
    if not os.path.exists(patch_img_root):
      os.makedirs(patch_img_root)  
    img_names = df['filename'].tolist()
    xmin_list = df['X1'].tolist()
    xmax_list = df['X3'].tolist()
    ymin_list = df['Y1'].tolist()
    ymax_list = df['Y3'].tolist()
    assert len(img_names) == 20256
    csv_file = '../generate_submit_csv/test_extar_'+str(crop_w)+'_'+str(crop_w)+'.csv'
    test_csv_offset = open(csv_file,"w")
    test_csv_offset.write("filename"+","+"x_start"+","+"y_start"+"\n")
    test_csv_offset.close()
    num_cpus = psutil.cpu_count(logical=False)
    e1 = time.time()
    pool = Pool(num_cpus)
    for i in range(len(img_names)):
        pool.apply_async(crop_img, (img_names[i],xmin_list[i],xmax_list[i],ymin_list[i],ymax_list[i],crop_w, crop_h, img_root, patch_img_root,csv_file,), callback=mycallback)
    pool.close()
    pool.join()
    e2 = time.time()
    print("time:{:.2f}s".format(float(e2 - e1)))