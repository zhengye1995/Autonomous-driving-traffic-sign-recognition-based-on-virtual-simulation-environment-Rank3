import numpy as np
import json
import pandas as pd


test_json_raw = json.load(open("jiaotongbiaozhi_test.json", "r"))
test_json = json.load(open("../mmdetection/result.pkl.json", "r"))
test_csv = open("submit.csv", "w")
offset_csv = open("test_extar_224_224.csv")
df = pd.read_csv(offset_csv)

img_names = df['filename'].tolist()
x_starts = df['x_start'].tolist()
y_starts = df['y_start'].tolist()
x_starts_filename = dict(zip(img_names, x_starts))
y_starts_filename = dict(zip(img_names, y_starts))
raw_image_filenames = []
images_ids = {}
for img in test_json_raw["images"]:
    images_ids[img["id"]] = img["file_name"]
    raw_image_filenames.append(img["file_name"])
raw_image_filenames = set(raw_image_filenames)

with_object_imgnames = []
test_csv.write("filename,X1,Y1,X2,Y2,X3,Y3,X4,Y4,type\n")
for anno in test_json:
    label = anno["category_id"]
    bbox = anno["bbox"]
    filename = images_ids[anno["image_id"]]
    with_object_imgnames.append(filename)
    w,h = bbox[2],bbox[3]
    x1 = bbox[0]
    y1 = bbox[1]
    x2 = bbox[0] + w
    y2 = bbox[1]
    x3 = bbox[0] + w
    y3 = bbox[1] + h
    x4 = bbox[0]
    y4 = bbox[1] + h
    x_start = x_starts_filename[filename]
    y_start = y_starts_filename[filename]
    x1 += x_start
    x2 += x_start
    x3 += x_start
    x4 += x_start
    y1 += y_start
    y2 += y_start
    y3 += y_start
    y4 += y_start
    test_csv.write(filename+","+str(x1)+","+str(y1)+","+str(x2)+","+str(y2)+","+str(x3)+","+str(y3)+","+str(x4)+","+str(y4)+","+str(label)+"\n")
with_object_imgnames = set(with_object_imgnames)

no_object_imgnames = raw_image_filenames - with_object_imgnames
for img in list(no_object_imgnames):
    test_csv.write(img + "," + str(0) + "," + str(0) + "," + str(0) + "," + str(0) + "," + str(0) + "," + str(
        0) + "," + str(0) + "," + str(0) + "," + str(0) + "\n")
