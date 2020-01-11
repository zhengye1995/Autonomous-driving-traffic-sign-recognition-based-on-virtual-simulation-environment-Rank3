import pandas as pd
import os.path as osp
import numpy as np
import json
import os


def save(images, annotations, name):
    ann = {}
    ann['type'] = 'instances'
    ann['images'] = images
    ann['annotations'] = annotations
    category = [
            {'id': 1, 'name': '停车场', 'supercategory': 'jiaotongbiaozhi'},
            {'id': 2, 'name': '停车让行', 'supercategory': 'jiaotongbiaozhi'},
            {'id': 3, 'name': '右侧行驶', 'supercategory': 'jiaotongbiaozhi'},
            {'id': 4, 'name': '向左和向右转弯', 'supercategory': 'jiaotongbiaozhi'},
            {'id': 5, 'name': '大客车通行', 'supercategory': 'jiaotongbiaozhi'},
            {'id': 6, 'name': '左侧行驶', 'supercategory': 'jiaotongbiaozhi'},
            {'id': 7, 'name': '慢行', 'supercategory': 'jiaotongbiaozhi'},
            {'id': 8, 'name': '机动车直行和右转弯', 'supercategory': 'jiaotongbiaozhi'},
            {'id': 9, 'name': '注意行人', 'supercategory': 'jiaotongbiaozhi'},
            {'id': 10, 'name': '环岛行驶', 'supercategory': 'jiaotongbiaozhi'},
            {'id': 11, 'name': '直行和右转弯', 'supercategory': 'jiaotongbiaozhi'},
            {'id': 12, 'name': '禁止大客车通行', 'supercategory': 'jiaotongbiaozhi'},
            {'id': 13, 'name': '禁止摩托车通行', 'supercategory': 'jiaotongbiaozhi'},
            {'id': 14, 'name': '禁止机动车通行', 'supercategory': 'jiaotongbiaozhi'},
            {'id': 15, 'name': '禁止非机动车通行', 'supercategory': 'jiaotongbiaozhi'},
            {'id': 16, 'name': '禁止鸣喇叭', 'supercategory': 'jiaotongbiaozhi'},
            {'id': 17, 'name': '立交直行和转弯行驶', 'supercategory': 'jiaotongbiaozhi'},
            {'id': 18, 'name': '限制速度40公里每小时', 'supercategory': 'jiaotongbiaozhi'},
            {'id': 19, 'name': '限速30公里每小时', 'supercategory': 'jiaotongbiaozhi'},
            {'id': 20, 'name': '鸣喇叭', 'supercategory': 'jiaotongbiaozhi'},
          ]
    ann['categories'] = category
    json.dump(ann, open(os.path.join(data_dir, 'raw_image_jiaotongbiaozhi_{}.json'.format(name)), 'w'), indent=2)


train_csv = "train_label_fix.csv"
img_root = '../mmdetection/data/jiaotongbiaozhi/train'
data_dir = '../mmdetection/data/jiaotongbiaozhi/annotations'
df = pd.read_csv(train_csv)

img_names = df['filename'].tolist()
xmin_list = df['X1'].tolist()
xmax_list = df['X3'].tolist()
ymin_list = df['Y1'].tolist()
ymax_list = df['Y3'].tolist()
class_label = df['type'].tolist()


json_images = []
json_annos = []
image_id = -1
idx = 1
print("transfrom csv label to json format!")
for i, img_name in enumerate(tqdm(img_names)):
    path = img_root + img_name
    height, width = 1800, 3200
    bbox_xmin = xmin_list[i]
    bbox_xmax = xmax_list[i]
    bbox_ymin = ymin_list[i]
    bbox_ymax = ymax_list[i]
    bbox_w = bbox_xmax - bbox_xmin
    bbox_h = bbox_ymax - bbox_ymin
    
    image_id += 1
    image = {'file_name': img_name, 'width': width, 'height': height, 'id': image_id}
    json_images.append(image)
    ann = {'segmentation': [[]], 'area': bbox_w * bbox_h, 'iscrowd': 0, 'image_id': image_id,
           'bbox': [bbox_xmin, bbox_ymin, bbox_w, bbox_h], 'category_id': class_label[i], 'id': idx, 'ignore': 0}
    idx += 1
    json_annos.append(ann)

save(json_images, json_annos, "train")






