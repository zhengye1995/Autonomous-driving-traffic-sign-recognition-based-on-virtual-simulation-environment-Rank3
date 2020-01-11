import json
import os
from glob import glob
from tqdm import tqdm


def save(images, annotations, name):
    ann = {}
    ann['type'] = 'instances'
    ann['images'] = images
    ann['annotations'] = annotations

    category = [
            {'id': 1, 'name': 'Park', 'supercategory': 'jiaotongbiaozhi'},
            {'id': 2, 'name': 'Stop to give way', 'supercategory': 'jiaotongbiaozhi'},
            {'id': 3, 'name': 'Keep Right', 'supercategory': 'jiaotongbiaozhi'},
            {'id': 4, 'name': 'Left and right turns', 'supercategory': 'jiaotongbiaozhi'},
            {'id': 5, 'name': 'Bus passage', 'supercategory': 'jiaotongbiaozhi'},
            {'id': 6, 'name': 'left driving', 'supercategory': 'jiaotongbiaozhi'},
            {'id': 7, 'name': 'slow down', 'supercategory': 'jiaotongbiaozhi'},
            {'id': 8, 'name': 'Motor vehicles go straight and turn right', 'supercategory': 'jiaotongbiaozhi'},
            {'id': 9, 'name': 'Watch For Pedestrians', 'supercategory': 'jiaotongbiaozhi'},
            {'id': 10, 'name': 'roundabout', 'supercategory': 'jiaotongbiaozhi'},
            {'id': 11, 'name': 'Go straight and turn right', 'supercategory': 'jiaotongbiaozhi'},
            {'id': 12, 'name': 'No buses allowed', 'supercategory': 'jiaotongbiaozhi'},
            {'id': 13, 'name': 'No motorcycles allowed', 'supercategory': 'jiaotongbiaozhi'},
            {'id': 14, 'name': 'No Motor Vehicles', 'supercategory': 'jiaotongbiaozhi'},
            {'id': 15, 'name': 'No non-motor vehicles allowed', 'supercategory': 'jiaotongbiaozhi'},
            {'id': 16, 'name': 'No Honking', 'supercategory': 'jiaotongbiaozhi'},
            {'id': 17, 'name': 'Go straight and turn at the overpass', 'supercategory': 'jiaotongbiaozhi'},
            {'id': 18, 'name': '40 kilometers', 'supercategory': 'jiaotongbiaozhi'},
            {'id': 19, 'name': '30 kilometers', 'supercategory': 'jiaotongbiaozhi'},
            {'id': 20, 'name': 'honking', 'supercategory': 'jiaotongbiaozhi'},

    ]
    ann['categories'] = category
    json.dump(ann, open('../generate_submit_csv/jiaotongbiaozhi_{}.json'.format(name), 'w'))


def test_dataset(im_dir):
    im_list = glob(os.path.join(im_dir, '*.jpg'))
    idx = 1
    image_id = 20190000000
    images = []
    annotations = []
    h, w, = 1800, 3200
    for im_path in tqdm(im_list):
        image_id += 1
        image = {'file_name': os.path.split(im_path)[-1], 'width': w, 'height': h, 'id': image_id}
        images.append(image)
        labels = [[10, 10, 20, 20]]
        for label in labels:
            bbox = [label[0], label[1], label[2] - label[0], label[3] - label[1]]
            seg = []
            ann = {'segmentation': [seg], 'area': bbox[2] * bbox[3], 'iscrowd': 0, 'image_id': image_id,
                   'bbox': bbox, 'category_id': 1, 'id': idx, 'ignore': 0}
            idx += 1
            annotations.append(ann)
    save(images, annotations, 'test')


if __name__ == '__main__':
    test_dir = '../mmdetection/data/jiaotongbiaozhi/test'
    print("generate test json label file.")
    test_dataset(test_dir)