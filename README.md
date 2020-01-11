代码说明-交通标志识别+学习使我快乐+89520


1. 环境要求：
    ubuntu16.04
    GPU Nvidia-1080ti*2
    pytorch1.1 (注意一定是最新版本)
    torchvision
    python3.6.5
    python-opencv
    numpy
    pandas
    psutil
    tqdm
    glob
    cython
    future

2. 数据及预训练模型准备：
    a. 将训练数据解压到目录xxx1，然后建立软链 ln -s xxx1 code/mmdetection/data/jiaotongbiaozhi/train
    b. 将训练数据train_label_fix.csv 文件放置在目录 code/data_process 下
    c. 将测试数据解压到目录xxx2，然后建立软链 ln -s xxx2 code/mmdetection/data/jiaotongbiaozhi/test
    运行：
    d. 进入目录 code/data_process
    e. chmod +x data_process.sh
    f. ./data_process.sh
    最终会生成训练及测试所需要的数据及label，由于采用了多进程来加速，可能会导致CPU占用过高，详情见crop_patch_test_mutilprocess.py 中的进程池
    g. 下载预训练模型：https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/hrnet/cascade_rcnn_hrnetv2_w32_fpn_20e_20190522-55bec4ee.pth
        并且放置于：code/mmdetection/data/pretrained 目录下
       执行：
       python change_coco_weights_class_num.py
      
3. 编译：
    a. 进入目录 code/mmdetection
    b. chmod +x complie.sh
    c. ./complie.sh
    d. python setup.py install
    
4. 训练：
    a. 进入目录 code/mmdetection
    b. chmod +x tools/dist_train.sh
    c. ./tools/dist_train.sh config/jiaotongbiaozhi/cascade_rcnn_hrnetv2p_w32_20e.py 2

5. 预测：
    预测需要我们训练好的模型，如有需要请邮件联系，提供网盘地址，下载后放置到：
    code/mmdetection/work_dirs/cascade_rcnn_hrnetv2p_w32/ 目录下
    a. chmod +x tools/dist_test.sh
    b. ./tools/dist_test.sh config/jiaotongbiaozhi/cascade_rcnn_hrnetv2p_w32_20e.py work_dirs/cascade_rcnn_hrnetv2p_w32/latest.pth --eval bbox --out result.pkl
    c. 进入目录 code/generate_submit_csv
    d. 执行：
        python json2csv_crop.py
        得到submit.csv
    


