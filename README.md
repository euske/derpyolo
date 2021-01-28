# Mini YOLO

My own implementation of YOLO (mixture of v1 and v2).


## Prerequisites:

 - Python3 (https://www.python.org/)
 - PyTorch (https://pytorch.org/)
 - COCO dataset (https://cocodataset.org/)
 - (optinal) CUDA


## Train

 1. Put the COCO files ./COCO/ directory:
    - ./COCO/train2017.zip
    - ./COCO/annotations/instances_train2017.json

 2. Do this:

    `$ python train.py -n10 -o./yolo_net.pt ./COCO/train2017.zip ./COCO/annotations/instances_train2017.json`


## Use

### Annotate image:

    $ ./detect.py -O. yolo_net.pt image.jpg

### Real-time detection with camera:

    $ ./detect.py -V yolo_net.pt
