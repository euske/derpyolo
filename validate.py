#!/usr/bin/env python
##
##  validate.py - Mini YOLO validator.
##
##  usage:
##    $ ./validate.py -C ./yolo_net.pt ./COCO/train.zip ./COCO/instances.json
##
import sys
import torch
import logging
from torch.utils.data import DataLoader
from categories import CATEGORIES
from model import YOLONet
from detect import detect, softnms
from objutils import COCODataset, rect_fit, rect_map, adjust_image

# main
def main(argv):
    import getopt
    def usage():
        print(f'usage: {argv[0]} [-d] [-C] [-t threshold] [-i model.pt] images.zip annots.json')
        return 100
    try:
        (opts, args) = getopt.getopt(argv[1:], 'dCt:i:')
    except getopt.GetoptError:
        return usage()
    level = logging.INFO
    device_type = 'cuda'
    threshold = 0.20
    model_path = './yolo_net.pt'
    iou = 0.50
    for (k, v) in opts:
        if k == '-d': level = logging.DEBUG
        elif k == '-C': device_type = 'cpu'
        elif k == '-t': threshold = float(v)
        elif k == '-i': model_path = v

    image_path = './COCO/val2017.zip'
    annot_path = './COCO/annotations/instances_val2017.json'
    if args:
        image_path = args.pop(0)
    if args:
        annot_path = args.pop(0)

    logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s', level=level)

    torch.set_grad_enabled(False)

    device = torch.device(device_type)
    logging.info(f'Device: {device}')

    logging.info(f'Loading: {model_path}...')
    try:
        params = torch.load(model_path, map_location=device)
    except FileNotFoundError as e:
        logging.error(f'Error: {e}')
        raise
    max_objs = params['max_objs']
    model = YOLONet(device, max_objs*(5+len(CATEGORIES)))
    model.load_state_dict(params['model'])
    model.eval()

    dataset = COCODataset(image_path, annot_path)
    dataset.open()
    loader = DataLoader(dataset, collate_fn=lambda x:x)

    m = 0
    n = 0
    for sample in loader:
        for (image,annot) in sample:
            if image.mode != 'RGB': continue
            if not annot: continue
            (window,_) = rect_fit(YOLONet.IMAGE_SIZE, image.size)
            image = adjust_image(image, window, YOLONet.IMAGE_SIZE)
            found = detect(model, image, max_objs=max_objs)
            found = softnms(found, threshold)
            correct = set()
            for (cat, bbox) in annot:
                bbox = rect_map(window, image.size, bbox)
                for obj in found:
                    if obj.cat == cat and iou <= obj.get_iou(bbox):
                        correct.add(obj)
            tp = 0
            for k in range(len(found)):
                if found[k] in correct:
                    objs = set(found[:k+1])
                    tp += len(objs.intersection(correct)) / len(objs)
            ap = tp / len(annot)
            print(f'Image {n}: found={len(found)}, annot={len(annot)}, correct={len(correct)}, ap={ap:.2f}')
            m += ap
            n += 1

    print(f'mAP: {m/max(n,1)}')
    return

if __name__ == '__main__':
    sys.exit(main(sys.argv))
