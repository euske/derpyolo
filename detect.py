#!/usr/bin/env python
##
##  detect.py - Mini YOLO detection.
##
##  usage:
##    (files)    $ ./detect.py -O. -i yolo_net.pt image.jpg
##    (realtime) $ ./detect.py -i -V yolo_net.pt
##
import sys
import torch
import torch.nn as nn
import numpy
import logging
import os.path
from math import exp
from PIL import Image, ImageDraw
from categories import CATEGORIES
from model import YOLONet
from objutils import GridCell, adjust_image, image2numpy
from objutils import rect_fit, rect_intersect, argmax


##  YOLOObject
##
class YOLOObject:

    def __init__(self, p, conf, cat, bbox):
        assert 0 < cat
        self.p = p
        self.conf = conf
        self.cat = cat
        (self.name, self.color) = CATEGORIES[cat]
        self.bbox = bbox
        return

    def __repr__(self):
        return (f'<YOLOObject({self.cat}:{self.name}): conf={self.conf:.3f}, bbox={self.bbox}>')

    def get_iou(self, bbox):
        (_,_,w,h) = rect_intersect(self.bbox, bbox)
        if w <= 0 or h <= 0: return 0
        (_,_,w0,h0) = self.bbox
        return (w*h)/(w0*h0)

# detect
def detect(model, image, max_objs=2):
    assert image.mode == 'RGB'
    rw = image.width / 7
    rh = image.height / 7
    inputs = [image2numpy(image)]
    inputs = torch.from_numpy(numpy.array(inputs))
    outputs = model(inputs)
    outputs = outputs.view(-1, 7, 7, max_objs, 5+len(CATEGORIES))
    # Collect objects found.
    found = []
    for (i,row) in enumerate(outputs[0]):
        for (j,cell) in enumerate(row):
            objs = [ GridCell.from_tensor((j,i), v) for v in cell ]
            for obj in objs:
                (cat,prob) = obj.get_cat()
                if cat == 0: continue
                (x,y,w,h) = obj.get_bbox()
                w *= image.width
                h *= image.height
                x = x*image.width-w/2 + (j*rw+rw/2)
                y = y*image.height-h/2 + (i*rh+rh/2)
                bbox = (int(x), int(y), int(w), int(h))
                found.append(YOLOObject((j,i), obj.conf*prob, cat, bbox))
    return found

# softnms: https://arxiv.org/abs/1704.04503
def softnms(objs, threshold):
    result = []
    score = { obj:obj.conf for obj in objs }
    while objs:
        (i,conf) = argmax(objs, key=lambda obj:score[obj])
        if conf < threshold: break
        m = objs[i]
        result.append(m)
        del objs[i]
        for obj in objs:
            v = m.get_iou(obj.bbox)
            score[obj] = score[obj] * exp(-3*v*v)
    return sorted(result, key=lambda obj:score[obj])

# renderobjs
def renderobjs(image, objs):
    draw = ImageDraw.Draw(image)
    for obj in objs:
        assert obj.color is not None
        (x,y,w,h) = obj.bbox
        draw.rectangle((x,y,x+w,y+h), outline=obj.color)
        draw.text((x,y), obj.name, fill=obj.color)
    return image

# init_model
def init_model(model_path, device_type='cpu'):
    torch.set_grad_enabled(False)
    device = torch.device(device_type)
    logging.info(f'Device: {device}')
    logging.info(f'Loading: {model_path}...')
    params = torch.load(model_path, map_location=device)
    max_objs = params['max_objs']
    model = YOLONet(device, max_objs*(5+len(CATEGORIES)))
    model.load_state_dict(params['model'])
    model.eval()
    return (model, max_objs)

# main
def main(argv):
    import getopt
    def usage():
        print(f'usage: {argv[0]} [-d] [-C] [-V] [-O output] [-t threshold] [-i model.pt] image.jpg ...')
        return 100
    try:
        (opts, args) = getopt.getopt(argv[1:], 'dCVO:t:i:')
    except getopt.GetoptError:
        return usage()
    level = logging.INFO
    device_type = 'cuda'
    video_capture = False
    output_dir = None
    threshold = 0.20
    model_path = 'yolo_net.pt'
    for (k, v) in opts:
        if k == '-d': level = logging.DEBUG
        elif k == '-C': device_type = 'cpu'
        elif k == '-V': video_capture = True
        elif k == '-O': output_dir = v
        elif k == '-t': threshold = float(v)
        elif k == '-i': model_path = v

    logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s', level=level)

    try:
        (model, max_objs) = init_model(model_path, device_type)
    except FileNotFoundError as e:
        logging.error(f'Error: {e}')
        raise

    if video_capture:
        import cv2
        video = cv2.VideoCapture(0)
        logging.info(f'Video capture: {video}...')
        while True:
            (ok, image_bgr) = video.read()
            if not ok: break
            image = numpy.flip(image_bgr, axis=2) # BGR -> RGB
            image = Image.fromarray(image, 'RGB')
            (window,_) = rect_fit(YOLONet.IMAGE_SIZE, image.size)
            image = adjust_image(image, window, YOLONet.IMAGE_SIZE)
            found = detect(model, image, max_objs=max_objs)
            found = softnms(found, threshold)
            print('Detected:', found)
            image = renderobjs(image, found)
            image = numpy.asarray(image)
            image_bgr = numpy.flip(image, axis=2) # RGB -> BGR
            cv2.imshow('YOLO', image_bgr)
            if 0 <= cv2.waitKey(1): break
    else:
        for path in args:
            logging.info(f'Load: {path}')
            image = Image.open(path)
            (window,_) = rect_fit(YOLONet.IMAGE_SIZE, image.size)
            image = adjust_image(image, window, YOLONet.IMAGE_SIZE)
            found = detect(model, image, max_objs=max_objs)
            mat = [ ['--']*7 for _ in range(7) ]
            for obj in found:
                (j,i) = obj.p
                mat[i][j] = f'{obj.cat:2d}'
            found = softnms(found, threshold)
            print('Detected:', found)
            for row in mat:
                print(' '.join(row))
            print()
            if output_dir is not None:
                image = renderobjs(image, found)
                (name,ext) = os.path.splitext(os.path.basename(path))
                outpath = os.path.join(output_dir, f'out_{name}.png')
                logging.info(f'Saved: {outpath}')
                image.save(outpath)
    return

if __name__ == '__main__':
    sys.exit(main(sys.argv))
