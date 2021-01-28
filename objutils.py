#!/usr/bin/env python
##
##  objutils.py - Utilities
##
import json
import zipfile
import os.path
import math
import numpy
import logging
import torch
from torch.utils.data import Dataset
from torch.nn.functional import mse_loss
from PIL import Image
from categories import CATEGORIES


##  COCODataset
##
class COCODataset(Dataset):

    def __init__(self, image_path, annot_path):
        Dataset.__init__(self)
        self.image_path = image_path
        self.annot_path = annot_path
        self.logger = logging.getLogger()
        return

    def open(self):
        self.logger.info(f'COCODataset: image_path={self.image_path}')
        self.image_zip = zipfile.ZipFile(self.image_path)
        images = {}
        for name in self.image_zip.namelist():
            if name.endswith('/'): continue
            (image_id,ext) = os.path.splitext(os.path.basename(name))
            if ext != '.jpg': continue
            images[int(image_id)] = name
        self.logger.info(f'COCODataset: images={len(images)}')
        catname2idx = { k:idx for (idx,(k,_)) in CATEGORIES.items() }
        annots = {}
        self.logger.info(f'COCODataset: annot_path={self.annot_path}')
        with open(self.annot_path) as fp:
            objs = json.load(fp)
            catid2idx = {}
            for obj in objs['categories']:
                cat_id = obj['id']
                cat_name = obj['name']
                assert cat_name in catname2idx, cat_name
                catid2idx[cat_id] = catname2idx[cat_name]
            for obj in objs['annotations']:
                cat_id = obj['category_id']
                if cat_id not in catid2idx: continue
                image_id = obj['image_id']
                bbox = obj['bbox']
                if image_id in annots:
                    a = annots[image_id]
                else:
                    a = annots[image_id] = []
                a.append((catid2idx[cat_id], bbox))
        self.logger.info(f'COCODataset: annots={sum(map(len, annots.values()))}')
        self.data = [ (images[i], annots.get(i)) for i in sorted(images.keys()) ]
        return

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        (name, annot) = self.data[index]
        with self.image_zip.open(name) as fp:
            image = Image.open(fp)
            image.load()
        return (image, annot or [])


##  GridCell
##

def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))

# diomgis: inverse sigmoid()
def diomgis(y):
    if 0 < y and y < 1:
        return -math.log(1.0/y - 1.0)
    raise ValueError(y)

def argmax(a, key=lambda x:x):
    (imax, vmax) = (None, None)
    for (i,x) in enumerate(a):
        v = key(x)
        if vmax is None or vmax < v:
            (imax, vmax) = (i, v)
    if imax is None: raise ValueError(a)
    return (imax, vmax)

def mse(cat, cprobs):
    target = cprobs.new([ 1. if i == cat else 0. for i in range(len(cprobs)) ])
    return mse_loss(cprobs, target)

class GridCell:

    L_NOOBJ = 0.5
    L_COORD = 5.0

    def __init__(self, p, conf, x, y, w, h, cat=0, cprobs=None):
        assert 0 <= conf <= 1, conf
        assert 0 <= x <= 1 and 0 <= y <= 1, (x,y)
        assert 0 <= w <= 1 and 0 <= h <= 1, (w,h)
        assert cat != 0 or cprobs is not None
        self.p = p
        self.conf = conf
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.cat = cat
        self.cprobs = cprobs
        return

    def __repr__(self):
        (cat,prob) = self.get_cat()
        return (f'<GridCell{self.p}: conf={self.conf:.3f}, cat={cat}({prob:.2f}),'
                f' bbox=({self.x:.2f},{self.y:.2f},{self.w:.2f},{self.h:.2f})>')

    @classmethod
    def frombox(klass, p, conf, x, y, w, h, cat):
        return klass(p, conf, sigmoid(x), sigmoid(y), w, h, cat=cat)

    @classmethod
    def fromvec(klass, p, v):
        return klass(p, v[0], v[1], v[2], v[3], v[4], cprobs=v[5:])

    def get_bbox(self):
        return (diomgis(self.x), diomgis(self.y), self.w, self.h)

    def get_cat(self):
        if self.cat != 0: return (self.cat, 1.0)
        (i,p) = argmax(self.cprobs)
        return (i, p.item())

    def get_cost_noobj(self):
        assert self.cprobs is not None
        return (self.L_NOOBJ *
                (self.conf - 0)**2 +
                mse(0, self.cprobs))

    def get_cost_full(self, obj):
        assert self.cat != 0
        return (self.L_COORD *
                ((self.x - obj.x)**2 +
                 (self.y - obj.y)**2 +
                 (self.w - obj.w)**2 +
                 (self.h - obj.h)**2) +
                (self.conf - obj.conf)**2 +
                mse(self.cat, obj.cprobs))


##  Utils.
##

# image2numpy: convert a PIL image to a NumPy array.
def image2numpy(image):
    assert image.mode == 'RGB'
    a = numpy.array(image, dtype=numpy.float32)
    return (a/255.0).transpose(2,0,1)

# adjust_image:
def adjust_image(src, window, image_size):
    (xd,yd,wd,hd) = window
    dst = Image.new(src.mode, image_size)
    dst.paste(src.resize((wd,hd)), (xd,yd))
    return dst

# rect_split: returns [((x,y),(rx,ry,rw,rh)), ...]
def rect_split(rect, grid):
    (x,y,w,h) = rect
    (nx,ny) = grid
    for i in range(ny):
        for j in range(nx):
            yield ((j, i), (x+j*w//nx, y+i*h//ny, w//nx, h//ny))
    return

# rect_intersect: returns the area of the two rects intersect.
def rect_intersect(rect0, rect1):
    (x0,y0,w0,h0) = rect0
    (x1,y1,w1,h1) = rect1
    x = max(x0, x1)
    y = max(y0, y1)
    w = min(x0+w0, x1+w1) - x
    h = min(y0+h0, y1+h1) - y
    return (x, y, w, h)

# rect_fit: returns (rect_in_frame, frame_in_rect)
def rect_fit(frame, size):
    (w0,h0) = frame
    (w1,h1) = size
    if h1*w0 < w1*h0:
        # horizontal fit (w1->w0)
        (wd,hd) = (w0, h1*w0//w1) # < (w0,h0)
        (ws,hs) = (w1, h0*w1//w0) # > (w1,h1)
    else:
        # vertical fit (x w0/w1)
        (wd,hd) = (w1*h0//h1, h0) # < (w0,h0)
        (ws,hs) = (w0*h1//h0, h1) # > (w1,h1)
    (xd,yd) = ((w0-wd)//2, (h0-hd)//2)
    (xs,ys) = ((w1-ws)//2, (h1-hs)//2)
    return ((xd,yd,wd,hd), (xs,ys,ws,hs))
assert rect_fit((100,100), (200,200)) == ((0,0,100,100), (0,0,200,200))
assert rect_fit((100,100), (100,200)) == ((25,0,50,100), (-50,0,200,200))
assert rect_fit((200,100), (100,200)) == ((75,0,50,100), (-150,0,400,200))

# rect_map:
def rect_map(frame, size, rect):
    (x0,y0,w0,h0) = frame
    (ww,hh) = size
    (x,y,w,h) = rect
    (x1,y1) = (x+w, y+h)
    return (x0+x*w0//ww, y0+y*h0//hh, w*w0//ww, h*h0//hh)
assert rect_map((0,0,10,10), (100,100), (10,10,20,20)) == (1,1,2,2)
assert rect_map((-5,-5,10,10), (100,100), (50,0,50,100)) == (0,-5,5,10)
