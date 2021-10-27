#!/usr/bin/env python
##
##  vocutils.py - PASCAL VOX Utilities
##
##  usage: (view annotations)
##    $ ./vocutils.py -n10 VOC2007.zip
##
import sys
import os.path
import zipfile
import math
import numpy
import logging
import torch
from torch.utils.data import Dataset
from torch.nn.functional import mse_loss
from PIL import Image
from xml.etree.ElementTree import XML


CATEGORIES = [
    (None, 0),                  # 0
    ('aeroplane', 331),         # 1
    ('bicycle', 418),           # 2
    ('bird', 599),              # 3
    ('boat', 398),              # 4
    ('bottle', 634),            # 5
    ('bus', 272),               # 6
    ('car', 1644),              # 7
    ('cat', 389),               # 8
    ('chair', 1432),            # 9
    ('cow', 356),               # 10
    ('diningtable', 310),       # 11
    ('dog', 538),               # 12
    ('horse', 406),             # 13
    ('motorbike', 390),         # 14
    ('person', 5447),           # 15
    ('pottedplant', 625),       # 16
    ('sheep', 353),             # 17
    ('sofa', 425),              # 18
    ('train', 328),             # 19
    ('tvmonitor', 367)          # 20
]

COLORS = (
    'red', 'green', 'blue', 'magenta', 'cyan', 'yellow',
    'black', 'white', 'gray', 'orange', 'brown', 'pink',
    )

CAT2INDEX = { k:idx for (idx,(k,_)) in enumerate(CATEGORIES) }


##  PASCALDataset
##
class PASCALDataset(Dataset):

    def __init__(self, zip_path,
                 image_base='VOCtrainval_06-Nov-2007/VOCdevkit/VOC2007/JPEGImages/',
                 annot_base='VOCtrainval_06-Nov-2007/VOCdevkit/VOC2007/Annotations/'):
        Dataset.__init__(self)
        self.logger = logging.getLogger('PASCALDataset')
        self.image_base = image_base
        self.annot_base = annot_base
        self.data_zip = zipfile.ZipFile(zip_path)
        self.logger.info(f'zip_path={zip_path}')
        images = set()
        for name in self.data_zip.namelist():
            if not name.startswith(image_base): continue
            if not name.endswith('.jpg'): continue
            k = os.path.basename(name)[:-4]
            images.add(k)
        self.logger.info(f'images={len(images)}')
        annots = set()
        for name in self.data_zip.namelist():
            if not name.startswith(annot_base): continue
            if not name.endswith('.xml'): continue
            k = os.path.basename(name)[:-4]
            annots.add(k)
        self.logger.info(f'annots={len(annots)}')
        self.keys = sorted(images.intersection(annots))
        self.keys.sort()
        return

    def close(self):
        if self.data_zip is not None:
            self.data_zip.close()
            self.data_zip = None
        return

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index):
        if self.data_zip is None:
            raise IOError('file already closed')
        k = self.keys[index]
        image_path = self.image_base + k + '.jpg'
        with self.data_zip.open(image_path) as fp:
            image = Image.open(fp)
            image.load()
        annot_path = self.annot_base + k + '.xml'
        with self.data_zip.open(annot_path) as fp:
            elem = XML(fp.read())
        assert elem.tag == 'annotation'
        filename = None
        annot = []
        for obj in elem:
            if obj.tag == 'filename':
                filename = obj.text
                assert filename == k+'.jpg'
            elif obj.tag == 'object':
                name = None
                x0 = x1 = y0 = y1 = None
                for e in obj:
                    if e.tag == 'name':
                        name = e.text
                    elif e.tag == 'bndbox':
                        for c in e:
                            if c.tag == 'xmin':
                                x0 = int(c.text)
                            elif c.tag == 'xmax':
                                x1 = int(c.text)
                            elif c.tag == 'ymin':
                                y0 = int(c.text)
                            elif c.tag == 'ymax':
                                y1 = int(c.text)
                if (name is not None and x0 is not None and x1 is not None and
                    y0 is not None and y1 is not None):
                    annot.append((name, (x0,y0,x1-x0,y1-y0)))
        return (image, annot)


##  GridCell
##

def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))

# inv_sigmoid: inverse sigmoid()
def inv_sigmoid(y):
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

    def __init__(self, pos, conf, x, y, w, h, cat=0, cprobs=None):
        assert 0 <= conf <= 1, conf
        assert 0 <= x <= 1 and 0 <= y <= 1, (x,y)
        assert 0 <= w <= 1 and 0 <= h <= 1, (w,h)
        assert cat != 0 or cprobs is not None
        self.pos = pos
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
        return (f'<GridCell{self.pos}: conf={self.conf:.3f}, cat={cat}({prob:.2f}),'
                f' bbox=({self.x:.2f},{self.y:.2f},{self.w:.2f},{self.h:.2f})>')

    @classmethod
    def from_annot(klass, pos, conf, x, y, w, h, cat):
        return klass(pos, conf, sigmoid(x), sigmoid(y), w, h, cat=cat)

    @classmethod
    def from_tensor(klass, pos, v):
        return klass(pos, v[0], v[1], v[2], v[3], v[4], cprobs=v[5:])

    def get_bbox(self):
        return (inv_sigmoid(self.x), inv_sigmoid(self.y), self.w, self.h)

    def get_cat(self):
        if self.cat != 0: return (self.cat, 1.0)
        (cat,prob) = argmax(self.cprobs)
        return (cat, prob)

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
    return (x0+x*w0//ww, y0+y*h0//hh, w*w0//ww, h*h0//hh)
assert rect_map((0,0,10,10), (100,100), (10,10,20,20)) == (1,1,2,2)
assert rect_map((-5,-5,10,10), (100,100), (50,0,50,100)) == (0,-5,5,10)

# get_cells:
def get_cells(image, annot, input_size, output_size):
    (width, height) = input_size
    (rows, cols) = output_size
    src_size = image.size
    (dst_window,_) = rect_fit(input_size, src_size)
    cells = []
    for (pos, (x0,y0,w0,h0)) in rect_split((0,0,width,height), output_size):
        objs = []
        for (cat, (x1,y1,w1,h1)) in annot:
            assert 0 < cat
            (x1,y1,w1,h1) = rect_map(dst_window, src_size, (x1,y1,w1,h1))
            (_,_,iw,ih) = rect_intersect((x0,y0,w0,h0), (x1,y1,w1,h1))
            if 0 < iw and 0 < ih:
                (cx0,cy0) = (x0+w0/2,y0+h0/2)
                (cx1,cy1) = (x1+w1/2,y1+h1/2)
                objs.append(GridCell.from_annot(
                    pos,              # grid position
                    (iw*ih)/(w0*h0),  # objectness
                    (cx1-cx0)/width,  # delta x
                    (cy1-cy0)/height, # delta y
                    w1/width,         # w
                    h1/height,        # h
                    cat,
                ))
        if not objs:
            cells.append(None)
        else:
            (i,_) = argmax(objs, key=lambda obj:obj.conf)
            cells.append(objs[i])
    assert len(cells) == rows*cols
    image = adjust_image(image, dst_window, (width,height))
    return (image, cells)


# main
def main(argv):
    import getopt
    from PIL import ImageDraw
    def usage():
        print(f'usage: {argv[0]} [-O output] [-n images] voc.zip')
        return 100
    try:
        (opts, args) = getopt.getopt(argv[1:], 'dO:n:')
    except getopt.GetoptError:
        return usage()
    level = logging.INFO
    output_dir = None
    num_images = 0
    for (k, v) in opts:
        if k == '-d': level = logging.DEBUG
        elif k == '-O': output_dir = v
        elif k == '-n': num_images = int(v)

    logging.basicConfig(level=level)

    path = './PASCAL/VOC2007.zip'
    dataset = PASCALDataset(path)

    if output_dir is not None:
        for (i,(image,annot)) in enumerate(dataset):
            output = os.path.join(output_dir, f'output_{i:06d}.png')
            print(f'Image {i}: size={image.size}, annot={len(annot)}, output={output}')
            draw = ImageDraw.Draw(image)
            for (name,(x,y,w,h)) in annot:
                cat = CAT2INDEX[name]
                color = COLORS[cat % len(COLORS)]
                draw.rectangle((x,y,x+w,y+h), outline=color)
                draw.text((x,y), name, fill=color)
            image.save(output)
            if i+1 == num_images: break

    return

if __name__ == '__main__':
    sys.exit(main(sys.argv))
