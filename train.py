#!/usr/bin/env python
##
##  train.py - Mini YOLO training.
##
##  usage:
##    $ ./train.py -o./yolo_net.pt -n10 ./COCO/train.zip ./COCO/instances.json
##
import sys
import torch
import torch.optim as optim
import numpy
import random
import logging
from PIL import Image
from torch.utils.data import DataLoader
from categories import CATEGORIES
from model import YOLONet
from objutils import COCODataset, GridCell, adjust_image, image2numpy
from objutils import rect_fit, rect_map, rect_intersect, rect_split, argmax

# train
def train(loader, model, optimizer, max_objs=2):
    n = 0
    for (batch, sample) in enumerate(loader):
        inputs = []
        targets = []
        for (image, annot) in sample:
            if image.mode != 'RGB': continue
            (image, cells) = get_cells(image, annot, YOLONet.IMAGE_SIZE, (7,7))
            inputs.append(image2numpy(image))
            targets.append(cells)
        assert len(inputs) == len(targets)
        optimizer.zero_grad()
        # targets: [bs, 7*7, max_objs, 5+ncats]
        inputs = torch.from_numpy(numpy.array(inputs))
        # inputs: [bs, 3, 224, 224]
        outputs = model(inputs)
        # outputs: [bs, 7, 7, nvals]
        outputs = outputs.view(-1, 7*7, max_objs, 5+len(CATEGORIES))
        # outputs: [bs, 7*7, max_objs, 5+ncats]
        loss = torch.tensor(0.)
        for (cells0,cells1) in zip(targets, outputs):
            for (i,(obj0,objs1)) in enumerate(zip(cells0, cells1)):
                pos = (i%7, i//7)
                if obj0 is None:
                    for vec1 in objs1:
                        obj1 = GridCell.from_tensor(pos, vec1)
                        loss += obj1.get_cost_noobj()
                else:
                    (idx,_) = argmax(objs1, key=lambda obj:obj[0])
                    obj1 = GridCell.from_tensor(pos, objs1[idx])
                    loss += obj0.get_cost_full(obj1)
        loss.backward()
        optimizer.step()
        n += len(inputs)
        logging.info(f'Batch {batch}: n={n}, loss={loss.item()/len(inputs):.4f}')
    return

# main
def main(argv):
    import getopt
    def usage():
        print(f'usage: {argv[0]} [-d] [-C] [-o model.pt] [-s seed] [-n epochs] [-b size] [-r rate] images.zip annots.json')
        return 100
    try:
        (opts, args) = getopt.getopt(argv[1:], 'dCo:n:s:b:r:')
    except getopt.GetoptError:
        return usage()
    level = logging.INFO
    device_type = 'cuda'
    model_path = None
    random_seed = None
    num_epochs = 1
    batch_size = 64
    rate = 0.001
    shuffle = True
    max_objs = 2
    for (k, v) in opts:
        if k == '-d': level = logging.DEBUG
        elif k == '-C': device_type = 'cpu'
        elif k == '-o': model_path = v
        elif k == '-s': random_seed = int(v)
        elif k == '-n': num_epochs = int(v)
        elif k == '-b': batch_size = int(v)
        elif k == '-r': rate = float(v)

    logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s', level=level)

    image_path = './COCO/val2017.zip'
    annot_path = './COCO/annotations/instances_val2017.json'
    if args:
        image_path = args.pop(0)
    if args:
        annot_path = args.pop(0)

    device = torch.device(device_type)
    logging.info(f'Device: {device}')

    logging.info(f'Train: seed={random_seed}')
    if random_seed is not None:
        random.seed(random_seed)
        torch.manual_seed(random_seed)

    model = None
    if model_path is not None:
        logging.info(f'Loading: {model_path}...')
        try:
            params = torch.load(model_path, map_location=device)
            max_objs = params['max_objs']
            model = YOLONet(device, max_objs*(5+len(CATEGORIES)))
            model.load_state_dict(params['model'])
        except FileNotFoundError as e:
            logging.error(f'Error: {e}')
    if model is None:
        model = YOLONet(device, max_objs*(5+len(CATEGORIES)))
    model.train()

    dataset = COCODataset(image_path, annot_path)
    dataset.open()
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle,
        collate_fn=lambda x:x)
    catratio = dataset.get_catratio()
    GridCell.L_CPROBS = { i:pow(1.0/r, 0.5) for (i,r) in catratio.items() }

    optimizer = optim.Adam(model.parameters(), lr=rate)
    for epoch in range(num_epochs):
        logging.info(f'Train: epoch={epoch}/{num_epochs}, batch_size={batch_size}')
        train(loader, model, optimizer, max_objs=max_objs)
        if model_path is not None:
            logging.info(f'Saving: {model_path}...')
            params = {
                'max_objs': max_objs,
                'model': model.state_dict(),
            }
            torch.save(params, model_path)

    return

if __name__ == '__main__':
    sys.exit(main(sys.argv))
