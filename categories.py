#!/usr/bin/env python
#
# usage:
#   $ python categories.py ./COCO/annotations/instances_val2017.json >> categories.py
#

import sys

COLORS = (
  'blue', 'red', 'green', 'yellow', 'cyan', 'magenta',
  'orange', 'pink', 'brown', 'white', 'gray', 'black',
)

# main
def main(argv):
    import getopt
    import json
    import random
    def usage():
        print(f'usage: {argv[0]} [-d] annot_path')
        return 100
    try:
        (opts, args) = getopt.getopt(argv[1:], 'd')
    except getopt.GetoptError:
        return usage()
    debug = 0
    for (k, v) in opts:
        if k == '-d': debug += 1
    if not args: return usage()

    path = args.pop(0)
    categories = {}
    with open(path) as fp:
        objs = json.load(fp)
        for obj in objs['categories']:
            cat_id = obj['id']
            cat_name = obj['name']
            assert cat_id not in categories
            categories[cat_id] = cat_name

    print('# Categories')
    print('CATEGORIES = {')
    print('   0: (None, None),')
    for (i,cat_id) in enumerate(sorted(categories.keys())):
        name = categories[cat_id]
        color = COLORS[cat_id % len(COLORS)]
        print(f'  {i+1:2d}: ({name!r}, {color!r}),')
    print('}')
    return 0

if __name__ == '__main__': sys.exit(main(sys.argv))

# Categories
CATEGORIES = {
   0: (None, None),
   1: ('person', 'red'),
   2: ('bicycle', 'green'),
   3: ('car', 'yellow'),
   4: ('motorcycle', 'cyan'),
   5: ('airplane', 'magenta'),
   6: ('bus', 'orange'),
   7: ('train', 'pink'),
   8: ('truck', 'brown'),
   9: ('boat', 'white'),
  10: ('traffic light', 'gray'),
  11: ('fire hydrant', 'black'),
  12: ('stop sign', 'red'),
  13: ('parking meter', 'green'),
  14: ('bench', 'yellow'),
  15: ('bird', 'cyan'),
  16: ('cat', 'magenta'),
  17: ('dog', 'orange'),
  18: ('horse', 'pink'),
  19: ('sheep', 'brown'),
  20: ('cow', 'white'),
  21: ('elephant', 'gray'),
  22: ('bear', 'black'),
  23: ('zebra', 'blue'),
  24: ('giraffe', 'red'),
  25: ('backpack', 'yellow'),
  26: ('umbrella', 'cyan'),
  27: ('handbag', 'pink'),
  28: ('tie', 'brown'),
  29: ('suitcase', 'white'),
  30: ('frisbee', 'gray'),
  31: ('skis', 'black'),
  32: ('snowboard', 'blue'),
  33: ('sports ball', 'red'),
  34: ('kite', 'green'),
  35: ('baseball bat', 'yellow'),
  36: ('baseball glove', 'cyan'),
  37: ('skateboard', 'magenta'),
  38: ('surfboard', 'orange'),
  39: ('tennis racket', 'pink'),
  40: ('bottle', 'brown'),
  41: ('wine glass', 'gray'),
  42: ('cup', 'black'),
  43: ('fork', 'blue'),
  44: ('knife', 'red'),
  45: ('spoon', 'green'),
  46: ('bowl', 'yellow'),
  47: ('banana', 'cyan'),
  48: ('apple', 'magenta'),
  49: ('sandwich', 'orange'),
  50: ('orange', 'pink'),
  51: ('broccoli', 'brown'),
  52: ('carrot', 'white'),
  53: ('hot dog', 'gray'),
  54: ('pizza', 'black'),
  55: ('donut', 'blue'),
  56: ('cake', 'red'),
  57: ('chair', 'green'),
  58: ('couch', 'yellow'),
  59: ('potted plant', 'cyan'),
  60: ('bed', 'magenta'),
  61: ('dining table', 'pink'),
  62: ('toilet', 'gray'),
  63: ('tv', 'blue'),
  64: ('laptop', 'red'),
  65: ('mouse', 'green'),
  66: ('remote', 'yellow'),
  67: ('keyboard', 'cyan'),
  68: ('cell phone', 'magenta'),
  69: ('microwave', 'orange'),
  70: ('oven', 'pink'),
  71: ('toaster', 'brown'),
  72: ('sink', 'white'),
  73: ('refrigerator', 'gray'),
  74: ('book', 'blue'),
  75: ('clock', 'red'),
  76: ('vase', 'green'),
  77: ('scissors', 'yellow'),
  78: ('teddy bear', 'cyan'),
  79: ('hair drier', 'magenta'),
  80: ('toothbrush', 'orange'),
}
