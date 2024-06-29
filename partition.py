import shutil
import os
import json
import numpy as np
import random

from utils_various import *


createDirectory(os.path.join(mySPEED_dir, 'images'))
createDirectory(os.path.join(mySPEED_dir, 'images/train'))
createDirectory(os.path.join(mySPEED_dir, 'images/dev'))
createDirectory(os.path.join(mySPEED_dir, 'images/test'))

# Load JSON file with labels of original train set
with open(originalSPEED_dir + '/train.json') as jFile:
    jData = json.load(jFile)  # list of dictionaries w/ filename & pose

jDataNoBG = jData[:6000]
random.shuffle(jDataNoBG)
jDataBG = jData[6000:]
random.shuffle(jDataBG)

# Partition the original train set into train, dev, test (64%, 16%, 20%)
# (bc labels are not disclosed for the original test set)
train_len = int(0.65*len(jDataNoBG))
dev_len = int(0.15*len(jDataNoBG))

partitions = [{'name': 'Training', 'dir': 'train',
                  'data': jDataNoBG[:train_len] + jDataBG[:train_len]},
              {'name': 'Development', 'dir': 'dev',
                  'data': jDataNoBG[train_len:train_len+dev_len] + jDataBG[train_len:train_len+dev_len]},
              {'name': 'Test', 'dir': 'test',
                  'data': jDataNoBG[train_len+dev_len:] + jDataBG[train_len+dev_len:]}]


## Write new repartitioned dataset (SPEED_MP) JSON files

for mySet in partitions:
    count = 0
    for image in mySet['data']:
        src = os.path.join( originalSPEED_dir, 'images/train', image['filename'] )
        dst = os.path.join(mySPEED_dir, 'images', mySet['dir'])
        shutil.copy2(src, dst)
        count += 1
        if (count % 100) == 0:
            print('%s set: %i%%' % (mySet['name'], int(count / len(mySet['data']) * 100)), end='\r')
            # last argument to delete the previously printed line at each progress update

    # Save train,dev,test list of dicts as JSON files
    with open(os.path.join('./sharedData', mySet['dir'] + '.json'), 'w') as fp:
        json.dump(mySet['data'], fp)