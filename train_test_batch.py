
import os
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt


## da_list = [2, 3, 4, 5]
da_list = [2, 3, 4, 5]

ec = 0

epochs = 400

for da in da_list:

    # cmd_str = f'nohup python CNNPestes.py --ds CropPestAndDiseaseDetection --arch alexnet --optim none ' + \
    #           f'--bs 128 --lr 0.00001 --ep {epochs} --optimizer Adam --ft --da {da} --ec {ec} --es ' + \
    #           f'--patience 21 --num_workers 0'
    # ec = ec + 1
    # os.system(cmd_str)

    cmd_str = f'nohup python CNNPestes.py --ds CropPestAndDiseaseDetection --arch efficientnet_b4 --optim none ' + \
              f'--bs 16 --lr 0.0001 --ep {epochs} --optimizer Adam --ft --da {da} --ec {ec} --es ' + \
              f'--patience 21 --num_workers 0'
    ec = ec + 1
    os.system(cmd_str)

    # cmd_str = f'nohup python CNNPestes.py --ds CropPestAndDiseaseDetection --arch resnet50 --optim none ' + \
    #           f'--bs 16 --lr 0.00001 --ep {epochs} --optimizer Adam --ft --da {da} --ec {ec} --es ' + \
    #           f'--patience 21 --num_workers 0'
    # ec = ec + 1
    # os.system(cmd_str)

    cmd_str = f'nohup python CNNPestes.py --ds CropPestAndDiseaseDetection --arch vit --optim none ' + \
              f'--bs 32 --lr 0.00001 --ep {epochs} --optimizer Adam --ft --da {da} --ec {ec} --es ' + \
              f'--patience 21 --num_workers 0'
    
    ec = ec + 1
    os.system(cmd_str)

print('\nDone! (train_test_batch)')


