import os
import warnings
from argparse import ArgumentParser

import torch
import torch.nn as nn
import torch.optim as optim
from ignite.contrib.handlers import ProgressBar
from ignite.contrib.handlers.tensorboard_logger import *
from ignite.engine import Events, Engine
from ignite.metrics import RunningAverage, Loss, ConfusionMatrix, IoU
from ignite.utils import convert_tensor
from torch.utils.data import DataLoader

from weathernet.datasets import DENSE, Normalize, Compose, RandomHorizontalFlip
from weathernet.datasets.transforms import ToTensor
from weathernet.model import WeatherNet
from weathernet.utils import save

import numpy as np
import h5py


def get_data_loaders(data_dir, batch_size=None, num_workers=None):
    normalize = Normalize(mean=DENSE.mean(), std=DENSE.std())
    transforms = Compose([
        RandomHorizontalFlip(),
        ToTensor(),
        normalize
    ])

    test_loader = DataLoader(DENSE(root=data_dir, split='test', transform=transforms),
                              batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)

    return test_loader

def run(args):

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    num_classes = DENSE.num_classes()
    model = WeatherNet(num_classes)

    device_count = torch.cuda.device_count()
    if device_count > 1:
        print("Using %d GPU(s)" % device_count)
        model = nn.DataParallel(model)
        args.batch_size = device_count * args.batch_size
        args.val_batch_size = device_count * args.val_batch_size

    model = model.to(device)

    test_loader = get_data_loaders(args.dataset_dir, args.batch_size, args.num_workers)


    #Init and Load in model
    model.load_state_dict(torch.load('checkpoints/model_epoch7_mIoU=75.5.pth'))

    print(test_loader.__len__())

    #Evaluate data

    testnum = 0

    model.eval()
    for data in test_loader:
        #print(len(data))
        #print(data[0].shape)
        #print(data[1].shape, data[2].shape, data[3].shape)
        with torch.no_grad():

            #Get predictions
            pred = model(data[0].cuda(), data[1].cuda())
            #print(pred.shape)
            pred = pred[[:], 1:, :, :]
            pred = torch.argmax(pred, dim=1, keepdim=True)
            #print(pred.shape)

            #print(data[0].shape, data[1].shape)


            #save predictions back into hdf5
            distance_1 = torch.squeeze(data[0]).cpu().numpy()
            reflectivity_1 = torch.squeeze(data[1]).cpu().numpy()
            label_1 = torch.squeeze(pred).cpu().numpy()
            sensorX = torch.squeeze(data[3]).numpy()
            sensorY = torch.squeeze(data[4]).numpy()
            sensorZ = torch.squeeze(data[5]).numpy()

            label_dict= {0:0, 1:100, 2:101, 3:102}
            label_1 = np.vectorize(label_dict.get)(label_1)


            #<KeysViewHDF5 ['distance_m_1', 'intensity_1', 'labels_1', 'sensorX_1', 'sensorY_1', 'sensorZ_1']>
            hf = h5py.File('processed_data/test' + str(testnum) + '.hdf5', 'w')
            hf.create_dataset('distance_m_1', data=distance_1)
            hf.create_dataset('intensity_1', data=reflectivity_1)
            hf.create_dataset('labels_1', data=label_1)
            hf.create_dataset('sensorX_1', data=sensorX)
            hf.create_dataset('sensorY_1', data=sensorY)
            hf.create_dataset('sensorZ_1', data=sensorZ)
            hf.close()
            testnum = testnum + 1
            #print(label_1)            


        #exit()



if __name__ == '__main__':
    parser = ArgumentParser('WeatherNet with PyTorch')
    parser.add_argument('--batch-size', type=int, default=10,
                        help='input batch size for training')
    parser.add_argument('--val-batch-size', type=int, default=10,
                        help='input batch size for validation')
    parser.add_argument('--num-workers', type=int, default=16,
                        help='number of workers')
    parser.add_argument('--output-dir', default='checkpoints',
                        help='directory to save model checkpoints')
    parser.add_argument("--dataset-dir", type=str, default="data/DENSE",
                        help="location of the dataset")

    run(parser.parse_args())
