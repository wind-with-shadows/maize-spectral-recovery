from random import shuffle
import torch
import torch.nn as nn
import argparse
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.autograd import Variable
import os
from hsi_dataset import TrainDataset, ValidDataset
from architecture import *
from utils import AverageMeter, initialize_logger, save_checkpoint, record_loss, \
    time2file_name, Loss_MRAE, Loss_RMSE, Loss_PSNR
import datetime
import cv2
import numpy as np
import scipy.io as sio


pretrained_model_path=r'' 
method='hscnn_plus'
# model
pretrained_model_path = pretrained_model_path
method = method
model = model_generator(method, pretrained_model_path).cuda()
print('Parameters number is ', sum(param.numel() for param in model.parameters()))

bgr_path=r''
bgr = cv2.imread(bgr_path)
bgr = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
bgr = np.float32(bgr)
bgr = (bgr - bgr.min()) / (bgr.max() - bgr.min())
bgr = np.transpose(bgr, [2, 0, 1])
input=bgr
print(input.shape)
input=torch.FloatTensor(input.reshape(1,input.shape[0],input.shape[1],-1))
input=input.cuda()
output=model(input)
print(output.shape)

#gray
# output=output.detach().cpu().numpy().reshape(31,512,512)
# cv2.imshow('1',output[20,:,:])
# cv2.waitKey(10000)

#fake-RGB
# print(output[0,2,:,:].shape)
# output=torch.cat([output[:,7,:,:],output[:,16,:,:],output[:,29,:,:]])
# output=output.detach().cpu().numpy().reshape(3,512,512)
# print(output.shape)
# output=output.transpose(1,2,0)
# cv2.imshow('1',output)
# cv2.waitKey(10000)

#save
output=output.detach().cpu().numpy().reshape(31,512,512)
output=output.transpose(1,2,0)
print(output.shape)
sio.savemat('',{'X':output})