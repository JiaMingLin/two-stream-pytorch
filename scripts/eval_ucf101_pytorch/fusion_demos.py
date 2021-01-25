#!/usr/bin/env python

import os
import sys
import collections
import numpy as np
import cv2
import math
import random
import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets

sys.path.insert(0, "../../")
import models

from VideoTemporalPrediction import VideoTemporalPrediction
from VideoSpatialPrediction import VideoSpatialPrediction

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]="1"

rgb_clip_root = '/home/jiaming/action_data/ucf101/jpegs_256'
flow_clip_root = '/home/jiaming/action_data/ucf101/ucf101_LK'


def softmax(x):
    y = [math.exp(k) for k in x]
    sum_y = math.fsum(y)
    z = [k/sum_y for k in y]

    return z

def main():
    idx2class = {}
    with open ('../../datasets/ucf101_splits/classInd.txt', 'r') as f:
        for lines in f.readlines():
            classNo, className = lines.strip('\n').split(' ')
            idx2class[classNo] = className
    
    amountList = [0]*len(idx2class)
    correctList = [0]*len(idx2class)
    
    report_file = "./fusion_model_LK256_t2/fusion_validation_report.txt"
    f_re = open(report_file,'w', buffering = 1)

    Tmodel_path = './fusion_model_LK256_t2/temporal.pth.tar'
    Smodel_path = './fusion_model_LK256_t2/spatial.pth.tar'
    print('temporal_model_path {}'.format(os.path.abspath(Tmodel_path)), file = f_re)
    print('spatial_model_path {}'.format(os.path.abspath(Smodel_path)), file = f_re)
    Tmodel_use = 'flow_mobilenet' #flow_mobilenet, flow_vgg16
    Smodel_use = 'rgb_mobilenet' #rgb_mobilenet, rgb_vgg16
    print('Temporal model backbone: {}'.format(Tmodel_use), file = f_re)
    print('Spatial model backbone: {}'.format(Smodel_use), file = f_re)

    start_frame = 0
    num_categories = 101

    model_start_time = time.time()
    temporal_net = models.__dict__[Tmodel_use](pretrained=False, num_classes=101)
    temporal_net.load_state_dict(torch.load(Tmodel_path)['state_dict'])
    temporal_net.cuda()
    temporal_net.eval()
    model_end_time = time.time()
    model_time = model_end_time - model_start_time
    print("Action recognition temporal model is loaded in %4.4f seconds." % (model_time))
    print("Action recognition temporal model is loaded in %4.4f seconds." % (model_time), file = f_re)

    model_start_time = time.time()
    spatial_net = models.__dict__[Smodel_use](pretrained=False, num_classes=101)
    spatial_net.load_state_dict(torch.load(Smodel_path)['state_dict'])
    spatial_net.cuda()
    spatial_net.eval()
    model_end_time = time.time()
    model_time = model_end_time - model_start_time
    print("Action recognition spatial model is loaded in %4.4f seconds." % (model_time))
    print("Action recognition spatial model is loaded in %4.4f seconds." % (model_time), file = f_re)

    Tval_file = "../../datasets/settings/ucf101/val_lk_flow_split1.txt"
    Sval_file = "../../datasets/settings/ucf101/val_rgb_split1.txt"
    print('temporal validation file = {}'.format(os.path.abspath(Tval_file)), file = f_re)
    print('spatial validation file = {}'.format(os.path.abspath(Sval_file)), file = f_re)
    f_Tval = open(Tval_file, "r")
    f_Sval = open(Sval_file, "r")
    val_Tlist = f_Tval.readlines()
    val_Slist = f_Sval.readlines()
    print("we got %d test videos" % len(val_Tlist))
    print("we got %d test videos" % len(val_Tlist), file = f_re)

    line_id = 1
    match_count = 0
    spatial_result_list = []
    temporal_result_list = []
    fusion_result_list = []
    ground_truth_list = []

    print("\nDetail Prediction:", file = f_re)
    for lineT, lineS in zip(val_Tlist, val_Slist):
        line_info = lineT.split(" ")
        clip_Tpath = os.path.join(flow_clip_root, line_info[0])
        clip_Spath = os.path.join(rgb_clip_root, lineS.split(" ")[0])
        
        input_video_label = int(line_info[1]) - 1

        temporal_prediction = VideoTemporalPrediction(
                clip_Tpath,
                temporal_net,
                num_categories,
                start_frame)

        spatial_prediction = VideoSpatialPrediction(
                clip_Spath,
                spatial_net,
                num_categories,
                start_frame)

        avg_temporal_pred_fc8 = np.mean(temporal_prediction, axis=1)
        avg_spatial_pred_fc8 = np.mean(spatial_prediction, axis=1)
        avg_fusionByMean_pred = (avg_temporal_pred_fc8 + avg_spatial_pred_fc8)/2
        temporal_result_list.append(avg_temporal_pred_fc8)
        spatial_result_list.append(avg_spatial_pred_fc8)
        fusion_result_list.append(avg_fusionByMean_pred)
        ground_truth_list.append(input_video_label)
        # avg_spatial_pred = softmax(avg_spatial_pred_fc8)

        spatial_pred_index = np.argmax(avg_spatial_pred_fc8)
        temporal_pred_index = np.argmax(avg_temporal_pred_fc8)
        fusion_pred_index = np.argmax(avg_fusionByMean_pred)
        
        print("Sample %d/%d: GT: %d, Spatial Prediction: %d, Temporal Prediction: %d, Fusion Prediction: %d" % (line_id, len(val_Tlist), input_video_label, spatial_pred_index, temporal_pred_index, fusion_pred_index))
        print("Sample %d/%d: GT: %d, Spatial Prediction: %d, Temporal Prediction: %d, Fusion Prediction: %d" % (line_id, len(val_Tlist), input_video_label, spatial_pred_index, temporal_pred_index, fusion_pred_index), file = f_re )
        amountList[input_video_label] = amountList[input_video_label] + 1
        if fusion_pred_index == input_video_label:
            match_count += 1
            correctList[input_video_label] = correctList[input_video_label] + 1
        line_id += 1

    print(match_count)
    print(len(val_Tlist))
    print("Accuracy {}/{}".format(match_count, len(val_Tlist)), file = f_re)
    print("Accuracy is %4.4f" % (float(match_count)/len(val_Tlist)))
    print("Accuracy is %4.4f" % (float(match_count)/len(val_Tlist)), file = f_re)
    
    print("\nPrediction Distribution:", file = f_re)
    for idx in range(len(idx2class)):
        if ( amountList[idx] !=0 ):
            print('{:<5} {:<20} {:>8} / {:<8} = {:4.4f}'.format(idx, idx2class[str(idx+1)], correctList[idx], amountList[idx], float(correctList[idx])/amountList[idx]))
            print('{:<5} {:<20} {:>8} / {:<8} = {:4.4f}'.format(idx, idx2class[str(idx+1)], correctList[idx], amountList[idx], float(correctList[idx])/amountList[idx]), file = f_re)
        elif ( amountList[idx] == 0 ):
            print('{:<5} {:<20} {:>8} / {:<8} = {:4.4f}'.format(idx, idx2class[str(idx+1)], correctList[idx], amountList[idx], 0))
            print('{:<5} {:<20} {:>8} / {:<8} = {:4.4f}'.format(idx, idx2class[str(idx+1)], correctList[idx], amountList[idx], 0), file = f_re)

    np.savez('{}/ucf101_S{}T{}_fusion_result.npz'.format(os.path.abspath(report_file+'/../'),Smodel_use, Tmodel_use), correctList=np.array(correctList), amountList=np.array(amountList), spatialresultList=np.array(spatial_result_list), temporalresultList=np.array(temporal_result_list), fusionresultList=np.array(fusion_result_list), groundtruthList=np.array(ground_truth_list))

    f_re.close()

if __name__ == "__main__":
    main()
