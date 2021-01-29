import os, sys
import time
import argparse
import shutil
import numpy as np

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data

# import video_transforms
# import models
# import datasets

import traceback
import logging
from functools import partial
import distiller
from distiller.models import create_model
import distiller.apputils.image_classifier as classifier
import distiller.apputils as apputils
import parser
import os
import numpy as np
from ptq_lapq import image_classifier_ptq_lapq

# Logger handle
msglogger = logging.getLogger()

# model_names = sorted(name for name in models.__dict__
#     if name.islower() and not name.startswith("__")
#     and callable(models.__dict__[name]))

# dataset_names = sorted(name for name in datasets.__all__)

# parser = argparse.ArgumentParser(description='PyTorch Two-Stream Action Recognition')
# parser.add_argument('data', metavar='DIR', default='/home/jiaming/action_data/ucf101/jpegs_256',
#                     help='path to dataset')
# parser.add_argument('--settings', metavar='DIR', default='./datasets/settings',
#                     help='path to datset setting files')
# parser.add_argument('--modality', '-m', metavar='MODALITY', default='rgb',
#                     choices=["rgb", "tvl1_flow", "lk_flow"],
#                     help='modality: rgb | flow')
# parser.add_argument('--dataset', '-d', default='ucf101',
#                     choices=["ucf101", "hmdb51"],
#                     help='dataset: ucf101 | hmdb51')
# parser.add_argument('--arch', '-a', metavar='ARCH', default='rgb_resnet152',
#                     choices=model_names,
#                     help='model architecture: ' +
#                         ' | '.join(model_names) +
#                         ' (default: rgb_vgg16)')
# parser.add_argument('-s', '--split', default=1, type=int, metavar='S',
#                     help='which split of data to work on (default: 1)')
# parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
#                     help='number of data loading workers (default: 4)')
# parser.add_argument('--epochs', default=250, type=int, metavar='N',
#                     help='number of total epochs to run')
# parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
#                     help='manual epoch number (useful on restarts)')
# parser.add_argument('-b', '--batch-size', default=25, type=int,
#                     metavar='N', help='mini-batch size (default: 50)')
# parser.add_argument('--iter-size', default=5, type=int,
#                     metavar='I', help='iter size as in Caffe to reduce memory usage (default: 5)')
# parser.add_argument('--new_length', default=1, type=int,
#                     metavar='N', help='length of sampled video frames (default: 1)')
# parser.add_argument('--gpu', default=0, type=int,
#                     metavar='N', help='executing GPU number')
# parser.add_argument('--new_width', default=340, type=int,
#                     metavar='N', help='resize width (default: 340)')
# parser.add_argument('--new_height', default=256, type=int,
#                     metavar='N', help='resize height (default: 256)')
# parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
#                     metavar='LR', help='initial learning rate')
# parser.add_argument('--lr_steps', default=[100, 200], type=float, nargs="+",
#                     metavar='LRSteps', help='epochs to decay learning rate by 10')
# parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
#                     help='momentum')
# parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
#                     metavar='W', help='weight decay (default: 5e-4)')
# parser.add_argument('--print-freq', default=50, type=int,
#                     metavar='N', help='print frequency (default: 50)')
# parser.add_argument('--save-freq', default=25, type=int,
#                     metavar='N', help='save frequency (default: 25)')
# parser.add_argument('--resume', default='./checkpoints', type=str, metavar='PATH',
#                     help='path to latest checkpoint (default: none)')
# parser.add_argument('--save-path', default='./checkpoints', type=str, metavar='PATH',
#                     help='path to latest checkpoint (default: none)')
# parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
#                     help='evaluate model on validation set')


def main():
    # Parse arguments
    args = parser.add_cmdline_args(classifier.init_classifier_compression_arg_parser(True)).parse_args()
    app = ActionRecognizerCompressor(args, script_dir=os.path.dirname(__file__))
    if app.handle_subapps():
        return
    init_knowledge_distillation(app.args, app.model, app.compression_scheduler)
    app.run_training_loop()
    # Finally run results on the test set
    return app.test()

def handle_subapps(model, criterion, optimizer, compression_scheduler, pylogger, args):
    def load_test_data(args):
        test_loader = classifier.load_data(args, load_train=False, load_val=False, load_test=True)
        return test_loader

    do_exit = False
    if args.greedy:
        greedy(model, criterion, optimizer, pylogger, args)
        do_exit = True
    elif args.summary:
        # This sample application can be invoked to produce various summary reports
        for summary in args.summary:
            distiller.model_summary(model, summary, args.dataset)
        do_exit = True
    elif args.export_onnx is not None:
        distiller.export_img_classifier_to_onnx(model,
                                                os.path.join(msglogger.logdir, args.export_onnx),
                                                args.dataset, add_softmax=True, verbose=False)
        do_exit = True
    elif args.qe_calibration and not (args.evaluate and args.quantize_eval):
        classifier.acts_quant_stats_collection(model, criterion, pylogger, args, save_to_file=True)
        do_exit = True
    elif args.activation_histograms:
        classifier.acts_histogram_collection(model, criterion, pylogger, args)
        do_exit = True
    elif args.sensitivity is not None:
        test_loader = load_test_data(args)
        sensitivities = np.arange(*args.sensitivity_range)
        sensitivity_analysis(model, criterion, test_loader, pylogger, args, sensitivities)
        do_exit = True
    elif args.evaluate:
        if args.quantize_eval and args.qe_lapq:
            image_classifier_ptq_lapq(model, criterion, pylogger, args)
        else:
            test_loader = load_test_data(args)
            classifier.evaluate_model(test_loader, model, criterion, pylogger,
                classifier.create_activation_stats_collectors(model, *args.activation_stats),
                args, scheduler=compression_scheduler)
        do_exit = True
    elif args.thinnify:
        assert args.resumed_checkpoint_path is not None, \
            "You must use --resume-from to provide a checkpoint file to thinnify"
        distiller.contract_model(model, compression_scheduler.zeros_mask_dict, args.arch, args.dataset, optimizer=None)
        apputils.save_checkpoint(0, args.arch, model, optimizer=None, scheduler=compression_scheduler,
                                 name="{}_thinned".format(args.resumed_checkpoint_path.replace(".pth.tar", "")),
                                 dir=msglogger.logdir)
        msglogger.info("Note: if your model collapsed to random inference, you may want to fine-tune")
        do_exit = True
    return do_exit


def init_knowledge_distillation(args, model, compression_scheduler):
    args.kd_policy = None
    if args.kd_teacher:
        teacher = create_model(args.kd_pretrained, args.dataset, args.kd_teacher, device_ids=args.gpus)
        if args.kd_resume:
            teacher = apputils.load_lean_checkpoint(teacher, args.kd_resume)
        dlw = distiller.DistillationLossWeights(args.kd_distill_wt, args.kd_student_wt, args.kd_teacher_wt)
        args.kd_policy = distiller.KnowledgeDistillationPolicy(model, teacher, args.kd_temp, dlw)
        compression_scheduler.add_policy(args.kd_policy, starting_epoch=args.kd_start_epoch, ending_epoch=args.epochs,
                                         frequency=1)
        msglogger.info('\nStudent-Teacher knowledge distillation enabled:')
        msglogger.info('\tTeacher Model: %s', args.kd_teacher)
        msglogger.info('\tTemperature: %s', args.kd_temp)
        msglogger.info('\tLoss Weights (distillation | student | teacher): %s',
                       ' | '.join(['{:.2f}'.format(val) for val in dlw]))
        msglogger.info('\tStarting from Epoch: %s', args.kd_start_epoch)


def early_exit_init(args):
    if not args.earlyexit_thresholds:
        return
    args.num_exits = len(args.earlyexit_thresholds) + 1
    args.loss_exits = [0] * args.num_exits
    args.losses_exits = []
    args.exiterrors = []
    msglogger.info('=> using early-exit threshold values of %s', args.earlyexit_thresholds)


class ActionRecognizerCompressor(classifier.ClassifierCompressor):
    def __init__(self, args, script_dir):
        super().__init__(args, script_dir)
        early_exit_init(self.args)
        # Save the randomly-initialized model before training (useful for lottery-ticket method)
        if args.save_untrained_model:
            ckpt_name = '_'.join((self.args.name or "", "untrained"))
            apputils.save_checkpoint(0, self.args.arch, self.model,
                                     name=ckpt_name, dir=msglogger.logdir)


    def handle_subapps(self):
        return handle_subapps(self.model, self.criterion, self.optimizer,
                              self.compression_scheduler, self.pylogger, self.args)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n-- KeyboardInterrupt --")
    except Exception as e:
        if msglogger is not None:
            # We catch unhandled exceptions here in order to log them to the log file
            # However, using the msglogger as-is to do that means we get the trace twice in stdout - once from the
            # logging operation and once from re-raising the exception. So we remove the stdout logging handler
            # before logging the exception
            handlers_bak = msglogger.handlers
            msglogger.handlers = [h for h in msglogger.handlers if type(h) != logging.StreamHandler]
            msglogger.error(traceback.format_exc())
            msglogger.handlers = handlers_bak
        raise
    finally:
        if msglogger is not None and hasattr(msglogger, 'log_filename'):
            msglogger.info('')
            msglogger.info('Log file for this run: ' + os.path.realpath(msglogger.log_filename))
