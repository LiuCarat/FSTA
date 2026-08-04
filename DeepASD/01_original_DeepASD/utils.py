import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import time

import numpy as np
import torch


def one_hot(x, class_count):
    print(x)
    return torch.eye(class_count)[x,:]

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.trunc_normal_(m.weight, std=0.01)
        nn.init.trunc_normal_(m.bias, std=0.01)


def computer_class_mean(xs_1, xs_2, xs_3, xt, xs_1_label, xs_2_label, xs_3_label, xt_label, class_number):


   #-------------------------------------------#
    class_mean_xs_1_list = []
    class_mean_xs_2_list = []
    class_mean_xs_3_list = []
    class_mean_xt_list = []

    for c in range(class_number):        
        idx_xs_1_c = torch.nonzero(xs_1_label == c).squeeze()
        idx_xs_2_c = torch.nonzero(xs_2_label == c).squeeze()
        idx_xs_3_c = torch.nonzero(xs_3_label == c).squeeze()
        idx_xt_c = torch.nonzero(xt_label == c).squeeze()

        xs_1_c = torch.index_select(xs_1, 0, idx_xs_1_c)
        xs_2_c = torch.index_select(xs_2, 0, idx_xs_2_c)
        xs_3_c = torch.index_select(xs_3, 0, idx_xs_3_c)
        xt_c = torch.index_select(xt, 0, idx_xt_c)
        
        class_mean_xs_1_list.append(torch.mean(xs_1_c, 0))
        class_mean_xs_2_list.append(torch.mean(xs_2_c, 0))
        class_mean_xs_3_list.append(torch.mean(xs_3_c, 0))
        class_mean_xt_list.append(torch.mean(xt_c, 0))


    class_mean_xs_1 = torch.stack(class_mean_xs_1_list)
    class_mean_xs_2 = torch.stack(class_mean_xs_2_list)
    class_mean_xs_3 = torch.stack(class_mean_xs_3_list)
    class_mean_xt = torch.stack(class_mean_xt_list)
    #-----------------------------------------------------------#
    return class_mean_xs_1, class_mean_xs_2, class_mean_xs_3, class_mean_xt



def get_delta(class_mean_xs_1, class_mean_xs_2, class_mean_xs_3, class_mean_xt):
    delta_xs_1 = torch.mean(torch.sum(torch.square(class_mean_xs_1-class_mean_xt), 1))
    delta_xs_2 = torch.mean(torch.sum(torch.square(class_mean_xs_2-class_mean_xt), 1))
    delta_xs_3 = torch.mean(torch.sum(torch.square(class_mean_xs_3-class_mean_xt), 1))

    delta_3 = 0.5 * (torch.div(torch.exp(delta_xs_1), 1+torch.exp(delta_xs_1)) + torch.div(torch.exp(delta_xs_2),1+torch.exp(delta_xs_2)))
    delta_2 = 0.5 * (torch.div(torch.exp(delta_xs_1), 1+torch.exp(delta_xs_1)) + torch.div(torch.exp(delta_xs_3),1+torch.exp(delta_xs_3)))
    delta_1 = 0.5 * (torch.div(torch.exp(delta_xs_2), 1+torch.exp(delta_xs_2)) + torch.div(torch.exp(delta_xs_3),1+torch.exp(delta_xs_3)))
    return delta_1, delta_2, delta_3, delta_xs_1, delta_xs_2, delta_xs_3



def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

        
def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


class Saver():
    def __init__(self, args, logfilename):

        self.args = args

        self.save_file = os.path.join(args.log_path, 'log', logfilename)
        if not os.path.exists(self.save_file):
            mkdirs(self.save_file) 

        self.log_name = os.path.join(self.save_file, 'loss_log.txt')

        with open(self.log_name, "a") as log_file:
            now = time.strftime("%c")
            log_file.write('================ Training Loss (%s) ================\n' % now)


        # self.imgsave_dir = os.path.join(self.save_file, 'images')
        # if not os.path.exists(self.imgsave_dir):
        # # print('create image directory %s...' % self.imgsave_dir)
        #     mkdirs(self.imgsave_dir)

    def print_current_errors(self, epoch, errors):
        message = '(epoch: %d) ' % (epoch)
        for k, v in errors.items():
            message += '%s: %.3f ' % (k, v)

        print(message)
        with open(self.log_name, "a") as log_file:
            log_file.write('%s\n' % message)
    # def print_current_errors(self, epoch, i, errors):
    #     message = '(epoch: %d, iters: %d) ' % (epoch, i)
    #     for k, v in errors.items():
    #         message += '%s: %.3f ' % (k, v)

    #     print(message)
    #     with open(self.log_name, "a") as log_file:
    #         log_file.write('%s\n' % message)


        # save to the disk
    def print_config(self):
        opt = vars(self.args)
        file_name = os.path.join(self.save_file, 'opt.txt')
        with open(file_name, 'wt') as opt_file:
            opt_file.write('------------ Options -------------\n')
            for k, v in sorted(opt.items()):
                opt_file.write('%s: %s\n' % (str(k), str(v)))
            opt_file.write('-------------- End ----------------\n')

       
def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    D = torch.sum(adj, -1)
    d_inv_sqrt = torch.pow(D, -0.5)
    d_inv_sqrt = torch.diagflat(d_inv_sqrt)
    adj = torch.mm(d_inv_sqrt, torch.mm(adj, d_inv_sqrt))
    return adj
