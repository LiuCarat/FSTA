import numpy as np
import pandas as pd
import json
import torch
import torch.optim as optim
import torch.nn as nn
import argparse
import logging
import os
import copy
import datetime
import random
from sklearn.model_selection import StratifiedKFold, train_test_split
from tensorboardX import SummaryWriter

from train import *
from utils import *


def get_args():
    TIMESTAMP="{0:%Y-%m-%dT%H-%M-%S/}".format(datetime.datetime.now())
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='DeepASD', help='neural network used in training')
    parser.add_argument('--dataset', type=str, default='ABIDE_B')
    parser.add_argument('--datadir', type=str, required=False, default="../data/", help="Data directory")
    parser.add_argument('--lr_G', type=float, default=0.004)
    parser.add_argument('--lr_D', type=float, default=0.001)
    parser.add_argument('--lr_GC', type=float, default=0.01)
    parser.add_argument('--lr_MP', type=float, default=0.0005)
    parser.add_argument('--epoch', type=int, default=500)
    parser.add_argument('--beta', type=float, default=0.03, help='control adversarial loss')
    parser.add_argument('--tau', type=float, default=0.004, help='control regularization term, cannot be an integer')
    parser.add_argument('--GC_mode', type=str, default='weighted-cosine', help='weighted-cosine/adaptive-learning')
    parser.add_argument('--th', type=float, default=0.9, help='threshold of weighted cosine')
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--d', type=int, default=256, help='the dimension of common subspace')
    parser.add_argument('--d_hid', type=int, default=512, help='the dimension of hidden vectors')
    parser.add_argument('--d_dis_hidden', type=int, default=128, help='the dimension of hidden vectors in discriminator')
    parser.add_argument('--init_seed', type=int, default=0, help="Random seed")
    parser.add_argument('--log_path', type=str, default='./logs/Train_'+TIMESTAMP)
    parser.add_argument('--result_path', type=str, default='./results/Test_'+TIMESTAMP)
    parser.add_argument('--save_graph', type=int, default=1, help='save graph embedding')
    parser.add_argument('--save_roc', type=int, default=1, help='save roc curve')
    parser.add_argument('--iter_num', type=int, default=10, help="Cross-Validation fold")
    parser.add_argument('--wait_num', type=int, default=50, help='save best model in validate strategy')
    parser.add_argument('--min_epoch', type=int, default=150, help='minimun epoch num to save')
    parser.add_argument('--with_Loss_l', type=int, default=0, help='whether contains Loss alignment')
    parser.add_argument('--mu', type=float, default=0.0, help='mu')
    parser.add_argument('--std', type=float, default=0.95, help='std')
    parser.add_argument('--K', type=float, default=5, help='K times')
    args = parser.parse_args()
    return args

def set_rng_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


if __name__ == '__main__':
    args = get_args()

    savefilename = os.path.join(args.dataset+'') 
    summary_writer = SummaryWriter(os.path.join(args.log_path, 'log', savefilename))
    saver = Saver(args, savefilename)
    saver.print_config()

    seed = args.init_seed
    set_rng_seed(seed)

    res_df = pd.DataFrame(data=None, columns=['fold', 'test_acc', 'test_auc', 'test_sen', 'test_spe'])
    if args.save_roc:
        roc_df = pd.DataFrame(data=None, columns=['fold', 'label', 'prob_0', 'prob_1'])
        fold_list = []
        y_list = []
        prob_0_list = []
        prob_1_list = []

    if args.dataset == 'ABIDE_A':
        # load dataset
        path = args.datadir + args.dataset + '/'
        modal_feat_dict = np.load(path + 'modal_feat_dict.npy', allow_pickle=True).item()
        data = pd.read_csv(path + 'processed_standard_data.csv').values
        print('data shape: ', data.shape)
        input_data_dims = []
        for i in modal_feat_dict.keys():
            input_data_dims.append(len(modal_feat_dict[i]))
        print('Modal dims ', input_data_dims)
        input_data = data[:,:-1]
        label = data[:,-1]-1

        modal_weight_list = []


        skf = StratifiedKFold(n_splits=args.iter_num, random_state=0, shuffle=True)
        clk = 0
        for train_index, test_index in skf.split(input_data, label):
            # valid_index = test_index
            train_index, valid_index = train_test_split(train_index, test_size=0.1)
            clk += 1
            modal_weight = None
            if args.model == 'DeepASD':
                if clk > 1:
                    args.save_graph = 0
                test_acc, test_auc, test_sen, test_spe, prob_0, prob_1, modal_weight = train_net_DeepASD(input_data, label, train_index, valid_index, test_index, input_data_dims, args, summary_writer, saver)
            elif args.model == 'adv':
                test_acc, test_auc, test_sen, test_spe = train_net_adv(input_data, label, train_index, test_index, valid_index, input_data_dims, args, summary_writer, saver)
            elif args.model == 'mlp':
                test_acc, test_auc, test_sen, test_spe = train_net_mlp(input_data, label, train_index, test_index, valid_index, input_data_dims, args, summary_writer, saver)
            
            res_df.loc[len(res_df.index)] = [clk-1, test_acc, test_auc, test_sen, test_spe]
            modal_weight_list.append(modal_weight)
            print("the accuracy of f(x) is: " + str(test_acc*100))
            print("the auc of f(x) is: " + str(test_auc*100))
            print("the sen of f(x) is: " + str(test_sen*100))
            print("the spe of f(x) is: " + str(test_spe*100))
            print("modal weight", modal_weight)

            if args.save_roc:
                if test_acc > 0.8:
                    fold_list.extend([str(clk-1)] * len(test_index))
                    y_list.extend(label[test_index])
                    prob_0_list.extend(prob_0)
                    prob_1_list.extend(prob_1)
            torch.cuda.empty_cache()
                    
            
        res_save_path = os.path.join(args.result_path)
        if not os.path.exists(res_save_path):
            mkdirs(res_save_path) 
    
        res_df.to_csv(os.path.join(res_save_path, args.dataset+'_result.csv'), index=False)
        modal_weight_df = pd.DataFrame(modal_weight_list)
        modal_weight_df.to_csv(os.path.join(res_save_path, args.dataset+'_weight.csv'), index=False)


        if args.save_roc:
            roc_df['fold'] = fold_list
            roc_df['label'] = y_list
            roc_df['prob_0'] = prob_0_list
            roc_df['prob_1'] = prob_1_list
            roc_df.to_csv(os.path.join(res_save_path, args.dataset+'_roc.csv'), index=False)
            

    elif args.dataset == 'ABIDE_B':
        # load dataset
        path = args.datadir + args.dataset + '/'
        # data = np.load(path + 'processed_standard_data.npz', allow_pickle=True)
        # input_data_dims = [48, 6670, 12880, 19900]

        data = np.load(path + 'RFE_512_processed_standard_data.npz', allow_pickle=True)
        # input_data_dims = [48, 6670, 12880, 19900]
        # input_data_dims = [48, 256, 256, 256]
        input_data_dims = [48, 512, 512, 512]

        input_data = data["data"]
        label = data["label"]

        skf = StratifiedKFold(n_splits=args.iter_num, random_state=0, shuffle=True)
        clk = 0

        modal_weight_list = []

        for train_index, test_index in skf.split(input_data, label):
            # valid_index = test_index
            train_index, valid_index = train_test_split(train_index, test_size=0.1)
            clk += 1
            modal_weight = None
            if args.model == 'DeepASD':
                if clk > 1:
                    args.save_graph = 0
                test_acc, test_auc, test_sen, test_spe, prob_0, prob_1, modal_weight = train_net_DeepASD(input_data, label, train_index, valid_index, test_index, input_data_dims, args, summary_writer, saver)
            elif args.model == 'adv':
                test_acc, test_auc, test_sen, test_spe = train_net_adv(input_data, label, train_index, test_index, valid_index, input_data_dims, args, summary_writer, saver)
            elif args.model == 'mlp':
                test_acc, test_auc, test_sen, test_spe = train_net_mlp(input_data, label, train_index, test_index, valid_index, input_data_dims, args, summary_writer, saver)
            
            res_df.loc[len(res_df.index)] = [clk-1, test_acc, test_auc, test_sen, test_spe]
            modal_weight_list.append(modal_weight)
            print("the accuracy of f(x) is: " + str(test_acc*100))
            print("the auc of f(x) is: " + str(test_auc*100))
            print("the sen of f(x) is: " + str(test_sen*100))
            print("the spe of f(x) is: " + str(test_spe*100))
            print("modal weight", modal_weight)

            if args.save_roc:
                fold_list.extend([str(clk-1)] * len(test_index))
                y_list.extend(label[test_index])
                prob_0_list.extend(prob_0)
                prob_1_list.extend(prob_1)
                    
            
        res_save_path = os.path.join(args.result_path)
        if not os.path.exists(res_save_path):
            mkdirs(res_save_path) 
    
        res_df.to_csv(os.path.join(res_save_path, args.dataset+'_result.csv'), index=False)
        modal_weight_df = pd.DataFrame(modal_weight_list)
        modal_weight_df.to_csv(os.path.join(res_save_path, args.dataset+'_weight.csv'), index=False)

        if args.save_roc:
            roc_df['fold'] = fold_list
            roc_df['label'] = y_list
            roc_df['prob_0'] = prob_0_list
            roc_df['prob_1'] = prob_1_list
            roc_df.to_csv(os.path.join(res_save_path, args.dataset+'_roc.csv'), index=False)
            
        # # load dataset
        # path = args.datadir + args.dataset + '/'
        # data = np.load(path + 'abide_multi_fmri.npz', allow_pickle=True)
        # input_data_dims = [6670, 12880, 19900]
        # print('Modal dims ', input_data_dims)
        
        # train_index_list = data['train_index_list']
        # valid_index_list = data['valid_index_list']
        # test_index_list = data['test_index_list']
        # input_data_list = data['input_data_list']
        # label_list = data['label_list']
        
        # clk = 0
        # for i in range(args.iter_num):
        #     clk += 1
        #     train_index = train_index_list[i]
        #     valid_index = valid_index_list[i]
        #     test_index = test_index_list[i]
        #     input_data = input_data_list[i]
        #     label = label_list[i]

        #     if args.model == 'DeepASD':
        #         if clk > 1:
        #             args.save_graph = 0
        #         test_acc, test_auc, test_sen, test_spe, prob_0, prob_1 = train_net_DeepASD(input_data, label, train_index, valid_index, test_index, input_data_dims, args, summary_writer, saver)
        #     elif args.model == 'adv':
        #         test_acc, test_auc, test_sen, test_spe = train_net_adv(input_data, label, train_index, valid_index, test_index, input_data_dims, args, summary_writer, saver)
            
        #     res_df.loc[len(res_df.index)] = [clk-1, test_acc, test_auc, test_sen, test_spe]
        #     print("the accuracy of f(x) is: " + str(test_acc*100))
        #     print("the auc of f(x) is: " + str(test_auc*100))
        #     print("the sen of f(x) is: " + str(test_sen*100))
        #     print("the spe of f(x) is: " + str(test_spe*100))

        #     if args.save_roc:
        #         if test_acc > 0.7:
        #             fold_list.extend([str(clk-1)] * len(test_index))
        #             y_list.extend(label[test_index])
        #             prob_0_list.extend(prob_0)
        #             prob_1_list.extend(prob_1)
                    
        # res_save_path = os.path.join(args.result_path)
        # if not os.path.exists(res_save_path):
        #     mkdirs(res_save_path) 
    
        # res_df.to_csv(os.path.join(res_save_path, args.dataset+'_result.csv'), index=False)

        # if args.save_roc:
        #     roc_df['fold'] = fold_list
        #     roc_df['label'] = y_list
        #     roc_df['prob_0'] = prob_0_list
        #     roc_df['prob_1'] = prob_1_list
        #     roc_df.to_csv(os.path.join(res_save_path, args.dataset+'_roc.csv'), index=False)

    elif args.dataset == 'VOC':
        path = args.datadir + args.dataset + '/'
        data = np.load(path + 'voc.npz', allow_pickle=True)
        #====使用image作为主要的，text作为次要的=====#
        view_1 = data['view_1']
        view_0 = data['view_0']
        label = data['labels']
        input_data = np.hstack((view_1, view_0))
        input_data_dims = [399, 512]
        skf = StratifiedKFold(n_splits=args.iter_num, random_state=0, shuffle=True)
        clk = 0
        for train_index, test_index in skf.split(input_data, label):
            valid_index = test_index
            clk += 1
            if args.model == 'DeepASD':
                if clk > 1:
                    args.save_graph = 0
                test_acc, test_auc, test_sen, test_spe, prob_0, prob_1 = train_net_DeepASD(input_data, label, train_index, valid_index, test_index, input_data_dims, args, summary_writer, saver)
            elif args.model == 'adv':
                test_acc, test_auc, test_sen, test_spe = train_net_adv(input_data, label, train_index, test_index, valid_index, input_data_dims, args, summary_writer, saver)
            elif args.model == "mlp":
                test_acc, test_auc, test_sen, test_spe = train_net_mlp(input_data, label, train_index, test_index, valid_index, input_data_dims, args, summary_writer, saver)
            
            
            res_df.loc[len(res_df.index)] = [clk-1, test_acc, test_auc, test_sen, test_spe]
            print("the accuracy of f(x) is: " + str(test_acc*100))
            print("the auc of f(x) is: " + str(test_auc*100))
            print("the sen of f(x) is: " + str(test_sen*100))
            print("the spe of f(x) is: " + str(test_spe*100))

            if args.save_roc:
                if test_acc > 0.1:
                    fold_list.extend([str(clk-1)] * len(test_index))
                    y_list.extend(label[test_index])
                    prob_0_list.extend(prob_0)
                    prob_1_list.extend(prob_1)
                    
        res_save_path = os.path.join(args.result_path)
        if not os.path.exists(res_save_path):
            mkdirs(res_save_path) 
    
        res_df.to_csv(os.path.join(res_save_path, args.dataset+'_result.csv'), index=False)

        if args.save_roc:
            roc_df['fold'] = fold_list
            roc_df['label'] = y_list
            roc_df['prob_0'] = prob_0_list
            roc_df['prob_1'] = prob_1_list
            roc_df.to_csv(os.path.join(res_save_path, args.dataset+'_roc.csv'), index=False)
    
    elif args.dataset == 'compare_A':
        # load dataset
        path = args.datadir + "ABIDE_A" + '/'
        modal_feat_dict = np.load(path + 'modal_feat_dict.npy', allow_pickle=True).item()
        data = pd.read_csv(path + 'processed_standard_data.csv').values
        print('data shape: ', data.shape)
        input_data_dims = []
        for i in modal_feat_dict.keys():
            input_data_dims.append(len(modal_feat_dict[i]))
        print('Modal dims ', input_data_dims)
        input_data = data[:,:-1]
        
        tmp_dim = 0
        for idx, dim in enumerate(input_data_dims):
            if idx == 0:
                PHENO_Exp = input_data[:, tmp_dim : tmp_dim + dim]
            elif idx == 1:
                ANAT_Exp = input_data[:, tmp_dim : tmp_dim + dim]
            elif idx == 2:
                FUNC_Exp = input_data[:, tmp_dim : tmp_dim + dim]
            else:
                FMRI_Exp = input_data[:, tmp_dim : tmp_dim + dim]
            tmp_dim += dim
        
        split_dims = {
            'PHENO': input_data_dims[0],
            'ANAT': input_data_dims[1],
            'FUNC': input_data_dims[2],
            'FMRI': input_data_dims[3],
        }
        
        split_exp = {
            'PHENO': PHENO_Exp,
            'ANAT': ANAT_Exp,
            'FUNC': FUNC_Exp,
            'FMRI': FMRI_Exp,
        }
        
        compare_method = [
                        'PHENO', 'ANAT', 'FUNC', 'FMRI', \
                        'PHENO+ANAT', 'PHENO+FUNC', 'PHENO+FMRI', \
                        'ANAT+FUNC', 'ANAT+FMRI', 'FUNC+FMRI', \
                        'PHENO+ANAT+FUNC', 'PHENO+ANAT+FMRI', 'PHENO+FUNC+FMRI', 'ANAT+FUNC+FMRI', \
                        'PHENO+ANAT+FUNC+FMRI']
        
        comp_df = pd.DataFrame(data=None, columns=['comb_method', 'test_acc', 'test_auc', 'test_sen', 'test_spe'])
        
        for comb in compare_method:
            if args.save_roc:
                roc_df = pd.DataFrame(data=None, columns=['fold', 'label', 'prob_0', 'prob_1'])
                fold_list = []
                y_list = []
                prob_0_list = []
                prob_1_list = []
            contain_attr = comb.split('+')
            input_data = split_exp[contain_attr[0]]
            input_data_dims = [split_dims[contain_attr[0]]]
            if len(contain_attr) > 1:
                for attr in contain_attr[1:]:
                    input_data = np.hstack((input_data, split_exp[attr]))
                    input_data_dims.append(split_dims[attr])
            
            print(comb, input_data_dims)
            
            args.dataset = comb
            
            label = data[:,-1]-1
            
            res_df = pd.DataFrame(data=None, columns=['fold', 'test_acc', 'test_auc', 'test_sen', 'test_spe'])
            
            skf = StratifiedKFold(n_splits=args.iter_num, random_state=0, shuffle=True)
            clk = 0
            modal_weight_list = []
            for train_index, test_index in skf.split(input_data, label):
                # valid_index = test_index
                train_index, valid_index = train_test_split(train_index, test_size=0.1)
                clk += 1
                modal_weight = None
                if args.model == 'DeepASD':
                    if clk > 1:
                        args.save_graph = 0
                    test_acc, test_auc, test_sen, test_spe, prob_0, prob_1, modal_weight = train_net_DeepASD(input_data, label, train_index, valid_index, test_index, input_data_dims, args, summary_writer, saver)
                elif args.model == 'adv':
                    test_acc, test_auc, test_sen, test_spe = train_net_adv(input_data, label, train_index, valid_index, test_index, input_data_dims, args, summary_writer, saver)
                
                res_df.loc[len(res_df.index)] = [clk-1, test_acc, test_auc, test_sen, test_spe]
                modal_weight_list.append(modal_weight)
                print("the accuracy of f(x) is: " + str(test_acc*100))
                print("the auc of f(x) is: " + str(test_auc*100))
                print("the sen of f(x) is: " + str(test_sen*100))
                print("the spe of f(x) is: " + str(test_spe*100))

                if args.save_roc:
                    # if test_acc > 0.8:
                    fold_list.extend([str(clk-1)] * len(test_index))
                    y_list.extend(label[test_index])
                    prob_0_list.extend(prob_0)
                    prob_1_list.extend(prob_1)
            
            res_save_path = os.path.join(args.result_path)
            if not os.path.exists(res_save_path):
                mkdirs(res_save_path) 
            
            mean_acc = np.round(res_df['test_acc'].mean() * 100, 2)
            std_acc = np.round(res_df['test_acc'].std() * 100, 2) 
            mean_auc = np.round(res_df['test_auc'].mean() * 100, 2)
            std_auc = np.round(res_df['test_auc'].std() * 100, 2)
            mean_sen = np.round(res_df['test_sen'].mean() * 100, 2)
            std_sen = np.round(res_df['test_sen'].std() * 100, 2)
            mean_spe = np.round(res_df['test_spe'].mean() * 100, 2)
            std_spe = np.round(res_df['test_spe'].std() * 100, 2)
            
            comp_df.loc[len(comp_df.index)] = [args.dataset, str(mean_acc)+'±'+str(std_acc), str(mean_auc)+'±'+str(std_auc), str(mean_sen)+'±'+str(std_sen), str(mean_spe)+'±'+str(std_spe)]
    
            res_df.to_csv(os.path.join(res_save_path, args.dataset+'_result.csv'), index=False)
            modal_weight_df = pd.DataFrame(modal_weight_list)
            modal_weight_df.to_csv(os.path.join(res_save_path, args.dataset+'_weight.csv'), index=False)
            comp_df.to_csv(os.path.join(res_save_path, 'compare_A_result.csv'), index=False)

            if args.save_roc:
                roc_df['fold'] = fold_list
                roc_df['label'] = y_list
                roc_df['prob_0'] = prob_0_list
                roc_df['prob_1'] = prob_1_list
                roc_df.to_csv(os.path.join(res_save_path, args.dataset+'_roc.csv'), index=False)
            torch.cuda.empty_cache()
                
    elif args.dataset == 'compare_B':
        # load dataset
        path = args.datadir + "ABIDE_B" + '/'
        data = np.load(path + 'RFE_512_processed_standard_data.npz', allow_pickle=True)
        # input_data_dims = [48, 6670, 12880, 19900]
        # input_data_dims = [48, 256, 256, 256]
        input_data_dims = [48, 512, 512, 512]
        # input_data_dims = [48, 2000, 2000, 2000]
        input_data = data["data"]
        label = data["label"]

        tmp_dim = 0
        for idx, dim in enumerate(input_data_dims):
            if idx == 0:
                PHENO_Exp = input_data[:, tmp_dim : tmp_dim + dim]
            elif idx == 1:
                AAL_Exp = input_data[:, tmp_dim : tmp_dim + dim]
            elif idx == 2:
                DOS_Exp = input_data[:, tmp_dim : tmp_dim + dim]
            elif idx == 3:
                CC200_Exp = input_data[:, tmp_dim : tmp_dim + dim]
            tmp_dim += dim
        
        split_dims = {
            'PHENO': input_data_dims[0],
            'AAL': input_data_dims[1],
            'DOS': input_data_dims[2],
            'CC200': input_data_dims[3],
        }
        
        split_exp = {
            'PHENO': PHENO_Exp,
            'AAL': AAL_Exp,
            'DOS': DOS_Exp,
            'CC200': CC200_Exp,
        }
        
        compare_method = [
                        'PHENO', 'AAL', 'DOS', 'CC200', \
                        'PHENO+AAL', 'PHENO+DOS', 'PHENO+CC200', \
                        'AAL+DOS', 'AAL+CC200', 'DOS+CC200', \
                        'PHENO+AAL+DOS', 'PHENO+AAL+CC200', 'PHENO+DOS+CC200', 'AAL+DOS+CC200', \
                        'PHENO+AAL+DOS+CC200']
        
        comp_df = pd.DataFrame(data=None, columns=['comb_method', 'test_acc', 'test_auc', 'test_sen', 'test_spe'])
        
        for comb in compare_method:
            if args.save_roc:
                roc_df = pd.DataFrame(data=None, columns=['fold', 'label', 'prob_0', 'prob_1'])
                fold_list = []
                y_list = []
                prob_0_list = []
                prob_1_list = []
            contain_attr = comb.split('+')
            input_data = split_exp[contain_attr[0]]
            input_data_dims = [split_dims[contain_attr[0]]]
            if len(contain_attr) > 1:
                for attr in contain_attr[1:]:
                    input_data = np.hstack((input_data, split_exp[attr]))
                    input_data_dims.append(split_dims[attr])
            
            print(comb, input_data_dims)
            
            args.dataset = comb
            
            res_df = pd.DataFrame(data=None, columns=['fold', 'test_acc', 'test_auc', 'test_sen', 'test_spe'])
            
            skf = StratifiedKFold(n_splits=args.iter_num, random_state=0, shuffle=True)
            clk = 0
            modal_weight_list = []
            for train_index, test_index in skf.split(input_data, label):
                train_index, valid_index = train_test_split(train_index, test_size=0.1)
                # valid_index = test_index
                clk += 1
                if args.model == 'DeepASD':
                    if clk > 1:
                        args.save_graph = 0
                    test_acc, test_auc, test_sen, test_spe, prob_0, prob_1, modal_weight = train_net_DeepASD(input_data, label, train_index, valid_index, test_index, input_data_dims, args, summary_writer, saver)
                elif args.model == 'adv':
                    test_acc, test_auc, test_sen, test_spe = train_net_adv(input_data, label, train_index, valid_index, test_index, input_data_dims, args, summary_writer, saver)
                
                res_df.loc[len(res_df.index)] = [clk-1, test_acc, test_auc, test_sen, test_spe]
                modal_weight_list.append(modal_weight)
                print("the accuracy of f(x) is: " + str(test_acc*100))
                print("the auc of f(x) is: " + str(test_auc*100))
                print("the sen of f(x) is: " + str(test_sen*100))
                print("the spe of f(x) is: " + str(test_spe*100))

                if args.save_roc:
                    # if test_acc > 0.8:
                    fold_list.extend([str(clk-1)] * len(test_index))
                    y_list.extend(label[test_index])
                    prob_0_list.extend(prob_0)
                    prob_1_list.extend(prob_1)
            
            res_save_path = os.path.join(args.result_path)
            if not os.path.exists(res_save_path):
                mkdirs(res_save_path) 
            
            mean_acc = np.round(res_df['test_acc'].mean() * 100, 2)
            std_acc = np.round(res_df['test_acc'].std() * 100, 2) 
            mean_auc = np.round(res_df['test_auc'].mean() * 100, 2)
            std_auc = np.round(res_df['test_auc'].std() * 100, 2)
            mean_sen = np.round(res_df['test_sen'].mean() * 100, 2)
            std_sen = np.round(res_df['test_sen'].std() * 100, 2)
            mean_spe = np.round(res_df['test_spe'].mean() * 100, 2)
            std_spe = np.round(res_df['test_spe'].std() * 100, 2)
            
            comp_df.loc[len(comp_df.index)] = [args.dataset, str(mean_acc)+'±'+str(std_acc), str(mean_auc)+'±'+str(std_auc), str(mean_sen)+'±'+str(std_sen), str(mean_spe)+'±'+str(std_spe)]
    
            res_df.to_csv(os.path.join(res_save_path, args.dataset+'_result.csv'), index=False)
            modal_weight_df = pd.DataFrame(modal_weight_list)
            modal_weight_df.to_csv(os.path.join(res_save_path, args.dataset+'_weight.csv'), index=False)
            comp_df.to_csv(os.path.join(res_save_path, 'compare_B_result.csv'), index=False)

            if args.save_roc:
                roc_df['fold'] = fold_list
                roc_df['label'] = y_list
                roc_df['prob_0'] = prob_0_list
                roc_df['prob_1'] = prob_1_list
                roc_df.to_csv(os.path.join(res_save_path, args.dataset+'_roc.csv'), index=False)
            torch.cuda.empty_cache()
        # # load dataset
        # path = args.datadir + "ABIDE_B" + '/'
        # data = np.load(path + 'abide_multi_fmri.npz', allow_pickle=True)
        # input_data_dims = [6670, 12880, 19900]
        # print('Modal dims ', input_data_dims)
        
        # train_index_list = data['train_index_list']
        # valid_index_list = data['valid_index_list']
        # test_index_list = data['test_index_list']
        # input_data_list = data['input_data_list']
        # label_list = data['label_list']
        
        
        # compare_method = [
        #                 'AAL', 'DOS', 'CC200', \
        #                 'AAL+DOS', 'AAL+CC200', 'DOS+CC200']
        
        # comp_df = pd.DataFrame(data=None, columns=['comb_method', 'test_acc', 'test_auc', 'test_sen', 'test_spe'])
        
        
        # for comb in compare_method:
        #     if args.save_roc:
        #         roc_df = pd.DataFrame(data=None, columns=['fold', 'label', 'prob_0', 'prob_1'])
        #         fold_list = []
        #         y_list = []
        #         prob_0_list = []
        #         prob_1_list = []
        #     contain_attr = comb.split('+')
            
        #     res_df = pd.DataFrame(data=None, columns=['fold', 'test_acc', 'test_auc', 'test_sen', 'test_spe'])
            
        #     clk = 0
        #     for i in range(args.iter_num):  
        #         clk += 1
        #         train_index = train_index_list[i]
        #         valid_index = valid_index_list[i]
        #         test_index = test_index_list[i]
        #         input_data = input_data_list[i]
        #         label = label_list[i]
                
                
        #         input_data_dims = [6670, 12880, 19900]
            
        #         tmp_dim = 0
        #         for idx, dim in enumerate(input_data_dims):
        #             if idx == 0:
        #                 AAL_Exp = input_data[:, tmp_dim : tmp_dim + dim]
        #             elif idx == 1:
        #                 DOS_Exp = input_data[:, tmp_dim : tmp_dim + dim]
        #             elif idx == 2:
        #                 CC200_Exp = input_data[:, tmp_dim : tmp_dim + dim]
        #             tmp_dim += dim
                
        #         split_dims = {
        #             'AAL': input_data_dims[0],
        #             'DOS': input_data_dims[1],
        #             'CC200': input_data_dims[2],
        #         }
                
        #         split_exp = {
        #             'AAL': AAL_Exp,
        #             'DOS': DOS_Exp,
        #             'CC200': CC200_Exp,
        #         }
                
                
        #         input_data = split_exp[contain_attr[0]]
        #         input_data_dims = [split_dims[contain_attr[0]]]
        #         if len(contain_attr) > 1:
        #             for attr in contain_attr[1:]:
        #                 input_data = np.hstack((input_data, split_exp[attr]))
        #                 input_data_dims.append(split_dims[attr])
                
        #         print(comb, input_data_dims)
                
        #         args.dataset = comb

        #         if args.model == 'DeepASD':
        #             if clk > 1:
        #                 args.save_graph = 0
        #             test_acc, test_auc, test_sen, test_spe, prob_0, prob_1 = train_net_DeepASD(input_data, label, train_index, valid_index, test_index, input_data_dims, args, summary_writer, saver)
        #         elif args.model == 'adv':
        #             test_acc, test_auc, test_sen, test_spe = train_net_adv(input_data, label, train_index, test_index, valid_index, input_data_dims, args, summary_writer, saver)
                
        #         res_df.loc[len(res_df.index)] = [clk-1, test_acc, test_auc, test_sen, test_spe]
        #         print("the accuracy of f(x) is: " + str(test_acc*100))
        #         print("the auc of f(x) is: " + str(test_auc*100))
        #         print("the sen of f(x) is: " + str(test_sen*100))
        #         print("the spe of f(x) is: " + str(test_spe*100))

        #         if args.save_roc:
        #             # if test_acc > 0.8:
        #             fold_list.extend([str(clk-1)] * len(test_index))
        #             y_list.extend(label[test_index])
        #             prob_0_list.extend(prob_0)
        #             prob_1_list.extend(prob_1)
                
        #     res_save_path = os.path.join(args.result_path)
        #     if not os.path.exists(res_save_path):
        #         mkdirs(res_save_path) 
            
        #     mean_acc = np.round(res_df['test_acc'].mean() * 100, 2)
        #     std_acc = np.round(res_df['test_acc'].std() * 100, 2)
        #     mean_auc = np.round(res_df['test_auc'].mean() * 100, 2)
        #     std_auc = np.round(res_df['test_auc'].std() * 100, 2)
        #     mean_sen = np.round(res_df['test_sen'].mean() * 100, 2)
        #     std_sen = np.round(res_df['test_sen'].std() * 100, 2)
        #     mean_spe = np.round(res_df['test_spe'].mean() * 100, 2)
        #     std_spe = np.round(res_df['test_spe'].std() * 100, 2)
            
        #     comp_df.loc[len(comp_df.index)] = [args.dataset, str(mean_acc)+'±'+str(std_acc), str(mean_auc)+'±'+str(std_auc), str(mean_sen)+'±'+str(std_sen), str(mean_spe)+'±'+str(std_spe)]
    
        #     res_df.to_csv(os.path.join(res_save_path, args.dataset+'_result.csv'), index=False)
        #     comp_df.to_csv(os.path.join(res_save_path, 'compare_B_result.csv'), index=False)

        #     if args.save_roc:
        #         roc_df['fold'] = fold_list
        #         roc_df['label'] = y_list
        #         roc_df['prob_0'] = prob_0_list
        #         roc_df['prob_1'] = prob_1_list
        #         roc_df.to_csv(os.path.join(res_save_path, args.dataset+'_roc.csv'), index=False)
    elif args.dataset == 'compare_VOC':
        # load dataset
        path = args.datadir + 'VOC' + '/'
        data = np.load(path + 'voc.npz', allow_pickle=True)
        #====使用image作为主要的，text作为次要的=====#
        view_1 = data['view_1'] # text
        view_0 = data['view_0'] # image
        label = data['labels']
        input_data = np.hstack((view_1, view_0))
        input_data_dims = [399, 512]
        
        tmp_dim = 0
        for idx, dim in enumerate(input_data_dims):
            if idx == 0:
                TEXT_Exp = input_data[:, tmp_dim : tmp_dim + dim]
            elif idx == 1:
                IMAGE_Exp = input_data[:, tmp_dim : tmp_dim + dim]
            tmp_dim += dim
        
        split_dims = {
            'TEXT': input_data_dims[0],
            'IMAGE': input_data_dims[1],
        }
        
        split_exp = {
            'TEXT': TEXT_Exp,
            'IMAGE': IMAGE_Exp,
        }
        
        compare_method = ['TEXT', 'IMAGE']
        
        comp_df = pd.DataFrame(data=None, columns=['comb_method', 'test_acc', 'test_auc', 'test_sen', 'test_spe'])
        
        for comb in compare_method:
            if args.save_roc:
                roc_df = pd.DataFrame(data=None, columns=['fold', 'label', 'prob_0', 'prob_1'])
                fold_list = []
                y_list = []
                prob_0_list = []
                prob_1_list = []
            
            res_df = pd.DataFrame(data=None, columns=['fold', 'test_acc', 'test_auc', 'test_sen', 'test_spe'])
            input_data = split_exp[comb]
            input_data_dims = [split_dims[comb]]
            
            print(comb, input_data_dims)
            
            args.dataset = comb

            skf = StratifiedKFold(n_splits=args.iter_num, random_state=0, shuffle=True)
            clk = 0
            for train_index, test_index in skf.split(input_data, label):
                valid_index = test_index
                clk += 1
                if args.model == 'DeepASD':
                    if clk > 1:
                        args.save_graph = 0
                    test_acc, test_auc, test_sen, test_spe, prob_0, prob_1 = train_net_DeepASD(input_data, label, train_index, valid_index, test_index, input_data_dims, args, summary_writer, saver)
                elif args.model == 'adv':
                    test_acc, test_auc, test_sen, test_spe = train_net_adv(input_data, label, train_index, valid_index, test_index, input_data_dims, args, summary_writer, saver)
                
                res_df.loc[len(res_df.index)] = [clk-1, test_acc, test_auc, test_sen, test_spe]
                print("the accuracy of f(x) is: " + str(test_acc*100))
                print("the auc of f(x) is: " + str(test_auc*100))
                print("the sen of f(x) is: " + str(test_sen*100))
                print("the spe of f(x) is: " + str(test_spe*100))

                if args.save_roc:
                    # if test_acc > 0.8:
                    fold_list.extend([str(clk-1)] * len(test_index))
                    y_list.extend(label[test_index])
                    prob_0_list.extend(prob_0)
                    prob_1_list.extend(prob_1)
            
            res_save_path = os.path.join(args.result_path)
            if not os.path.exists(res_save_path):
                mkdirs(res_save_path) 
            
            mean_acc = np.round(res_df['test_acc'].mean() * 100, 2)
            std_acc = np.round(res_df['test_acc'].std() * 100, 2) 
            mean_auc = np.round(res_df['test_auc'].mean() * 100, 2)
            std_auc = np.round(res_df['test_auc'].std() * 100, 2)
            mean_sen = np.round(res_df['test_sen'].mean() * 100, 2)
            std_sen = np.round(res_df['test_sen'].std() * 100, 2)
            mean_spe = np.round(res_df['test_spe'].mean() * 100, 2)
            std_spe = np.round(res_df['test_spe'].std() * 100, 2)
            
            comp_df.loc[len(comp_df.index)] = [args.dataset, str(mean_acc)+'±'+str(std_acc), str(mean_auc)+'±'+str(std_auc), str(mean_sen)+'±'+str(std_sen), str(mean_spe)+'±'+str(std_spe)]
    
            res_df.to_csv(os.path.join(res_save_path, args.dataset+'_result.csv'), index=False)
            comp_df.to_csv(os.path.join(res_save_path, 'compare_VOC_result.csv'), index=False)

            if args.save_roc:
                roc_df['fold'] = fold_list
                roc_df['label'] = y_list
                roc_df['prob_0'] = prob_0_list
                roc_df['prob_1'] = prob_1_list
                roc_df.to_csv(os.path.join(res_save_path, args.dataset+'_roc.csv'), index=False)
    
    if args.dataset == 'noise_A':
        # load dataset
        path = args.datadir + "ABIDE_A" + '/'
        modal_feat_dict = np.load(path + 'modal_feat_dict.npy', allow_pickle=True).item()
        data = pd.read_csv(path + 'processed_standard_data.csv').values
        print('data shape: ', data.shape)
        input_data_dims = []
        for i in modal_feat_dict.keys():
            input_data_dims.append(len(modal_feat_dict[i]))
        print('Modal dims ', input_data_dims)
        input_data = data[:,:-1]
        label = data[:,-1]-1

        tmp_dim = 0
        for idx, dim in enumerate(input_data_dims):
            if idx == 0:
                PHENO_Exp = input_data[:, tmp_dim : tmp_dim + dim]
            elif idx == 1:
                ANAT_Exp = input_data[:, tmp_dim : tmp_dim + dim]
            elif idx == 2:
                FUNC_Exp = input_data[:, tmp_dim : tmp_dim + dim]
            else:
                FMRI_Exp = input_data[:, tmp_dim : tmp_dim + dim]
            tmp_dim += dim
        
        def gaussian_noise(x):
            mu = args.mu
            # std = 0.95 * np.std(x)
            std = args.std
            noise = np.random.normal(mu, std, size = x.shape)
            x_noisy = x + noise
            return x_noisy
        noisy_fmri = gaussian_noise(FMRI_Exp)
        input_data = np.hstack((noisy_fmri, FMRI_Exp))
        print("input data shape:", input_data.shape)
        input_data_dims = [input_data_dims[-1], input_data_dims[-1]]
        print('noisy Modal dims ', input_data_dims)

        modal_weight_list = []


        skf = StratifiedKFold(n_splits=args.iter_num, random_state=0, shuffle=True)
        clk = 0
        for train_index, test_index in skf.split(input_data, label):
            # valid_index = test_index
            train_index, valid_index = train_test_split(train_index, test_size=0.1)
            clk += 1
            if args.model == 'DeepASD':
                if clk > 1:
                    args.save_graph = 0
                test_acc, test_auc, test_sen, test_spe, prob_0, prob_1, modal_weight = train_net_DeepASD(input_data, label, train_index, valid_index, test_index, input_data_dims, args, summary_writer, saver)
            elif args.model == 'adv':
                test_acc, test_auc, test_sen, test_spe = train_net_adv(input_data, label, train_index, test_index, valid_index, input_data_dims, args, summary_writer, saver)
            
            res_df.loc[len(res_df.index)] = [clk-1, test_acc, test_auc, test_sen, test_spe]
            modal_weight_list.append(modal_weight)
            print("the accuracy of f(x) is: " + str(test_acc*100))
            print("the auc of f(x) is: " + str(test_auc*100))
            print("the sen of f(x) is: " + str(test_sen*100))
            print("the spe of f(x) is: " + str(test_spe*100))
            print("modal weight", modal_weight)

            if args.save_roc:
                if test_acc > 0.8:
                    fold_list.extend([str(clk-1)] * len(test_index))
                    y_list.extend(label[test_index])
                    prob_0_list.extend(prob_0)
                    prob_1_list.extend(prob_1)
            torch.cuda.empty_cache()
                    
            
        res_save_path = os.path.join(args.result_path)
        if not os.path.exists(res_save_path):
            mkdirs(res_save_path) 
    
        res_df.to_csv(os.path.join(res_save_path, args.dataset+'_result.csv'), index=False)
        modal_weight_df = pd.DataFrame(modal_weight_list)
        modal_weight_df.to_csv(os.path.join(res_save_path, args.dataset+'_weight.csv'), index=False)


        if args.save_roc:
            roc_df['fold'] = fold_list
            roc_df['label'] = y_list
            roc_df['prob_0'] = prob_0_list
            roc_df['prob_1'] = prob_1_list
            roc_df.to_csv(os.path.join(res_save_path, args.dataset+'_roc.csv'), index=False)
            
    elif args.dataset == 'noise_B':
        # load dataset
        path = args.datadir + "ABIDE_B" + '/'
        # data = np.load(path + 'processed_standard_data.npz', allow_pickle=True)
        # input_data_dims = [48, 6670, 12880, 19900]

        data = np.load(path + 'RFE_512_processed_standard_data.npz', allow_pickle=True)
        # input_data_dims = [48, 6670, 12880, 19900]
        # input_data_dims = [48, 256, 256, 256]
        input_data_dims = [48, 512, 512, 512]

        input_data = data["data"]
        label = data["label"]

        tmp_dim = 0
        for idx, dim in enumerate(input_data_dims):
            if idx == 0:
                PHENO_Exp = input_data[:, tmp_dim : tmp_dim + dim]
            elif idx == 1:
                AAL_Exp = input_data[:, tmp_dim : tmp_dim + dim]
            elif idx == 2:
                DOS_Exp = input_data[:, tmp_dim : tmp_dim + dim]
            elif idx == 3:
                CC200_Exp = input_data[:, tmp_dim : tmp_dim + dim]
            tmp_dim += dim
        
        def gaussian_noise(x):
            mu = args.mu
            # std = 0.95 * np.std(x)
            std = args.std
            noise = np.random.normal(mu, std, size = x.shape)
            x_noisy = x + noise
            return x_noisy
        noisy_fmri = gaussian_noise(DOS_Exp)
        input_data = np.hstack((noisy_fmri, DOS_Exp))
        # input_data = np.hstack((DOS_Exp, noisy_fmri))
        print("input data shape:", input_data.shape)
        input_data_dims = [512, 512]
        print('noisy Modal dims ', input_data_dims)


        skf = StratifiedKFold(n_splits=args.iter_num, random_state=0, shuffle=True)
        clk = 0

        modal_weight_list = []

        for train_index, test_index in skf.split(input_data, label):
            # valid_index = test_index
            train_index, valid_index = train_test_split(train_index, test_size=0.1)
            clk += 1
            if args.model == 'DeepASD':
                if clk > 1:
                    args.save_graph = 0
                test_acc, test_auc, test_sen, test_spe, prob_0, prob_1, modal_weight = train_net_DeepASD(input_data, label, train_index, valid_index, test_index, input_data_dims, args, summary_writer, saver)
            elif args.model == 'adv':
                test_acc, test_auc, test_sen, test_spe = train_net_adv(input_data, label, train_index, test_index, valid_index, input_data_dims, args, summary_writer, saver)
            
            res_df.loc[len(res_df.index)] = [clk-1, test_acc, test_auc, test_sen, test_spe]
            modal_weight_list.append(modal_weight)
            print("the accuracy of f(x) is: " + str(test_acc*100))
            print("the auc of f(x) is: " + str(test_auc*100))
            print("the sen of f(x) is: " + str(test_sen*100))
            print("the spe of f(x) is: " + str(test_spe*100))
            print("modal weight", modal_weight)

            if args.save_roc:
                fold_list.extend([str(clk-1)] * len(test_index))
                y_list.extend(label[test_index])
                prob_0_list.extend(prob_0)
                prob_1_list.extend(prob_1)
                    
            
        res_save_path = os.path.join(args.result_path)
        if not os.path.exists(res_save_path):
            mkdirs(res_save_path) 
    
        res_df.to_csv(os.path.join(res_save_path, args.dataset+'_result.csv'), index=False)
        modal_weight_df = pd.DataFrame(modal_weight_list)
        modal_weight_df.to_csv(os.path.join(res_save_path, args.dataset+'_weight.csv'), index=False)

        if args.save_roc:
            roc_df['fold'] = fold_list
            roc_df['label'] = y_list
            roc_df['prob_0'] = prob_0_list
            roc_df['prob_1'] = prob_1_list
            roc_df.to_csv(os.path.join(res_save_path, args.dataset+'_roc.csv'), index=False)
            