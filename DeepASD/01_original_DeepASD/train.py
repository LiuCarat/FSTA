from typing import OrderedDict
import numpy as np
from collections import Counter
import torch
import torch.nn as nn
import torch.optim as optim
from model import *
from utils import *
import os.path as osp
from sklearn.metrics import roc_auc_score, confusion_matrix, recall_score, accuracy_score, auc, roc_curve, classification_report
import argparse
import tempfile
# from datetime import datetime
from torch.optim.lr_scheduler import StepLR
import scipy.sparse as sp
import torch_geometric.transforms as T

from model import *
from utils import *

def performance_metric(y_true, y_pred):
    # Flatten both prediction and GT tensors
    y_pred_flat = torch.flatten(y_pred)
    y_true_flat = torch.flatten(y_true)
    # calculate the parameters
    # tp = (y_pred_flat * y_true_flat).sum()
    # tn = ((1 - y_pred_flat) * (1- y_true_flat)).sum()
    # fp = (y_pred_flat * (1 - y_true_flat)).sum()
    # fn = ((1 - y_pred_flat) * y_true_flat).sum()
    tn = (y_pred_flat * y_true_flat).sum()
    tp = ((1 - y_pred_flat) * (1- y_true_flat)).sum()
    fn = (y_pred_flat * (1 - y_true_flat)).sum()
    fp = ((1 - y_pred_flat) * y_true_flat).sum()
    # continue the calculation in numpy
    tp = tp.cpu().detach().numpy()
    fp = fp.cpu().detach().numpy()
    tn = tn.cpu().detach().numpy()
    fn = fn.cpu().detach().numpy()
    return fp, tp, fn, tn


def test_DeepASD(ex_net, gc_net, mp_net, x, target, graph_save_path=None):
    was_ex_training = False
    was_gc_training = False
    was_mp_training = False
    if ex_net.training:
        ex_net.eval()
        was_ex_training = True
    if gc_net.training:
        gc_net.eval()
        was_gc_training = True
    if mp_net.training:
        mp_net.eval()
        was_mp_training = True

    use_cuda = torch.cuda.is_available()
    dev = torch.device('cuda' if use_cuda else 'cpu')
    with torch.no_grad():
        proj_list, out_list = ex_net(x)

        fusion_feats = None
        save_proj_list = []
        for proj in proj_list:
            save_proj_list.append(proj.detach().cpu().numpy())
            if fusion_feats is not None:
                fusion_feats = torch.hstack((fusion_feats, proj.detach()))
            elif fusion_feats is None:
                fusion_feats = proj.detach()

        # adj_list = gc_net(proj_list)
        # adj = torch.zeros(x.shape[0], x.shape[0]).to(dev)
        # for a in adj_list:
        #     adj += a
        # adj /= len(adj_list)
        adj = gc_net(proj_list)
        # print("intra_w")
        # print(gc_net.intra_w)
        # fusion_feats = torch.zeros(x.shape[0], proj_list[0].shape[1]).to(dev)
        # for proj in proj_list:
        #     fusion_feats += proj.detach()
        norm_adj = normalize_adj(adj + torch.eye(adj.size(0)).to(dev))
        # prob, xx = mp_net(fusion_feats, norm_adj)
        idx = torch.nonzero(norm_adj).T
        values = norm_adj[idx[0], idx[1]]
        prob, xx = mp_net(fusion_feats, idx, values)
        
        _, pred_label = torch.max(prob, 1)
        acc_res = accuracy_score(target.cpu().numpy(), pred_label.cpu().numpy())
        fpr, tpr, thresholds = roc_curve(target.cpu().numpy(), prob[:, 0].cpu().numpy(), pos_label=0)
        auc_res = auc(fpr, tpr)
        FP, TP, FN, TN = performance_metric(target, pred_label)
        sen_res = TP / (TP + FN)
        spe_res = TN / (TN + FP)
    
    if was_ex_training:
        ex_net.train()
    if was_gc_training:
        gc_net.train()
    if was_mp_training:
        mp_net.train()

    if graph_save_path:
        if not os.path.exists(graph_save_path):
            mkdirs(graph_save_path) 
        np.savez(os.path.join(graph_save_path, 'graph.npz'),
            adj=adj.detach().cpu().numpy(),
            fused=fusion_feats.detach().cpu().numpy(),
            proj_list=save_proj_list,
            embedding=xx.detach().cpu().numpy(),
            label=target.detach().cpu().numpy()
        )

    prob_0 = prob.detach().cpu().numpy()[:, 0]
    prob_1 = prob.detach().cpu().numpy()[:, 1]

    return acc_res, auc_res, sen_res, spe_res, prob_0, prob_1


def train_net_DeepASD(input_data, label, train_index, valid_index, test_index, modal_dims, args, summary_writer, saver):
    with torch.autograd.detect_anomaly():
        use_cuda = torch.cuda.is_available()
        dev = torch.device('cuda' if use_cuda else 'cpu')
    
        x = torch.from_numpy(input_data[train_index]).float().to(dev)
        target = torch.from_numpy(label[train_index]).long().to(dev)

        class_number = len(np.unique(label))

        ex_net = VariDim_Projection(modal_dims=modal_dims, dim_hid=args.d_hid, dim_out=args.d, n_classes=class_number).to(dev)
        d_net = Discriminator(d=args.d, d_hid=args.d_dis_hidden).to(dev)
        gc_net = VariModal_GraphLearn(mode=args.GC_mode, modal_dims=modal_dims, d=args.d, th=args.th).to(dev)
        # mp_net = GCN(d=args.d, dropout=args.dropout, n_classes=class_number).to(dev)
        mp_net = SSGC(num_features=args.d*len(modal_dims), nhid=args.d*len(modal_dims)//2, num_classes=class_number, K=args.K, dropout=args.dropout).to(dev)
        # mp_net = SSGC(num_features=args.d, nhid=args.d//2, num_classes=class_number, K=5, dropout=args.dropout).to(dev)

        ex_net.apply(weights_init)
        d_net.apply(weights_init)

        criterion = nn.CrossEntropyLoss().to(dev)
        criterion_D = nn.MSELoss().to(dev)

        optimizer_G = optim.Adam(filter(lambda p: p.requires_grad, ex_net.parameters()), lr=args.lr_G, weight_decay=args.tau)
        optimizer_D = optim.Adam(filter(lambda p: p.requires_grad, d_net.parameters()), lr=args.lr_D)
        optimizer_GC = optim.Adam(filter(lambda p: p.requires_grad, gc_net.parameters()), lr=args.lr_GC)
        optimizer_MP = optim.Adam(filter(lambda p: p.requires_grad, mp_net.parameters()), lr=args.lr_MP, weight_decay=0.0001)

        scheduler_G = StepLR(optimizer_G, step_size=100, gamma=0.398)
        scheduler_D = StepLR(optimizer_D, step_size=100, gamma=0.398)

        best_val_acc = 0
        model_sav = tempfile.TemporaryFile()

        for t in range(args.epoch):
            x.requires_grad = False
            target.requires_grad = False

            ex_net.train()
            optimizer_G.zero_grad()
            proj_list, out_list = ex_net(x)

            # loss f
            loss_classifier = 0
            for out in out_list:
                loss_classifier = loss_classifier + F.nll_loss(out, target)
            
            loss_f = loss_classifier

            # inverted label loss
            loss_domain_adv = 0
            for proj in proj_list:
                domain_adv_label = F.one_hot(torch.zeros(len(target), dtype=torch.int64), 2).to(torch.float32).to(dev)
                loss_domain_adv = loss_domain_adv + criterion_D(d_net(proj), domain_adv_label)

            loss_diff = 0
            if args.with_Loss_l:
                # loss diff
                loss_diff = torch.sum(torch.abs(ex_net.Gens[0].g2.weight - ex_net.Gens[-1].g2.weight)) \
                        + torch.sum(torch.abs(ex_net.Gens[0].g2.bias - ex_net.Gens[-1].g2.bias)) \
                        + torch.sum(torch.abs(ex_net.Gens[1].g2.weight - ex_net.Gens[-1].g2.weight)) \
                        + torch.sum(torch.abs(ex_net.Gens[1].g2.bias - ex_net.Gens[-1].g2.bias)) \
                        + torch.sum(torch.abs(ex_net.Gens[2].g2.weight - ex_net.Gens[-1].g2.weight)) \
                        + torch.sum(torch.abs(ex_net.Gens[2].g2.bias - ex_net.Gens[-1].g2.bias))

                loss_generator = loss_f + loss_diff + args.beta * loss_domain_adv
            else:
                loss_generator = loss_f + args.beta * loss_domain_adv

            loss_generator.backward()
            optimizer_G.step()

            # train discriminator
            d_net.train()
            optimizer_D.zero_grad()

            loss_discriminator = 0
            for proj in proj_list:
                domain_adv_label = F.one_hot(torch.ones(len(target), dtype=torch.int64), 2).to(torch.float32).to(dev)
                loss_discriminator = loss_discriminator + criterion_D(d_net(proj.detach()), domain_adv_label)
            
            loss_discriminator.backward()
            optimizer_D.step()

            scheduler_G.step()
            scheduler_D.step()

            # train GC_MP
            ex_net.eval()
            d_net.eval()
            gc_net.train()
            mp_net.train()
            optimizer_GC.zero_grad()
            optimizer_MP.zero_grad()

            proj_list_detach = []
            fusion_feats = None
            # fusion_feats = torch.zeros(x.shape[0], proj_list[0].shape[1]).to(dev)
            for proj in proj_list:
                proj_list_detach.append(proj.detach())
                # fusion_feats += proj.detach()
                if fusion_feats is not None:
                    fusion_feats = torch.hstack((fusion_feats, proj.detach()))
                elif fusion_feats is None:
                    fusion_feats = proj.detach()
            # adj_list = gc_net(proj_list_detach)
            # adj = torch.zeros(x.shape[0], x.shape[0]).to(dev)
            # for a in adj_list:
            #     adj += a
            # adj /= len(adj_list)
            adj = gc_net(proj_list_detach)
            normalized_adj = normalize_adj(adj + torch.eye(adj.size(0)).to(dev))
            # prob, xx = mp_net(fusion_feats, normalized_adj)
            idx = torch.nonzero(normalized_adj).T
            values = normalized_adj[idx[0], idx[1]]

            # edge_index=torch.sparse_coo_tensor(index,values,ed.shape).to(dev)
            prob, xx = mp_net(fusion_feats, idx, values)

            loss = F.nll_loss(prob, target)
            if loss.item() == 0:
                exit()
            loss.backward()
            optimizer_GC.step()
            optimizer_MP.step()

            # validate
            val_x = torch.from_numpy(input_data[valid_index]).float().to(dev)
            val_target = torch.from_numpy(label[valid_index]).long().to(dev)
            cur_val_acc, cur_val_auc, cur_val_sen, cur_val_spe, _, _ = test_DeepASD(ex_net, gc_net, mp_net, val_x, val_target)

            if cur_val_acc > best_val_acc:
                wait_cnt = 0
                best_val_acc = cur_val_acc
                model_sav.close()
                model_sav = tempfile.TemporaryFile()
                dict_list = [ex_net.state_dict(), gc_net.state_dict(), mp_net.state_dict()]
                torch.save(dict_list, model_sav)
            # else:
            #     wait_cnt += 1
            #     if wait_cnt > args.wait_num and t > args.min_epoch:
            #         break
            
            info = {
                    'loss_generator': loss_generator.item(),
                    'loss_discriminator': loss_discriminator.item(),
                    'loss_classifier': loss_classifier.item(),
                    # 'loss_diff': loss_diff.item(),
                    'cls_loss': loss.item(),
                    'cur_val_acc': cur_val_acc,
                    'cur_val_auc': cur_val_auc,
                    'cur_val_sen': cur_val_sen,
                    'cur_val_spe': cur_val_spe,
            }
            for tag, value in info.items():
                summary_writer.add_scalar(tag, value, global_step=t)

            errors = OrderedDict([
                        ('loss_generator', loss_generator.item()),
                        ('loss_discriminator', loss_discriminator.item()),
                        ('loss_classifier', loss_classifier.item()),
                        # ('loss_diff', loss_diff.item()),
                        ('cls_loss', loss.item()),
                        ('cur_val_acc', cur_val_acc),
                        ('cur_val_auc', cur_val_auc),
                        ('cur_val_sen', cur_val_sen),
                        ('cur_val_spe', cur_val_spe),
                        ])
            saver.print_current_errors((t+1), errors)  

    # test
    ex_net.eval()
    gc_net.eval()
    mp_net.eval()
    model_sav.seek(0)
    dict_list = torch.load(model_sav)
    ex_net.load_state_dict(dict_list[0])
    gc_net.load_state_dict(dict_list[1])
    mp_net.load_state_dict(dict_list[2])

    if args.save_graph:
        x = torch.from_numpy(input_data).float().to(dev)
        target = torch.from_numpy(label).long().to(dev)
        test_acc, test_auc, test_sen, test_spe, _, _ = test_DeepASD(ex_net, gc_net, mp_net, x, target, args.result_path)

    test_x = torch.from_numpy(input_data[test_index]).float().to(dev)
    test_target = torch.from_numpy(label[test_index]).long().to(dev)
    test_acc, test_auc, test_sen, test_spe, prob_0, prob_1 = test_DeepASD(ex_net, gc_net, mp_net, test_x, test_target)
    
    modal_weight = gc_net.intra_w.cpu().numpy()

    return test_acc, test_auc, test_sen, test_spe, prob_0, prob_1, modal_weight


def test(ex_net, x, target):
    was_ex_training = False
    if ex_net.training:
        ex_net.eval()
        was_ex_training = True

    use_cuda = torch.cuda.is_available()
    dev = torch.device('cuda' if use_cuda else 'cpu')
    with torch.no_grad():
        proj_list, out_list = ex_net(x)
        prob = torch.zeros(out_list[0].shape[0], out_list[0].shape[1]).to(dev)
        for out in out_list:
            prob += out
        prob /= len(out_list)
        _, pred_label = torch.max(prob, 1)
        acc_res = accuracy_score(target.cpu().numpy(), pred_label.cpu().numpy())
        fpr, tpr, thresholds = roc_curve(target.cpu().numpy(), prob[:, 0].cpu().numpy(), pos_label=0)
        auc_res = auc(fpr, tpr)
        FP, TP, FN, TN = performance_metric(target, pred_label)
        sen_res = TP / (TP + FN)
        spe_res = TN / (TN + FP)
        
    if was_ex_training:
        ex_net.train()
    
    return acc_res, auc_res, sen_res, spe_res


def train_net_adv(input_data, label, train_index, valid_index, test_index, modal_dims, args, summary_writer, saver):
    with torch.autograd.set_detect_anomaly(True):
        use_cuda = torch.cuda.is_available()
        dev = torch.device('cuda' if use_cuda else 'cpu')
    
        x = torch.from_numpy(input_data[train_index]).float().to(dev)
        target = torch.from_numpy(label[train_index]).long().to(dev)

        class_number = len(np.unique(label))

        ex_net = VariDim_Projection(modal_dims=modal_dims, dim_hid=args.d_hid, dim_out=args.d, n_classes=class_number).to(dev)
        d_net = Discriminator(d=args.d, d_hid=args.d_dis_hidden).to(dev)

        ex_net.apply(weights_init)
        d_net.apply(weights_init)

        criterion = nn.CrossEntropyLoss().to(dev)
        criterion_D = nn.MSELoss().to(dev)

        optimizer_G = optim.Adam(filter(lambda p: p.requires_grad, ex_net.parameters()), lr=args.lr_G, weight_decay=args.tau)
        optimizer_D = optim.Adam(filter(lambda p: p.requires_grad, d_net.parameters()), lr=args.lr_D)

        scheduler_G = StepLR(optimizer_G, step_size=100, gamma=0.398)
        scheduler_D = StepLR(optimizer_D, step_size=100, gamma=0.398)

        best_val_acc = 0
        model_sav = tempfile.TemporaryFile()

        for t in range(args.epoch):
            x.requires_grad = False
            target.requires_grad = False

            ex_net.train()
            optimizer_G.zero_grad()
            proj_list, out_list = ex_net(x)

            # loss f
            loss_classifier = 0
            for out in out_list:
                loss_classifier = loss_classifier + criterion(out, target)
            
            loss_f = loss_classifier

            # inverted label loss
            loss_domain_adv = 0
            for proj in proj_list:
                domain_adv_label = F.one_hot(torch.zeros(len(target), dtype=torch.int64), 2).to(torch.float32).to(dev)
                loss_domain_adv = loss_domain_adv + criterion_D(d_net(proj), domain_adv_label)

            if args.with_Loss_l:
                # loss diff
                loss_diff = torch.sum(torch.abs(ex_net.Gens[0].g2.weight - ex_net.Gens[-1].g2.weight)) \
                        + torch.sum(torch.abs(ex_net.Gens[0].g2.bias - ex_net.Gens[-1].g2.bias)) \
                        + torch.sum(torch.abs(ex_net.Gens[1].g2.weight - ex_net.Gens[-1].g2.weight)) \
                        + torch.sum(torch.abs(ex_net.Gens[1].g2.bias - ex_net.Gens[-1].g2.bias)) \
                        + torch.sum(torch.abs(ex_net.Gens[2].g2.weight - ex_net.Gens[-1].g2.weight)) \
                        + torch.sum(torch.abs(ex_net.Gens[2].g2.bias - ex_net.Gens[-1].g2.bias))

                loss_generator = loss_f + loss_diff + args.beta * loss_domain_adv
            else:
                loss_generator = loss_f + args.beta * loss_domain_adv

            loss_generator.backward()
            optimizer_G.step()

            # train discriminator
            d_net.train()
            optimizer_D.zero_grad()

            loss_discriminator = 0
            for proj in proj_list:
                domain_adv_label = F.one_hot(torch.ones(len(target), dtype=torch.int64), 2).to(torch.float32).to(dev)
                loss_discriminator = loss_discriminator + criterion_D(d_net(proj.detach()), domain_adv_label)
            
            loss_discriminator.backward()
            optimizer_D.step()

            scheduler_G.step()
            scheduler_D.step()

            # validate
            val_x = torch.from_numpy(input_data[valid_index]).float().to(dev)
            val_target = torch.from_numpy(label[valid_index]).long().to(dev)
            cur_val_acc, cur_val_auc, cur_val_sen, cur_val_spe = test(ex_net, val_x, val_target)

            if cur_val_acc > best_val_acc:
                wait_cnt = 0
                best_val_acc = cur_val_acc
                model_sav.close()
                model_sav = tempfile.TemporaryFile()
                dict_list = [ex_net.state_dict()]
                torch.save(dict_list, model_sav)
            else:
                wait_cnt += 1
                if wait_cnt > args.wait_num and t > args.min_epoch:
                    break
            
            info = {
                    'loss_generator': loss_generator.item(),
                    'loss_discriminator': loss_discriminator.item(),
                    'loss_classifier': loss_classifier.item(),
                    # 'loss_diff': loss_diff.item(),
                    'cur_val_acc': cur_val_acc,
                    'cur_val_auc': cur_val_auc,
                    'cur_val_sen': cur_val_sen,
                    'cur_val_spe': cur_val_spe,
            }
            for tag, value in info.items():
                summary_writer.add_scalar(tag, value, global_step=t)

            errors = OrderedDict([
                        ('loss_generator', loss_generator.item()),
                        ('loss_discriminator', loss_discriminator.item()),
                        ('loss_classifier', loss_classifier.item()),
                        # ('loss_diff', loss_diff.item()),
                        ('cur_val_acc', cur_val_acc),
                        ('cur_val_auc', cur_val_auc),
                        ('cur_val_sen', cur_val_sen),
                        ('cur_val_spe', cur_val_spe),
                        ])
            saver.print_current_errors((t+1), errors)  

    # test
    ex_net.eval()
    model_sav.seek(0)
    dict_list = torch.load(model_sav)
    ex_net.load_state_dict(dict_list[0])
    test_x = torch.from_numpy(input_data[test_index]).float().to(dev)
    test_target = torch.from_numpy(label[test_index]).long().to(dev)
    test_acc, test_auc, test_sen, test_spe = test(ex_net, test_x, test_target)
    return test_acc, test_auc, test_sen, test_spe


def train_net_mlp(input_data, label, train_index, valid_index, test_index, modal_dims, args, summary_writer, saver):
    with torch.autograd.set_detect_anomaly(True):
        use_cuda = torch.cuda.is_available()
        dev = torch.device('cuda' if use_cuda else 'cpu')
    
        x = torch.from_numpy(input_data[train_index]).float().to(dev)
        target = torch.from_numpy(label[train_index]).long().to(dev)

        class_number = len(np.unique(label))

        ex_net = VariDim_Projection(modal_dims=modal_dims, dim_hid=args.d_hid, dim_out=args.d, n_classes=class_number).to(dev)

        ex_net.apply(weights_init)

        criterion = nn.CrossEntropyLoss().to(dev)

        optimizer_G = optim.Adam(filter(lambda p: p.requires_grad, ex_net.parameters()), lr=args.lr_G, weight_decay=args.tau)

        scheduler_G = StepLR(optimizer_G, step_size=100, gamma=0.398)

        best_val_acc = 0
        model_sav = tempfile.TemporaryFile()

        for t in range(args.epoch):
            x.requires_grad = False
            target.requires_grad = False

            ex_net.train()
            optimizer_G.zero_grad()
            proj_list, out_list = ex_net(x)

            # loss f
            loss_classifier = 0
            for out in out_list:
                loss_classifier = loss_classifier + criterion(out, target)
            
            loss_f = loss_classifier

            loss_f.backward()
            optimizer_G.step()

            # validate
            val_x = torch.from_numpy(input_data[valid_index]).float().to(dev)
            val_target = torch.from_numpy(label[valid_index]).long().to(dev)
            cur_val_acc, cur_val_auc, cur_val_sen, cur_val_spe = test(ex_net, val_x, val_target)

            # if cur_val_acc > best_val_acc:
            #     wait_cnt = 0
            #     best_val_acc = cur_val_acc
            #     model_sav.close()
            #     model_sav = tempfile.TemporaryFile()
            #     dict_list = [ex_net.state_dict()]
            #     torch.save(dict_list, model_sav)
            # else:
            #     wait_cnt += 1
            #     if wait_cnt > args.wait_num and t > args.min_epoch:
            #         break
            
            info = {
                    'loss_classifier': loss_classifier.item(),
                    # 'loss_diff': loss_diff.item(),
                    'cur_val_acc': cur_val_acc,
                    'cur_val_auc': cur_val_auc,
                    'cur_val_sen': cur_val_sen,
                    'cur_val_spe': cur_val_spe,
            }
            for tag, value in info.items():
                summary_writer.add_scalar(tag, value, global_step=t)

            errors = OrderedDict([
                        ('loss_classifier', loss_classifier.item()),
                        # ('loss_diff', loss_diff.item()),
                        ('cur_val_acc', cur_val_acc),
                        ('cur_val_auc', cur_val_auc),
                        ('cur_val_sen', cur_val_sen),
                        ('cur_val_spe', cur_val_spe),
                        ])
            saver.print_current_errors((t+1), errors)  

    # test
    ex_net.eval()
    # model_sav.seek(0)
    # dict_list = torch.load(model_sav)
    # ex_net.load_state_dict(dict_list[0])
    test_x = torch.from_numpy(input_data[test_index]).float().to(dev)
    test_target = torch.from_numpy(label[test_index]).long().to(dev)
    test_acc, test_auc, test_sen, test_spe = test(ex_net, test_x, test_target)
    return test_acc, test_auc, test_sen, test_spe