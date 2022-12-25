import os
import sys
import copy
import higher
import pickle
import warnings

import torch
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.utils.data import TensorDataset, DataLoader

from losses import *
from gpu_weighted_cage import *

from logistic_regression import *
from deep_net import *

warnings.filterwarnings("ignore")

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
torch.backends.cudnn.benchmark = True
print(device)

torch.set_default_dtype(torch.float64)
torch.set_printoptions(threshold=20)

save_dir, dset_directory, n_classes, feat_model, qg_available, batch_size, lr_fnetwork, lr_gm, mode, metric \
    = sys.argv[1], sys.argv[10], int(sys.argv[11]), sys.argv[12], int(sys.argv[13]), int(sys.argv[14]), \
    float(sys.argv[15]), float(sys.argv[16]), sys.argv[17], sys.argv[18]

name_dset = dset_directory.split("/")[-1].lower()
print('dset is ', name_dset)

objs = []
metric_avg = 'macro'
lam1 = 1


def rewt_lfs(sample, lr_model, theta, pi_y, pi, wts):
    wts_param = torch.nn.Parameter(wts, requires_grad=True)
    lr_model.register_parameter("wts", wts_param)
    theta_param = torch.nn.Parameter(theta, requires_grad=True)
    lr_model.register_parameter("theta", theta_param)
    pi_y_param = torch.nn.Parameter(pi_y, requires_grad=True)
    lr_model.register_parameter("pi_y", pi_y_param)
    pi_param = torch.nn.Parameter(pi, requires_grad=True)
    lr_model.register_parameter("pi", pi_param)
    if feat_model == 'lr':
        optimizer = torch.optim.Adam([
            {'params': lr_model.linear.weight},  # linear_1.parameters()},
            # {'params': lr_model['params']['linear.bias']},
            # {'params':lr_model.linear_2.parameters()},
            # {'params':lr_model.out.parameters()},
            {'params': [lr_model.theta, lr_model.pi,
                        lr_model.pi_y], 'lr': 0.01, 'weight_decay':0}
        ], lr=1e-4)
    elif feat_model == 'nn':
        optimizer = torch.optim.Adam([
            {'params': lr_model.linear_1.parameters()},
            {'params': lr_model.linear_2.parameters()},
            {'params': lr_model.out.parameters()},
            {'params': [lr_model.theta, lr_model.pi, lr_model.pi_y],
                'lr': 0.01, 'weight_decay':0}
        ], lr=1e-4)
    with higher.innerloop_ctx(lr_model, optimizer) as (fmodel, diffopt):
        supervised_criterion = torch.nn.CrossEntropyLoss()
        optimizer.zero_grad()
        supervised_indices = sample[4].nonzero().view(-1)
        unsupervised_indices = (1 - sample[4]).nonzero().squeeze()
        if len(supervised_indices) > 0:
            loss_1 = supervised_criterion(
                fmodel(sample[0][supervised_indices]), sample[1][supervised_indices])
        else:
            loss_1 = 0
        unsupervised_lr_probability = torch.nn.Softmax(dim=1)(
            fmodel(sample[0][unsupervised_indices]).view(-1, n_classes))
        loss_2 = entropy(unsupervised_lr_probability)
        y_pred_unsupervised = np.argmax(
            probability(fmodel.theta, fmodel.pi_y, fmodel.pi, sample[2][unsupervised_indices],
                        sample[3][unsupervised_indices], k, n_classes,
                        continuous_mask, fmodel.wts).detach().numpy(), 1)
        loss_3 = supervised_criterion(
            fmodel(sample[0][unsupervised_indices]), torch.tensor(y_pred_unsupervised))
        if len(supervised_indices) > 0:
            loss_4 = log_likelihood_loss_supervised(
                fmodel.theta, fmodel.pi_y, fmodel.pi, sample[1][supervised_indices],
                sample[2][supervised_indices], sample[3][supervised_indices], k, n_classes,
                continuous_mask, fmodel.wts)
        else:
            loss_4 = 0
        loss_5 = log_likelihood_loss(fmodel.theta, fmodel.pi_y, fmodel.pi, sample[2][unsupervised_indices],
                                     sample[3][unsupervised_indices], k, n_classes, continuous_mask, fmodel.wts)
        prec_loss = precision_loss(fmodel.theta, k, n_classes, a, fmodel.wts)
        probs_graphical = probability(fmodel.theta, fmodel.pi_y, fmodel.pi,
                                      sample[2], sample[3], k, n_classes, continuous_mask, fmodel.wts)
        probs_graphical = (probs_graphical.t() / probs_graphical.sum(1)).t()
        probs_lr = torch.nn.Softmax(dim=1)(fmodel(sample[0]))
        loss_6 = kl_divergence(probs_graphical, probs_lr)
        loss = loss_1 + loss_2 + loss_4 + loss_6 + loss_3 + loss_5 + prec_loss
        # print('loss --> ', loss.item())
        diffopt.step(loss)
        # print('x_valid.shape',x_valid.shape)
        # print('y_valid.shape',y_valid.shape)
        gm_val_loss = log_likelihood_loss_supervised(fmodel.theta, fmodel.pi_y, fmodel.pi,
                                                     y_valid, l_valid, s_valid, k, n_classes, continuous_mask, fmodel.wts)
        sup_val_loss = supervised_criterion(fmodel(x_valid), y_valid)
        valid_loss = lam1 * sup_val_loss + (1-lam1) * gm_val_loss
        # + 1e-20 * torch.norm(list(fmodel.parameters(time=0))[0], p=1)
        grad_all = torch.autograd.grad(valid_loss, list(fmodel.parameters(time=0))[0],
                                       only_inputs=True)[0]
        if torch.norm(grad_all, p=2) != 0:
            temp_wts = torch.clamp(
                wts-5*(grad_all/torch.norm(grad_all, p=2)), min=0, max=1)
        else:
            temp_wts = wts
        return temp_wts


def rewt_lfs1(sample, lr_model, theta, pi_y, pi, wts, feat_model):
    wts_param = torch.nn.Parameter(wts, requires_grad=True)
    lr_model.register_parameter("wts", wts_param)
    theta_param = torch.nn.Parameter(theta, requires_grad=True)
    lr_model.register_parameter("theta", theta_param)
    pi_y_param = torch.nn.Parameter(pi_y, requires_grad=True)
    lr_model.register_parameter("pi_y", pi_y_param)
    pi_param = torch.nn.Parameter(pi, requires_grad=True)
    lr_model.register_parameter("pi", pi_param)
    # print(lr_model.linear.weight)
    if feat_model == 'lr':
        optimizer = torch.optim.Adam([
            {'params': lr_model.linear.weight},  # linear_1.parameters()},
            # {'params': lr_model['params']['linear.bias']},
            # {'params':lr_model.linear_2.parameters()},
            # {'params':lr_model.out.parameters()},
            {'params': [lr_model.theta, lr_model.pi,
                        lr_model.pi_y], 'lr': 0.01, 'weight_decay':0}
        ], lr=1e-4)
    elif feat_model == 'nn':
        optimizer = torch.optim.Adam([
            {'params': lr_model.linear_1.parameters()},
            {'params': lr_model.linear_2.parameters()},
            {'params': lr_model.out.parameters()},
            {'params': [lr_model.theta, lr_model.pi, lr_model.pi_y],
                'lr': 0.01, 'weight_decay':0}
        ], lr=1e-4)
    with higher.innerloop_ctx(lr_model, optimizer) as (fmodel, diffopt):
        supervised_criterion = torch.nn.CrossEntropyLoss()
        optimizer.zero_grad()
        supervised_indices = sample[4].nonzero().view(-1)
        unsupervised_indices = (1 - sample[4]).nonzero().squeeze()
        if (sys.argv[2] == 'l1'):
            if len(supervised_indices) > 0:
                loss_1 = supervised_criterion(
                    fmodel(sample[0][supervised_indices]), sample[1][supervised_indices])
            else:
                loss_1 = 0
        else:
            loss_1 = 0
        unsupervised_lr_probability = torch.nn.Softmax(dim=1)(
            fmodel(sample[0][unsupervised_indices]).view(-1, n_classes))
        if sys.argv[3] == 'l2':
            loss_2 = entropy(unsupervised_lr_probability)
        else:
            loss_2 = 0
        y_pred_unsupervised = np.argmax(
            probability(fmodel.theta, fmodel.pi_y, fmodel.pi, sample[2][unsupervised_indices],
                        sample[3][unsupervised_indices], k, n_classes,
                        continuous_mask, fmodel.wts, device=device).cpu().detach().numpy(), 1)
        if sys.argv[4] == 'l3':
            loss_3 = supervised_criterion(fmodel(sample[0][unsupervised_indices]), torch.tensor(
                y_pred_unsupervised, device=device))
        else:
            loss_3 = 0

        if sys.argv[5] == 'l4':
            if len(supervised_indices) > 0:
                loss_4 = log_likelihood_loss_supervised(
                    fmodel.theta, fmodel.pi_y, fmodel.pi, sample[1][supervised_indices],
                    sample[2][supervised_indices], sample[3][supervised_indices], k,
                    n_classes, continuous_mask, fmodel.wts, device)
            else:
                loss_4 = 0
        else:
            loss_4 = 0

        if sys.argv[6] == 'l5':
            loss_5 = log_likelihood_loss(
                fmodel.theta, fmodel.pi_y, fmodel.pi, sample[2][unsupervised_indices],
                sample[3][unsupervised_indices], k, n_classes, continuous_mask, fmodel.wts, device)
        else:
            loss_5 = 0

        if sys.argv[8] == 'qg':
            prec_loss = precision_loss(
                fmodel.theta, k, n_classes, a, fmodel.wts, device=device)
        else:
            prec_loss = 0
        probs_graphical = probability(fmodel.theta, fmodel.pi_y, fmodel.pi,
                                      sample[2], sample[3], k, n_classes, continuous_mask, fmodel.wts, device)
        probs_graphical = (probs_graphical.t() / probs_graphical.sum(1)).t()
        probs_lr = torch.nn.Softmax(dim=1)(fmodel(sample[0]))
        if sys.argv[7] == 'l6':
            loss_6 = kl_divergence(probs_graphical, probs_lr)
        else:
            loss_6 = 0
        loss = loss_1 + loss_2 + loss_4 + loss_6 + loss_3 + loss_5 + prec_loss
        # print('loss --> ', loss.item())
        diffopt.step(loss)
        # print('x_valid.shape',x_valid.shape)
        # print('y_valid.shape',y_valid.shape)
        gm_val_loss = log_likelihood_loss_supervised(fmodel.theta, fmodel.pi_y, fmodel.pi,
                                                     y_valid.to(device), l_valid.to(device), s_valid.to(device), k, n_classes, continuous_mask, fmodel.wts, device)
        sup_val_loss = supervised_criterion(
            fmodel(x_valid.to(device)), y_valid.to(device))
        valid_loss = lam1 * sup_val_loss + (1-lam1)*gm_val_loss
        # + 1e-20 * torch.norm(list(fmodel.parameters(time=0))[0], p=1)
        grad_all = torch.autograd.grad(valid_loss, list(fmodel.parameters(time=0))[0],
                                       only_inputs=True)[0]
        if torch.norm(grad_all, p=2) != 0:
            temp_wts = torch.clamp(
                wts-5*(grad_all/torch.norm(grad_all, p=2)), min=0, max=1)
        else:
            temp_wts = wts
        return temp_wts


if mode == 'normal':
    fname = dset_directory + "/" + mode + "_d_processed.p"
    print('fname is ', fname)
else:
    fname = dset_directory + "/d_processed.p"
with open(fname, 'rb') as f:
    while 1:
        try:
            o = pickle.load(f)
        except EOFError:
            break
        objs.append(o)

x_supervised = torch.tensor(objs[0]).double()
y_supervised = torch.tensor(objs[3]).long()
l_supervised = torch.tensor(objs[2]).long()
s_supervised = torch.tensor(objs[2]).double()

objs = []
if mode == 'normal':
    fname = dset_directory + "/" + mode + "_U_processed.p"
else:
    fname = dset_directory + "/U_processed.p"

with open(fname, 'rb') as f:
    while 1:
        try:
            o = pickle.load(f)
        except EOFError:
            break
        objs.append(o)

excl = []
idx = 0
for x in objs[1]:
    if (all(x == int(n_classes))):
        excl.append(idx)
    idx += 1
print('no of excludings are ', len(excl))

x_unsupervised = torch.tensor(np.delete(objs[0], excl, axis=0)).double()
y_unsupervised = torch.tensor(np.delete(objs[3], excl, axis=0)).long()
l_unsupervised = torch.tensor(np.delete(objs[2], excl, axis=0)).long()
s_unsupervised = torch.tensor(np.delete(objs[2], excl, axis=0)).double()

print('Length of U is', len(x_unsupervised))

objs = []
if mode == 'normal':
    fname = dset_directory + "/" + mode + "_validation_processed.p"
else:
    fname = dset_directory + "/validation_processed.p"

with open(fname, 'rb') as f:
    while 1:
        try:
            o = pickle.load(f)
        except EOFError:
            break
        objs.append(o)

x_valid = torch.tensor(objs[0]).double()
y_valid = torch.tensor(objs[3]).long()
l_valid = torch.tensor(objs[2]).long()
s_valid = torch.tensor(objs[2]).double()

objs1 = []
if mode == 'normal':
    fname = dset_directory + "/" + mode + "_test_processed.p"
else:
    fname = dset_directory + "/test_processed.p"


with open(fname, 'rb') as f:
    while 1:
        try:
            o = pickle.load(f)
        except EOFError:
            break
        objs1.append(o)
x_test = torch.tensor(objs1[0]).double()
y_test = torch.tensor(objs1[3]).long()
l_test = torch.tensor(objs1[2]).long()
s_test = torch.tensor(objs1[2]).double()


n_features = x_supervised.shape[1]

if mode == 'normal':
    fname = dset_directory + '/' + mode + '_k.npy'
else:
    fname = dset_directory + '/' + 'k.npy'
k = torch.from_numpy(np.load(fname)).to(device=device).long()

n_lfs = len(k)
print('LFs are ', k)
print('no of lfs are ', n_lfs)

if qg_available:
    a = torch.from_numpy(np.load(dset_directory+'/prec.npy')
                         ).to(device=device).double()
else:
    prec_lfs = []
    for i in range(n_lfs):
        correct = 0
        for j in range(len(y_valid)):
            if y_valid[j] == l_valid[j][i]:
                correct += 1
        prec_lfs.append(correct/len(y_valid))
    a = torch.tensor(prec_lfs, device=device).double()

continuous_mask = torch.zeros(n_lfs, device=device).double()

for i in range(s_supervised.shape[0]):
    for j in range(s_supervised.shape[1]):
        if s_supervised[i, j].item() > 0.999:
            s_supervised[i, j] = 0.999
        if s_supervised[i, j].item() < 0.001:
            s_supervised[i, j] = 0.001

for i in range(s_unsupervised.shape[0]):
    for j in range(s_unsupervised.shape[1]):
        if s_unsupervised[i, j].item() > 0.999:
            s_unsupervised[i, j] = 0.999
        if s_unsupervised[i, j].item() < 0.001:
            s_unsupervised[i, j] = 0.001

for i in range(s_valid.shape[0]):
    for j in range(s_valid.shape[1]):
        if s_valid[i, j].item() > 0.999:
            s_valid[i, j] = 0.999
        if s_valid[i, j].item() < 0.001:
            s_valid[i, j] = 0.001

for i in range(s_test.shape[0]):
    for j in range(s_test.shape[1]):
        if s_test[i, j].item() > 0.999:
            s_test[i, j] = 0.999
        if s_test[i, j].item() < 0.001:
            s_test[i, j] = 0.001


l = torch.cat([l_supervised, l_unsupervised])
s = torch.cat([s_supervised, s_unsupervised])
x_train = torch.cat([x_supervised, x_unsupervised])
y_train = torch.cat([y_supervised, y_unsupervised])
supervised_mask = torch.cat(
    [torch.ones(l_supervised.shape[0]), torch.zeros(l_unsupervised.shape[0])])

num_runs = int(sys.argv[9])
final_score_gm, final_score_lr, final_score_gm_val, final_score_lr_val = [], [], [], []
final_score_lr_prec, final_score_lr_recall, final_score_gm_prec, final_score_gm_recall = [], [], [], []

# BUGS: Add rerun if bugs happen
lo = 0
while lo < num_runs:
    try:
        pi = torch.ones((n_classes, n_lfs), device=device).double()
        pi.requires_grad = True

        theta = torch.ones((n_classes, n_lfs), device=device).double() * 1
        theta.requires_grad = True

        pi_y = torch.ones(n_classes, device=device).double()
        pi_y.requires_grad = True

        if feat_model == 'lr':
            lr_model = LogisticRegression(n_features, n_classes).to(device=device)
        elif feat_model == 'nn':
            n_hidden = 512
            lr_model = DeepNet(n_features, n_hidden, n_classes).to(device=device)
        else:
            print('Please provide feature based model : lr or nn')
            exit()

        optimizer = torch.optim.Adam([{"params": lr_model.parameters()}, {
                                    "params": [pi, pi_y, theta]}], lr=0.001)
        optimizer_lr = torch.optim.Adam(lr_model.parameters(), lr=lr_fnetwork)
        optimizer_gm = torch.optim.Adam(
            [theta, pi, pi_y], lr=lr_gm, weight_decay=0)
        supervised_criterion = torch.nn.CrossEntropyLoss()

        dataset = TensorDataset(x_train, y_train, l, s, supervised_mask)

        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        save_folder = sys.argv[1]
        print('num runs are ', sys.argv[1], num_runs)
        best_score_lr, best_score_gm, best_epoch_lr, best_epoch_gm, best_score_lr_val, best_score_gm_val = 0, 0, 0, 0, 0, 0
        best_score_lr_prec, best_score_lr_recall, best_score_gm_prec, best_score_gm_recall = 0, 0, 0, 0

        stop_pahle, stop_pahle_gm = [], []

        weights = torch.ones(k.shape[0], device=device)*0.5

        for epoch in range(100):
            lr_model.train()
            for batch_ndx, sample in enumerate(loader):
                for i in range(len(sample)):
                    sample[i] = sample[i].to(device=device)
                if feat_model == 'lr':
                    lr_model1 = LogisticRegression(
                        n_features, n_classes).to(device=device)
                elif feat_model == 'nn':
                    n_hidden = 512
                    lr_model1 = DeepNet(n_features, n_hidden,
                                        n_classes).to(device=device)
                lr_model1.load_state_dict(copy.deepcopy(lr_model.state_dict()))
                theta1 = copy.deepcopy(theta)
                pi_y1 = copy.deepcopy(pi_y)
                pi1 = copy.deepcopy(pi)
                weights = rewt_lfs1(sample, lr_model1, theta1,
                                    pi_y1, pi1, weights, feat_model)
                optimizer_lr.zero_grad()
                optimizer_gm.zero_grad()

                unsup = []
                sup = []
                supervised_indices = sample[4].nonzero().view(-1)
                unsupervised_indices = (1 - sample[4]).nonzero().squeeze()

                if (sys.argv[2] == 'l1'):
                    if len(supervised_indices) > 0:
                        loss_1 = supervised_criterion(
                            lr_model(sample[0][supervised_indices]), sample[1][supervised_indices])
                    else:
                        loss_1 = 0
                else:
                    loss_1 = 0

                if (sys.argv[3] == 'l2'):
                    unsupervised_lr_probability = torch.nn.Softmax()(
                        lr_model(sample[0][unsupervised_indices]))
                    loss_2 = entropy(unsupervised_lr_probability)
                else:
                    loss_2 = 0
                if (sys.argv[4] == 'l3'):
                    y_pred_unsupervised = np.argmax(
                        probability(theta, pi_y, pi, sample[2][unsupervised_indices],
                        sample[3][unsupervised_indices], k, n_classes, continuous_mask, weights,
                        device=device).cpu().detach().numpy(), 1)
                    loss_3 = supervised_criterion(lr_model(sample[0][unsupervised_indices]),
                                                torch.tensor(y_pred_unsupervised, device=device))
                else:
                    loss_3 = 0

                if (sys.argv[5] == 'l4' and len(supervised_indices) > 0):
                    loss_4 = log_likelihood_loss_supervised(
                        theta, pi_y, pi, sample[1][supervised_indices],
                        sample[2][supervised_indices], sample[3][supervised_indices],
                        k, n_classes, continuous_mask, weights, device)
                else:
                    loss_4 = 0

                if (sys.argv[6] == 'l5'):
                    loss_5 = log_likelihood_loss(
                        theta, pi_y, pi, sample[2][unsupervised_indices],
                        sample[3][unsupervised_indices], k, n_classes, continuous_mask, weights, device)
                else:
                    loss_5 = 0

                if (sys.argv[7] == 'l6'):
                    if (len(supervised_indices) > 0):
                        supervised_indices = supervised_indices.tolist()
                        probs_graphical = probability(
                            theta, pi_y, pi, torch.cat([sample[2][unsupervised_indices], sample[2][supervised_indices]]),
                            torch.cat([sample[3][unsupervised_indices], sample[3][supervised_indices]]), k, n_classes,
                            continuous_mask, weights, device=device)
                    else:
                        probs_graphical = probability(
                            theta, pi_y, pi, sample[2][unsupervised_indices], sample[3][unsupervised_indices],
                            k, n_classes, continuous_mask, weights, device=device)
                    probs_graphical = (probs_graphical.t() /
                                    probs_graphical.sum(1)).t()
                    probs_lr = torch.nn.Softmax()(lr_model(sample[0]))
                    loss_6 = kl_divergence(probs_lr, probs_graphical)
                else:
                    loss_6 = 0

                if (sys.argv[8] == 'qg'):
                    prec_loss = precision_loss(
                        theta, k, n_classes, a, weights, device=device)
                else:
                    prec_loss = 0

                loss = loss_1 + loss_2 + loss_3 + loss_4 + loss_6+loss_5 + prec_loss
    #            print('loss is',loss_1, loss_2, loss_3, loss_4, loss_5, loss_6, prec_loss)
                if loss != 0:
                    loss.backward()
                    optimizer_gm.step()
                    optimizer_lr.step()

            y_pred = np.argmax(probability(theta, pi_y, pi, l_test.to(device), s_test.to(
                device), k, n_classes, continuous_mask, weights, device=device).cpu().detach().numpy(), 1)

            if metric == 'accuracy':
                lr_prec, lr_recall, gm_prec, gm_recall = 0, 0, 0, 0
                gm_acc = accuracy_score(y_test, y_pred)
            else:
                gm_acc = f1_score(y_test, y_pred, average=metric_avg)
                gm_prec = precision_score(y_test, y_pred, average=metric_avg)
                gm_recall = recall_score(y_test, y_pred, average=metric_avg)

            y_pred = np.argmax(probability(theta, pi_y, pi, l_valid.to(device), s_valid.to(
                device), k, n_classes, continuous_mask, weights, device=device).cpu().detach().numpy(), 1)
            if metric == 'accuracy':
                lr_prec, lr_recall, gm_prec, gm_recall = 0, 0, 0, 0
                gm_valid_acc = accuracy_score(y_valid, y_pred)
            else:
                gm_valid_acc = f1_score(y_valid, y_pred, average="macro")

            probs = torch.nn.Softmax()(lr_model(x_test.to(device)))
            y_pred = np.argmax(probs.cpu().detach().numpy(), 1)

            if metric == 'accuracy':
                lr_prec, lr_recall, gm_prec, gm_recall = 0, 0, 0, 0
                lr_acc = accuracy_score(y_test, y_pred)

            else:
                lr_acc = f1_score(y_test, y_pred, average=metric_avg)
                lr_prec = precision_score(y_test, y_pred, average=metric_avg)
                lr_recall = recall_score(y_test, y_pred, average=metric_avg)

            probs = torch.nn.Softmax()(lr_model(x_valid.to(device)))
            y_pred = np.argmax(probs.cpu().detach().numpy(), 1)

            if metric == 'accuracy':
                lr_valid_acc = accuracy_score(y_valid, y_pred)
                lr_prec, lr_recall, gm_prec, gm_recall = 0, 0, 0, 0
            else:
                lr_valid_acc = f1_score(y_valid, y_pred, average=metric_avg)
            if epoch % 5 == 0:
                print("Epoch: {}\t Test GM accuracy_score: {}".format(epoch, gm_acc))
                print("Epoch: {}\tTest LR accuracy_score: {}".format(epoch, lr_acc))

            if epoch > 5 and lr_valid_acc >= best_score_lr_val and lr_valid_acc >= best_score_gm_val:
                if lr_valid_acc == best_score_lr_val or lr_valid_acc == best_score_gm_val:
                    if best_score_lr < lr_acc or best_score_gm < gm_acc:
                        best_epoch_lr = epoch
                        best_score_lr_val = lr_valid_acc
                        best_score_lr = lr_acc

                        best_epoch_gm = epoch
                        best_score_gm_val = gm_valid_acc
                        best_score_gm = gm_acc

                        best_score_lr_prec = lr_prec
                        best_score_lr_recall = lr_recall
                        best_score_gm_prec = gm_prec
                        best_score_gm_recall = gm_recall
                else:
                    best_epoch_lr = epoch
                    best_score_lr_val = lr_valid_acc
                    best_score_lr = lr_acc

                    best_epoch_gm = epoch
                    best_score_gm_val = gm_valid_acc
                    best_score_gm = gm_acc

                    best_score_lr_prec = lr_prec
                    best_score_lr_recall = lr_recall
                    best_score_gm_prec = gm_prec
                    best_score_gm_recall = gm_recall
                    stop_pahle = []
                    stop_pahle_gm = []
                checkpoint = {'theta': theta, 'pi': pi}
                checkpoint = {'params': lr_model.state_dict()}

            if len(stop_pahle) > 10 and len(stop_pahle_gm) > 10 and (all(best_score_lr_val >= k for k in stop_pahle)):
                print('Early Stopping at', best_epoch_gm,
                    best_score_gm, best_score_lr)
                print('Validation score Early Stopping at',
                    best_epoch_gm, best_score_lr_val, best_score_gm_val)
                break
            else:
                stop_pahle.append(lr_valid_acc)
                stop_pahle_gm.append(gm_valid_acc)

        print('Best Epoch LR', best_epoch_lr)
        print('Best Epoch GM', best_epoch_gm)
        print("Run \t", lo, "Epoch, GM, LR \t", best_score_gm, best_score_lr)
        print("Run \t", lo, "GM Val, LR Val \t",
            best_score_gm_val, best_score_lr_val)
        final_score_gm.append(best_score_gm)
        final_score_lr.append(best_score_lr)
        final_score_lr_prec.append(best_score_lr_prec)
        final_score_lr_recall.append(best_score_lr_recall)

        final_score_gm_prec.append(best_score_gm_prec)
        final_score_gm_recall.append(best_score_gm_recall)

        final_score_gm_val.append(best_score_gm_val)
        final_score_lr_val.append(best_score_lr_val)

        lo += 1
    except Exception as e:
        print(e)


print("===================================================")
print("TEST Averaged scores LR", np.mean(final_score_lr))
print("TEST Precision averaged scores LR", np.mean(final_score_lr_prec))
print("TEST Recall averaged scores LR", np.mean(final_score_lr_recall))
print("===================================================")
print("TEST Averaged scores GM",  np.mean(final_score_gm))
print("TEST Precision averaged scores GM", np.mean(final_score_gm_prec))
print("TEST Recall averaged scores GM", np.mean(final_score_gm_recall))
print("===================================================")
print("VALIDATION Averaged scores are GM,LR", np.mean(
    final_score_gm_val), np.mean(final_score_lr_val))
print("TEST STD GM,LR", np.std(final_score_gm), np.std(final_score_lr))
print("VALIDATION STD GM,LR", np.std(
    final_score_gm_val), np.std(final_score_lr_val))
wt = weights.cpu().numpy()
np.save(os.path.join(dset_directory, 'weights'), wt)
print('Sorted weights ', wt.argsort())
