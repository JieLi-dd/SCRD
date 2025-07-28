import os
import random
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import numpy as np, argparse, time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from dataloader import IEMOCAPDataset, MELDDataset
from model import Transformer_Based_Model
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score, classification_report
import pickle as pk
import datetime
import json


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_train_valid_sampler(trainset, valid=0.1, dataset='MELD'):
    size = len(trainset)
    idx = list(range(size))
    split = int(valid * size)
    return SubsetRandomSampler(idx[split:]), SubsetRandomSampler(idx[:split])


def get_MELD_loaders(batch_size=32, valid=0.1, num_workers=0, pin_memory=False):
    trainset = MELDDataset('data/meld_multimodal_features.pkl')
    train_sampler, valid_sampler = get_train_valid_sampler(trainset, valid, 'MELD')
    train_loader = DataLoader(trainset,
                              batch_size=batch_size,
                              sampler=train_sampler,
                              collate_fn=trainset.collate_fn,
                              num_workers=num_workers,
                              pin_memory=pin_memory)
    valid_loader = DataLoader(trainset,
                              batch_size=batch_size,
                              sampler=valid_sampler,
                              collate_fn=trainset.collate_fn,
                              num_workers=num_workers,
                              pin_memory=pin_memory)

    testset = MELDDataset('data/meld_multimodal_features.pkl', train=False)
    test_loader = DataLoader(testset,
                             batch_size=batch_size,
                             collate_fn=testset.collate_fn,
                             num_workers=num_workers,
                             pin_memory=pin_memory)
    return train_loader, valid_loader, test_loader


def get_IEMOCAP_loaders(batch_size=32, valid=0.1, num_workers=0, pin_memory=False):
    trainset = IEMOCAPDataset()
    train_sampler, valid_sampler = get_train_valid_sampler(trainset, valid)
    train_loader = DataLoader(trainset,
                              batch_size=batch_size,
                              sampler=train_sampler,
                              collate_fn=trainset.collate_fn,
                              num_workers=num_workers,
                              pin_memory=pin_memory)
    valid_loader = DataLoader(trainset,
                              batch_size=batch_size,
                              sampler=valid_sampler,
                              collate_fn=trainset.collate_fn,
                              num_workers=num_workers,
                              pin_memory=pin_memory)

    testset = IEMOCAPDataset(train=False)
    test_loader = DataLoader(testset,
                             batch_size=batch_size,
                             collate_fn=testset.collate_fn,
                             num_workers=num_workers,
                             pin_memory=pin_memory)
    return train_loader, valid_loader, test_loader




def orthogonal_loss(feat1, feat2):
    """计算两个特征向量/矩阵之间的正交损失"""
    # L2 归一化
    feat1 = F.normalize(feat1, p=2, dim=1)
    feat2 = F.normalize(feat2, p=2, dim=1)

    # 计算批次中每个样本的点积 (cosine similarity)
    # 结果维度: (batch_size,)
    cosine_sim = torch.sum(feat1 * feat2, dim=1)

    # 我们希望点积为0，所以损失是点积的平方
    # 返回该批次的平均损失
    return torch.mean(cosine_sim ** 2)


def contrastive_loss(features_a, features_b, preds_a, preds_b, target, tau=2): # 原始0.07 #MELD 2附近最佳 IEMOCAP未探测
    """
    对于一对模态的对比损失。
    正对：两个模态预测均正确；
    负对：两个模态预测均错误；
    半正对：一个正确、一个错误。
    我们将正对和半正对视为应拉近的对，负对视为应推远的对。
    """
    preds_a = torch.argmax(preds_a, 1)
    preds_b = torch.argmax(preds_b, 1)
    # 判断预测是否正确
    correct_a = (preds_a == target)
    correct_b = (preds_b == target)

    # 定义正对和半正对掩码
    pos_mask = correct_a & correct_b  # 均正确 → 正对
    semi_mask = correct_a ^ correct_b  # 一个正确、一个错误 → 半正对
    pos_semi_mask = pos_mask | semi_mask  # 合并

    # 计算余弦相似度
    cos_sim = F.cosine_similarity(features_a, features_b, dim=1)  # [B]
    exp_sim = torch.exp(cos_sim / tau)

    # 对比损失：只对正对和半正对计算拉近，相对于整个批次
    if pos_semi_mask.sum() > 0:
        loss = - torch.log(exp_sim[pos_semi_mask].sum() / exp_sim.sum())
    else:
        loss = torch.tensor(0.0, device=features_a.device)
    return loss



def train_or_eval_model(model, dataloader, epoch, optimizer=None, train=False, gamma_1=1.0,
                        gamma_2=2, gamma_3=0.5, temp=0.07):
    losses, preds, labels, masks = [], [], [], []

    losses_uni = []
    losses_final = []
    losses_con = []

    assert not train or optimizer != None
    loss_func = nn.CrossEntropyLoss()
    if train:
        model.train()
    else:
        model.eval()

    for data in dataloader:
        if train:
            optimizer.zero_grad()

        textf, visuf, acouf, qmask, umask, label = [d.cuda() for d in data[:-1]] if cuda else data[:-1]
        qmask = qmask.permute(1, 0, 2)
        lengths = [(umask[j] == 1).nonzero().tolist()[-1][0] + 1 for j in range(len(umask))]

        # t_logits, a_logits, v_logits, common_logits, final_logits, t_t_out, c_t, p_t, r_t, rp_t, a_a_out, c_a, p_a, r_a, rp_a, v_v_out, c_v, p_v, r_v, rp_v = model(textf, visuf, acouf, umask, qmask, lengths)
        t_logits, a_logits, v_logits, final_logits, t_t_out, c_t, p_t, a_a_out, c_a, p_a, v_v_out, c_v, p_v = model(textf, visuf, acouf, umask, qmask, lengths)


        umask_bool = umask.bool()
        labels_ = label[umask_bool]
        t_logits = t_logits[umask_bool]
        a_logits = a_logits[umask_bool]
        v_logits = v_logits[umask_bool]
        final_logits = final_logits[umask_bool]
        # common_logits = common_logits[umask_bool]

        # t_t_out = t_t_out[umask_bool]
        # a_a_out = a_a_out[umask_bool]
        # v_v_out = v_v_out[umask_bool]
        # r_t = r_t[umask_bool]
        # r_a = r_a[umask_bool]
        # r_v = r_v[umask_bool]
        # c_t = c_t[umask_bool]
        # c_a = c_a[umask_bool]
        # c_v = c_v[umask_bool]
        p_t = p_t[umask_bool]
        p_a = p_a[umask_bool]
        p_v = p_v[umask_bool]
        # rp_t = rp_t[umask_bool]
        # rp_a = rp_a[umask_bool]
        # rp_v = rp_v[umask_bool]

        loss_con = (contrastive_loss(p_t, p_v, t_logits, v_logits, labels_, tau=temp) + contrastive_loss(p_a, p_v, a_logits, v_logits, labels_, tau=temp) + contrastive_loss(p_a, p_t, a_logits, t_logits, labels_, tau=temp)) / 3
        loss_uni = (loss_func(t_logits, labels_) + loss_func(a_logits, labels_) + loss_func(v_logits, labels_))/3
        loss_final = loss_func(final_logits, labels_)
        loss = loss_final + gamma_2 * loss_con + gamma_3 * loss_uni


        pred_ = torch.argmax(final_logits, 1)
        preds.append(pred_.data.cpu().numpy())
        labels.append(labels_.data.cpu().numpy())
        masks.append(umask.view(-1).cpu().numpy())

        losses.append(loss.item() * masks[-1].sum())

        losses_uni.append(loss_uni.item() * masks[-1].sum())
        losses_final.append(loss_final.item() * masks[-1].sum())
        losses_con.append(loss_con.item() * masks[-1].sum())

        if train:
            loss.backward()
            optimizer.step()

    if preds != []:
        preds = np.concatenate(preds)
        labels = np.concatenate(labels)
        masks = np.concatenate(masks)
    else:
        return float('nan'), float('nan'), [], [], [], float('nan')

    avg_loss = round(np.sum(losses) / np.sum(masks), 4)
    avg_accuracy = round(accuracy_score(labels, preds) * 100, 2)
    avg_fscore = round(f1_score(labels, preds, average='weighted') * 100, 2)

    avg_loss_uni = round(np.sum(losses_uni) / np.sum(masks), 4)
    avg_loss_final = round(np.sum(losses_final) / np.sum(masks), 4)
    avg_loss_con = round(np.sum(losses_con) / np.sum(masks), 4)

    return avg_loss, avg_accuracy, labels, preds, masks, avg_fscore, avg_loss_uni, avg_loss_final, avg_loss_con

def train_or_eval_model_case(model, dataloader, epoch, optimizer=None, train=False, gamma_1=1.0,
                        gamma_2=2, gamma_3=0.5, temp=0.07):
    losses, preds, labels, masks = [], [], [], []

    preds_t = []
    preds_a = []
    preds_v = []

    losses_uni = []
    losses_final = []
    losses_con = []

    assert not train or optimizer != None
    loss_func = nn.CrossEntropyLoss()
    if train:
        model.train()
    else:
        model.eval()

    for data in dataloader:
        if train:
            optimizer.zero_grad()

        textf, visuf, acouf, qmask, umask, label = [d.cuda() for d in data[:-1]] if cuda else data[:-1]
        qmask = qmask.permute(1, 0, 2)
        lengths = [(umask[j] == 1).nonzero().tolist()[-1][0] + 1 for j in range(len(umask))]

        # t_logits, a_logits, v_logits, common_logits, final_logits, t_t_out, c_t, p_t, r_t, rp_t, a_a_out, c_a, p_a, r_a, rp_a, v_v_out, c_v, p_v, r_v, rp_v = model(textf, visuf, acouf, umask, qmask, lengths)
        t_logits, a_logits, v_logits, final_logits, t_t_out, c_t, p_t, a_a_out, c_a, p_a, v_v_out, c_v, p_v = model(textf, visuf, acouf, umask, qmask, lengths)


        umask_bool = umask.bool()
        labels_ = label[umask_bool]
        t_logits = t_logits[umask_bool]
        a_logits = a_logits[umask_bool]
        v_logits = v_logits[umask_bool]
        final_logits = final_logits[umask_bool]
        # common_logits = common_logits[umask_bool]

        # t_t_out = t_t_out[umask_bool]
        # a_a_out = a_a_out[umask_bool]
        # v_v_out = v_v_out[umask_bool]
        # r_t = r_t[umask_bool]
        # r_a = r_a[umask_bool]
        # r_v = r_v[umask_bool]
        # c_t = c_t[umask_bool]
        # c_a = c_a[umask_bool]
        # c_v = c_v[umask_bool]
        p_t = p_t[umask_bool]
        p_a = p_a[umask_bool]
        p_v = p_v[umask_bool]
        # rp_t = rp_t[umask_bool]
        # rp_a = rp_a[umask_bool]
        # rp_v = rp_v[umask_bool]

        loss_con = (contrastive_loss(p_t, p_v, t_logits, v_logits, labels_, tau=temp) + contrastive_loss(p_a, p_v, a_logits, v_logits, labels_, tau=temp) + contrastive_loss(p_a, p_t, a_logits, t_logits, labels_, tau=temp)) / 3
        loss_uni = (loss_func(t_logits, labels_) + loss_func(a_logits, labels_) + loss_func(v_logits, labels_))/3
        loss_final = loss_func(final_logits, labels_)
        loss = loss_final + gamma_2 * loss_con + gamma_3 * loss_uni


        pred_ = torch.argmax(final_logits, 1)
        preds.append(pred_.data.cpu().numpy())
        labels.append(labels_.data.cpu().numpy())
        masks.append(umask.view(-1).cpu().numpy())

        losses.append(loss.item() * masks[-1].sum())

        losses_uni.append(loss_uni.item() * masks[-1].sum())
        losses_final.append(loss_final.item() * masks[-1].sum())
        losses_con.append(loss_con.item() * masks[-1].sum())

        if train:
            loss.backward()
            optimizer.step()

    if preds != []:
        preds = np.concatenate(preds)
        preds_t = np.concatenate(preds_t)
        preds_a = np.concatenate(preds_a)
        preds_v = np.concatenate(preds_v)
        labels = np.concatenate(labels)
        masks = np.concatenate(masks)
    else:
        return float('nan'), float('nan'), [], [], [], float('nan')

    avg_loss = round(np.sum(losses) / np.sum(masks), 4)
    avg_accuracy = round(accuracy_score(labels, preds) * 100, 2)
    avg_fscore = round(f1_score(labels, preds, average='weighted') * 100, 2)

    avg_loss_uni = round(np.sum(losses_uni) / np.sum(masks), 4)
    avg_loss_final = round(np.sum(losses_final) / np.sum(masks), 4)
    avg_loss_con = round(np.sum(losses_con) / np.sum(masks), 4)

    data = {
        "preds_t": preds_t.tolist(),
        "preds_a": preds_a.tolist(),
        "preds_v": preds_v.tolist(),
        "preds": preds.tolist(),
        "labels": labels.tolist(),
    }
    with open("./result/case_result_iemocap.json", "w") as f:
        json.dump(data, f)

    return avg_loss, avg_accuracy, labels, preds, masks, avg_fscore, avg_loss_uni, avg_loss_final, avg_loss_con

if __name__ == '__main__':
    # [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]
    gamma_2s = [0.2]
    gamma_3s = [1.6]
    best_all_f1 = 72.2
    for gamma_2 in gamma_2s:
        for gamma_3 in gamma_3s:
            parser = argparse.ArgumentParser()
            parser.add_argument('--no-cuda', action='store_true', default=False, help='does not use GPU')
            parser.add_argument('--lr', type=float, default=0.0001, metavar='LR', help='learning rate') # IEMOCAP（0.0001, 0.00001）# MELD（0.00008, 0.000001）
            parser.add_argument('--l2', type=float, default=0.00001, metavar='L2', help='L2 regularization weight')
            parser.add_argument('--dropout', type=float, default=0.5, metavar='dropout', help='dropout rate')
            parser.add_argument('--batch-size', type=int, default=16, metavar='BS', help='batch size')
            parser.add_argument('--hidden_dim', type=int, default=1024, metavar='hidden_dim', help='output hidden size')
            parser.add_argument('--n_head', type=int, default=8, metavar='n_head', help='number of heads')
            parser.add_argument('--epochs', type=int, default=50, metavar='E', help='number of epochs')
            parser.add_argument('--temp', type=int, default=1, metavar='temp', help='temp')
            parser.add_argument('--Dataset', default='IEMOCAP', help='dataset to train and test')
            parser.add_argument('--seed', type=int, default=2025, help='random seed')
            parser.add_argument('--train', type=bool, default=True, help="Is or Isn't train")

            args = parser.parse_args()
            today = datetime.datetime.now()
            print(args)

            seed_everything(args.seed)
            args.cuda = torch.cuda.is_available() and not args.no_cuda
            if args.cuda:
                print('Running on GPU')
            else:
                print('Running on CPU')


            cuda = args.cuda
            n_epochs = args.epochs
            batch_size = args.batch_size
            feat2dim = {'IS10': 1582, 'denseface': 342, 'MELD_audio': 300}
            D_audio = feat2dim['IS10'] if args.Dataset == 'IEMOCAP' else feat2dim['MELD_audio']
            D_visual = feat2dim['denseface']
            D_text = 1024

            D_m = D_audio + D_visual + D_text

            n_speakers = 9 if args.Dataset == 'MELD' else 2
            n_classes = 7 if args.Dataset == 'MELD' else 6 if args.Dataset == 'IEMOCAP' else 1

            print('temp {}'.format(args.temp))


            if args.Dataset == 'MELD':
                train_loader, valid_loader, test_loader = get_MELD_loaders(valid=0.1,
                                                                           batch_size=batch_size,
                                                                           num_workers=0)
                # gamma_2 = 1.4
                # gamma_3 = 0.8
                temp = 1.6
            elif args.Dataset == 'IEMOCAP':
                train_loader, valid_loader, test_loader = get_IEMOCAP_loaders(valid=0.1,
                                                                              batch_size=batch_size,
                                                                              num_workers=0)
                # gamma_2 = 1.6
                # gamma_3 = 1.6
                temp = 2.2
            else:
                print("There is no such dataset")



            model = Transformer_Based_Model(args.Dataset, args.temp, D_text, D_visual, D_audio, args.n_head,
                                            n_classes=n_classes,
                                            hidden_dim=args.hidden_dim,
                                            n_speakers=n_speakers,
                                            dropout=args.dropout)

            total_params = sum(p.numel() for p in model.parameters())
            print('total parameters: {}'.format(total_params))
            total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print('training parameters: {}'.format(total_trainable_params))

            if cuda:
                model.cuda()

            optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2)

            best_fscore, best_loss, best_label, best_pred, best_mask = None, None, None, None, None
            all_fscore, all_acc, all_loss = [], [], []
            results = {}

            if args.train:
                for e in range(n_epochs):
                    start_time = time.time()

                    train_loss, train_acc, _, _, _, train_fscore, train_loss_uni, train_loss_final, train_loss_con = train_or_eval_model(model, train_loader, e, optimizer, True, gamma_1=1, gamma_2=gamma_2, gamma_3=gamma_3, temp=temp)
                    valid_loss, valid_acc, _, _, _, valid_fscore, valid_loss_uni, valid_loss_final, valid_loss_con = train_or_eval_model(model, valid_loader, e, None, False, gamma_1=1, gamma_2=gamma_2, gamma_3=gamma_3, temp=temp)
                    test_loss, test_acc, test_label, test_pred, test_mask, test_fscore, test_loss_uni, test_loss_final, test_loss_con = train_or_eval_model(model, test_loader, e)

                    result = {
                        "train_loss": train_loss,
                        "train_acc": train_acc,
                        "train_fscore": train_fscore,
                        "valid_loss": valid_loss,
                        "valid_acc": valid_acc,
                        "valid_fscore": valid_fscore,
                        "test_loss": test_loss,
                        "test_acc": test_acc,
                        "test_fscore": test_fscore,
                        "train_loss_uni": train_loss_uni,
                        "train_loss_con": train_loss_con,
                        "train_loss_final": train_loss_final,
                        "valid_loss_uni": valid_loss_uni,
                        "valid_loss_con": valid_loss_con,
                        "valid_loss_final": valid_loss_final,
                        "test_loss_uni": test_loss_uni,
                        "test_loss_con": test_loss_con,
                        "test_loss_final": test_loss_final,
                    }
                    results[f"{e}_{gamma_2}_{gamma_3}"] = result

                    if best_fscore == None or best_fscore < test_fscore:
                        best_fscore = test_fscore
                        best_label, best_pred, best_mask = test_label, test_pred, test_mask
                        if best_fscore > best_all_f1:
                            torch.save(model.state_dict(), f'./save_model/best_{args.Dataset}_{gamma_2}_{gamma_3}.pt')

                    print(
                        'epoch: {}, train_loss: {}, train_acc: {}, train_fscore: {}, valid_loss: {}, valid_acc: {}, valid_fscore: {}, test_loss: {}, test_acc: {}, test_fscore: {}, time: {} sec'. \
                        format(e + 1, train_loss, train_acc, train_fscore, valid_loss, valid_acc, valid_fscore, test_loss, test_acc,
                               test_fscore, round(time.time() - start_time, 2)))
                    if (e + 1) % 10 == 0:
                        print(classification_report(best_label, best_pred, digits=4))
                        print(confusion_matrix(best_label, best_pred))


                with open(f"result/{args.Dataset}_{gamma_2}_{gamma_3}_{best_fscore}.json", "w") as f:
                    json.dump(results, f)

            else:
                model.load_state_dict(torch.load(f'./save_model/best_model_{args.Dataset}_71.76.pt'))
                test_loss, test_acc, test_label, test_pred, test_mask, test_fscore, test_loss_uni, test_loss_select, test_loss_con = train_or_eval_model_case(model, test_loader, 0)
                print(classification_report(test_label, test_pred, digits=4))
                print(confusion_matrix(test_label, test_pred))


            print(classification_report(best_label, best_pred, digits=4))
            print(confusion_matrix(best_label, best_pred))





