from __future__ import division
from __future__ import print_function
import torch.nn.functional as F
import torch.optim as optim
from utils import *
from models_homo import MLP, HGCN
from sklearn.metrics import f1_score
import os
import argparse
from config import Config
#HOG-GCN
###################

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    parse = argparse.ArgumentParser()
    parse.add_argument("-d", "--dataset", help="dataset", default="cora", type=str, required=False)
    parse.add_argument("-l", "--labelrate",
                       help="labeled data for train per class", default=20, type=int, required=False)
    parse.add_argument("-r", "--r", help="threshhood", default=0.8, type=float, required=False)
    parse.add_argument("-r2", "--r2", help="threshhood", default=0.1, type=float, required=False)
    parse.add_argument("-epochs", "--epochs", help="epochs", default=100, type=float, required=False)
    args = parse.parse_args()
    config_file = "./config/" + str(args.dataset) + ".ini"
    config = Config(config_file)

    cuda = False

    use_seed = True  # True

    if use_seed:
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)
        if cuda:
            torch.cuda.manual_seed(config.seed)

   
    adj, sadj = load_graph_homo(args.labelrate, config)
    features, labels, idx_train, idx_val, idx_test = load_data(config)

    adj = torch.tensor(adj.todense(), dtype=torch.float32)

    print(type(features), features.shape)
    print(type(labels), labels.shape)

    model_MLP = MLP(n_feat=config.fdim,
                    n_hid=config.nhid2,
                    nclass=config.class_num,
                    dropout=config.dropout)
    optimizer_mlp = optim.Adam(model_MLP.parameters(), lr=config.lr, weight_decay=0.02)
    mlp_acc_val_best = 0

    ## MLP pre-train
    for i in range(10):
        model_MLP.train()
        optimizer_mlp.zero_grad()
        output = model_MLP(features)
        loss = F.nll_loss(output[idx_train], labels[idx_train])
        acc = accuracy(output[idx_train], labels[idx_train])
        loss.backward()
        optimizer_mlp.step()
        model_MLP.eval()
        acc_val = accuracy(output[idx_val], labels[idx_val])
        acc_test = accuracy(output[idx_test], labels[idx_test])
        print('epoch:{}'.format(i+1),
              'loss: {:.4f}'.format(loss.item()),
              'acc: {:.4f}'.format(acc.item()),
              'val: {:.4f}'.format(acc_val.item()),
              'test: {:.4f}'.format(acc_test.item()))


    si_adj = adj.clone()
    bi_adj = adj.mm(adj)
    labels_for_lp = one_hot_embedding(labels, labels.max().item() + 1, output).type(torch.FloatTensor)

    model_HGCN = HGCN(
              nfeat = config.fdim,
              adj = adj,
              nhid1 = config.nhid1,
              nhid2 = config.nhid2,
              nclass = config.class_num,
              n = config.n,
              dropout = config.dropout)
    optimizer_HGCN = optim.Adam(model_HGCN.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    best_acc_val_HGCN = 0
    best_f1 = 0
    best = 0
    best_test = 0
    for i in range(args.epochs):

        model_HGCN.train()
        model_MLP.train()
        optimizer_HGCN.zero_grad()
        optimizer_mlp.zero_grad()
        output = model_MLP(features)
        loss_mlp = F.nll_loss(output[idx_train], labels[idx_train])
        out, y_hat, adj_mask, emb = model_HGCN(features, si_adj, bi_adj, output, labels_for_lp)
        loss_lp = F.nll_loss(y_hat[idx_train], labels[idx_train])
        loss = loss_mlp + F.nll_loss(out[idx_train], labels[idx_train]) + 1*loss_lp
        acc = accuracy(out[idx_train], labels[idx_train])
        loss.backward()
        optimizer_HGCN.step()
        optimizer_mlp.step()
        model_HGCN.eval()
        model_MLP.eval()
        acc_test = accuracy(out[idx_test], labels[idx_test])
        acc_val = accuracy(out[idx_val], labels[idx_val])

        label_max = []
        for idx in idx_test:
            label_max.append(torch.argmax(out[idx]).item())
        labelcpu = labels[idx_test].data.cpu()
        macro_f1 = f1_score(labelcpu, label_max, average='macro')
        if acc_val > best_acc_val_HGCN:
            best_acc_val_HGCN = acc_val
            best_test = acc_test
            best_f1 = macro_f1
            #np.save("homo1_citeseer.npy", emb.data.numpy())
        if best < acc_test:
            best = acc_test

        print('e:{}'.format(i + 1),
              'loss: {:.4f}'.format(loss.item()-loss_mlp.item()),
              'train: {:.4f}'.format(acc.item()),
              'val: {:.4f}'.format(acc_val.item()),
              'test: {:.4f}'.format(acc_test.item()),
              'f1:{:.4f}'.format(macro_f1.item()))
        #homo_matrix = torch.matmul(output.clone().detach().exp(), output.clone().detach().exp().t())
        #print(homo_matrix)
        #homo_matrix = torch.where(homo_matrix > 0.5, homo_matrix, zero)

    print("Best：", best_test.item())
    exit()
    # labels = labels.numpy().tolist()
    # bi_adj = bi_adj.numpy().tolist()
    # homo = []
    # hete = []
    # adj_mask = adj_mask.detach().numpy().tolist()
    # for i in range(2708):
    #     for j in range(i):
    #         if bi_adj[i][j] > 0:
    #             if labels[i] == labels[j]:
    #                 homo.append(adj_mask[i][j])
    #             else:
    #                 hete.append(adj_mask[i][j])
    # np.save("homo_bi_cora.npy", np.array(homo))
    # np.save("hete_bi_cora.npy", np.array(hete))
    # exit()
    # homo_posi = 0
    # homo_total = 0
    # for i in homo:
    #     if i > 0:
    #         homo_posi += 1
    #     if i != 0:
    #         homo_total += 1
    # print(homo_posi / homo_total)
    # hete_posi = 0
    # hete_total = 0
    # for i in hete:
    #     if i < 0:
    #         hete_posi += 1
    #     if i != 0:
    #         hete_total += 1
    # print(hete_posi / hete_total)
    #
    #
    # exit()
    #
    #

    
    
