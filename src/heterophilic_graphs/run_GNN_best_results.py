from data_handling import get_data
import numpy as np
import torch.optim as optim
from models import *
from torch import nn
from best_params import best_params_dict, GraphCON_H2GCN_dict, H2GCN_dict, MLP_dict, GCN_dict

from torch_geometric.utils import to_dense_adj
from torch.nn import Sequential, ReLU, Linear

import argparse

def one_hot_embedding(labels, num_classes, soft):
    """Embedding labels to one-hot form.
    Args:
      labels: (LongTensor) class labels, sized [N,].
      num_classes: (int) number of classes.
    Returns:
      (tensor) encoded labels, sized [N, #classes].
    """
    soft = torch.argmax(soft.exp(), dim=1)
    y = torch.eye(num_classes)
    return y[soft]

def train_GNN(opt,split):
    data = get_data(opt['dataset'],split)

    best_eval = 10000
    bad_counter = 0
    best_test_acc = 0

    if opt['model'] == 'GraphCON_GCN':
        model = GraphCON_GCN(nfeat=data.num_features,nhid=opt['nhid'],nclass=5,
                             dropout=opt['drop'],nlayers=opt['nlayers'],dt=1.,
                             alpha=opt['alpha'],gamma=opt['gamma'],res_version=opt['res_version']).to(opt['device'])
    elif opt['model'] == 'GraphCON_GAT':
        model = GraphCON_GAT(nfeat=data.num_features, nhid=opt['nhid'], nclass=5,
                             dropout=opt['drop'], nlayers=opt['nlayers'], dt=1.,
                             alpha=opt['alpha'], gamma=opt['gamma'],nheads=opt['nheads']).to(opt['device'])
    elif opt['model'] == "GraphCON_GRAFF":
        
        adj = to_dense_adj(data.edge_index)[0].to(opt['device'])
        model = GraphCON_GRAFF(nfeat=data.num_features,nhid=opt['nhid'],nclass=5,
                             dropout=opt['drop'],nlayers=opt['nlayers'],dt=1.,
                             alpha=opt['alpha'],gamma=opt['gamma'], A=adj, step_size=3 / opt["nlayers"], device=opt['device']).to(opt['device'])
    
    elif opt["model"] == "GCN":
        model = GCN(nfeat=data.num_features, nhid=opt['nhid'], nclass=5,
                             dropout=opt['drop'], nlayers=opt['nlayers']).to(opt['device'])
    elif opt["model"] == "GAT":
        model = GAT(nfeat=data.num_features, nhid=opt['nhid'], nclass=5,
                             dropout=opt['drop'], nlayers=opt['nlayers'], nheads=opt["nheads"]).to(opt['device'])
    elif opt["model"] == "GRAFF":
        adj = to_dense_adj(data.edge_index)[0].to(opt['device'])
        model = GRAFF(nfeat=data.num_features, nhid=opt['nhid'], nclass=5,
                             dropout=opt['drop'], nlayers=opt['nlayers'], A=adj, step_size=3 / opt["nlayers"], device=opt["device"]).to(opt['device'])

    elif opt["model"] == "H2GCN":
        
        adj = to_dense_adj(data.edge_index)[0].to(opt['device']).to_sparse()
        model = H2GCN(nfeat=data.num_features, nhid=opt['nhid'], nclass=5,
                             dropout=opt['drop'], nlayers=opt['nlayers'], adj = adj).to(opt['device'])
    elif opt["model"] == "GraphCON_H2GCN":
        adj = to_dense_adj(data.edge_index)[0].to(opt['device']).to_sparse()
        model = GraphCON_H2GCN(nfeat=data.num_features, nhid=opt['nhid'], nclass=5,
                             dropout=opt['drop'], nlayers=opt['nlayers'], adj = adj,
                               alpha=opt['alpha'], gamma=opt['gamma']).to(opt['device'])
    elif opt["model"] == "MLP":
        model = MLP(n_feat=data.num_features, n_hid=opt['nhid'], nclass=5, nlayers=opt["nlayers"],
                             dropout=opt['drop']).to(opt['device'])
    else:
        raise Exception("Not a valid model specified")
    
    optimizer = optim.Adam(model.parameters(),lr=opt['lr'],weight_decay=opt['weight_decay'])
    lf = nn.CrossEntropyLoss()

    @torch.no_grad()
    def test(model, data):
        model.eval()
        logits, accs, losses = model(data), [], []
        for _, mask in data('train_mask', 'val_mask', 'test_mask'):
            loss = lf(out[mask], data.y.squeeze()[mask])
            pred = logits[mask].max(1)[1]
            acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
            accs.append(acc)
            losses.append(loss.item())
        return accs, losses

    for epoch in range(opt['epochs']):
        model.train()
        optimizer.zero_grad()
        out = model(data.to(opt['device']))
        loss = lf(out[data.train_mask], data.y.squeeze()[data.train_mask])
        loss.backward()
        optimizer.step()

        [train_acc, val_acc, test_acc], [train_loss, val_loss, test_loss] = test(model,data)

        if (val_loss < best_eval):
            best_eval = val_loss
            best_test_acc = test_acc
            bad_counter = 0
        else:
            bad_counter += 1

        if(bad_counter==opt['patience']):
            break

        log = 'Split: {:01d}, Epoch: {:03d}, Train: {:.4f}, Val: {:.4f}, Test: {:.4f}'
        #print(log.format(split, epoch, train_acc, val_acc, test_acc))

    return best_test_acc, model

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='training parameters')
    parser.add_argument('--dataset', type=str, default='texas',
                        help='cornell, wisconsin, texas')
    parser.add_argument('--model', type=str, default='GraphCON_GCN',
                        help='GraphCON_GCN, GraphCON_GAT, GraphCON_GRAFF')
    parser.add_argument('--nhid', type=int, default=64,
                        help='number of hidden node features')
    parser.add_argument('--nlayers', type=int, default=5,
                        help='number of layers')
    parser.add_argument('--alpha', type=float, default=1.,
                        help='alpha parameter of graphCON')
    parser.add_argument('--gamma', type=float, default=1.,
                        help='gamma parameter of graphCON')
    parser.add_argument('--nheads', type=int, default=4,
                        help='number of attention heads for GraphCON-GAT')
    parser.add_argument('--epochs', type=int, default=1500,
                        help='max epochs')
    parser.add_argument('--patience', type=int, default=100,
                        help='patience')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate')
    parser.add_argument('--drop', type=float, default=0.3,
                        help='dropout rate')
    parser.add_argument('--res_version', type=int, default=1,
                        help='version of residual connection')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                        help='weight_decay')
    parser.add_argument('--device', type=str, default=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
                        help='computing device')

    args = parser.parse_args()
    cmd_opt = vars(args)
    
    accuracies = {}
    
    it = 0
    
    for it in range(10):
        # , 
        for model_name in ["GCN", "GraphCON_GCN"]:#["GraphCON_H2GCN", "MLP", "H2GCN"]:
            cmd_opt["model"] = model_name

            for dataset in ["texas", "cornell", "wisconsin"]:

                cmd_opt["dataset"] = dataset
                print(it, model_name, dataset)

                if it == 0:
                    accuracies[model_name+'_'+dataset] = []

                best_opt = {}
                if cmd_opt["model"] == "H2GCN":
                    best_opt = H2GCN_dict[cmd_opt['dataset']]
                elif cmd_opt["model"] == "GraphCON_H2GCN":
                    best_opt = GraphCON_H2GCN_dict[cmd_opt['dataset']]
                elif cmd_opt["model"] == "MLP":
                    best_opt = MLP_dict[cmd_opt['dataset']]
                elif cmd_opt["model"] == "GraphCON_GCN":
                    best_opt = best_params_dict[cmd_opt['dataset']]
                elif cmd_opt["model"] == "GCN":
                    best_opt = GCN_dict[cmd_opt['dataset']]

                opt = {**cmd_opt, **best_opt}

                print(opt)

                n_splits = 10

                best = []
                for split in range(n_splits):
                    best_test_acc, model = train_GNN(opt,split)
                    best.append(best_test_acc)

                mean_accuracy = np.mean(np.array(best)*100)
                accuracies[model_name+'_'+dataset].append(mean_accuracy)

                print('Mean test accuracy: ', mean_accuracy,'std: ', np.std(np.array(best)*100))

import pickle

with open('accuracy_stats_GCN.pickle', 'wb') as handle:
    pickle.dump(accuracies, handle)

print(accuracies)

#with open('filename.pickle', 'rb') as handle:
#    b = pickle.load(handle)

    
# MLP [4275.4150390625, 1628.3748428344727, 13594.69755859375, 82894.022265625, 411499.57343750005, 427088.73906249995] 
# H2GCN [1928.9462402343747, 3241.3383544921876, 8498.007861328124, 24163.698632812502, 67618.14882812499, 184842.746875]
# GraphCON_H2GCN [13547.726171875, 25494.873828125004, 18640.529687500002, 42376.43515625, 57228.41328125, 104037.21250000001]
# GCN [1688.3285766601562, 1790.5974731445312, 1970.6301025390624, 2204.461083984375, 2406.882183837891, 2325.0643066406246]
# GAT [1751.253015136719, 1885.2213623046878, 2115.6904907226562, 2425.333737182617, 2690.9900527954105, 2501.997875976562]

# GRAFF [644.8661804199219, 2445.190661621094, 8697.16025390625, 28642.1345703125, 82667.047265625, 170942.0890625]
