best_params_dict = {'cornell': {'model': 'GraphCON_GCN', 'lr': 0.00721, 'nhid': 256, 'alpha': 0, 'gamma': 0, 'nlayers': 1, 'dropout': 0.15, 'weight_decay': 0.0012708787092020595, 'res_version': 1},
                    'wisconsin': {'model': 'GraphCON_GCN', 'lr': 0.00356, 'nhid': 64, 'alpha': 0, 'gamma': 0, 'nlayers': 2, 'dropout': 0.23, 'weight_decay': 0.008126619200091946, 'res_version': 2},
                    'texas': {'model': 'GraphCON_GCN', 'lr': 0.00155, 'nhid': 256, 'alpha': 0, 'gamma': 0, 'nlayers': 2, 'dropout': 0.68, 'weight_decay': 0.0008549327066268375, 'res_version': 2}
                    }

H2GCN_dict = {'texas': {'model': 'H2GCN', 'lr': 0.001, 'nhid': 64, 'nlayers': 3, 'dropout': 0.37, 'weight_decay': 2e-6, 'epochs':1500, 'patience':100, 'device':'cuda'},
              'cornell': {'model': 'H2GCN', 'lr': 0.002, 'nhid': 64, 'nlayers': 2, 'dropout': 0.21, 'weight_decay': 2e-4, 'epochs':1500, 'patience':100, 'device':'cuda'},
              'wisconsin': {'model': 'H2GCN', 'lr': 0.008, 'nhid': 64, 'nlayers': 3, 'dropout': 0.21, 'weight_decay': 9e-5, 'epochs':1500, 'patience':100, 'device':'cuda'}}

GraphCON_H2GCN_dict = {'texas': {'model': 'GraphCON_H2GCN', 'lr': 0.002, 'nhid': 64, 'alpha': 1.47, 'gamma': 1.42, 'nlayers': 2, 'dropout': 0.2, 'weight_decay': 1.5e-7, 'epochs':1500, 'patience':100, 'device':'cuda'},
                      'cornell': {'model': 'GraphCON_H2GCN', 'lr': 0.075, 'nhid': 64, 'alpha': 1.51, 'gamma': 0.93, 'nlayers': 2, 'dropout': 0.075, 'weight_decay': 8e-4, 'epochs':1500, 'patience':100, 'device':'cuda'},
                      'wisconsin': {'model': 'GraphCON_H2GCN', 'lr': 0.0037, 'nhid': 64, 'alpha': 0.59, 'gamma': 0.68, 'nlayers': 2, 'dropout': 0.28, 'weight_decay': 9e-7, 'epochs':1500, 'patience':100, 'device':'cuda'}}

MLP_dict = {'texas': {'model': 'MLP', 'lr': 0.0047, 'nhid': 64, 'nlayers':3, 'dropout': 0.38, 'weight_decay': 2e-4, 'epochs':1500, 'patience':100, 'device':'cuda'},
           'cornell': {'model': 'MLP', 'lr': 0.006, 'nhid': 64, 'nlayers':2,'dropout': 0.29, 'weight_decay': 4e-4, 'epochs':1500, 'patience':100, 'device':'cuda'},
           'wisconsin': {'model': 'MLP', 'lr': 0.0055, 'nhid': 64, 'nlayers':3 , 'dropout': 0.33, 'weight_decay': 1e-4, 'epochs':1500, 'patience':100, 'device':'cuda'}}

GCN_dict = {'texas': {'model': 'GCN', 'lr': 0.014, 'nhid': 64, 'nlayers':2, 'dropout': 0.25, 'weight_decay': 1e-5, 'epochs':1500, 'patience':100, 'device':'cuda'},
           'cornell': {'model': 'GCN', 'lr': 0.022, 'nhid': 64, 'nlayers':2,'dropout': 0.19, 'weight_decay': 2e-4, 'epochs':1500, 'patience':100, 'device':'cuda'},
           'wisconsin': {'model': 'GCN', 'lr': 0.014, 'nhid': 64, 'nlayers':2 , 'dropout': 0.14, 'weight_decay': 8e-7, 'epochs':1500, 'patience':100, 'device':'cuda'}}
