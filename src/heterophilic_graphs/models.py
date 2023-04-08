import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.utils import to_dense_adj
import torch_sparse

def dirichlet_energy(X, adj):
    # X is a matrix of shape (num_nodes, feature_channels)
    # adj_norm is a torch sparse coo tensor

    adj_norm = sym_norm_adj(adj)
    device = adj_norm.device
    adj_norm = adj_norm.to_sparse()
    L = torch.eye(adj_norm.shape[0]).to(device) - adj_norm
    b = torch.sparse.mm(L, X)
    product = torch.sparse.mm(torch.transpose(b,0,1), X)
    energy = torch.trace(product)
    return energy.cpu().detach().item()

def sym_norm_adj(A):
    
    A_tilde = torch.eye(A.shape[0]).to(A.device) + A 
    D_tilde = torch.diag(torch.sum(A_tilde,0)).to(A.device)
    D_tilde_inv_sqrt = 1 / torch.sqrt(D_tilde)    
    D_tilde_inv_sqrt[torch.isinf(D_tilde_inv_sqrt)] = 0.0
    A_tilde = A_tilde.to_sparse()
    D_tilde_inv_sqrt = D_tilde_inv_sqrt.to_sparse()
    adj_norm = torch.sparse.mm(torch.sparse.mm(D_tilde_inv_sqrt, A_tilde), D_tilde_inv_sqrt)
    return adj_norm

class MLP(nn.Module):
    def __init__(self, n_feat, n_hid, nclass, dropout, nlayers=2):
        super(MLP, self).__init__()
        layer_list = [nn.Linear(n_feat, n_hid)]
        
        for i in range(nlayers-2):
            layer_list.append(nn.ReLU())
            layer_list.append(nn.Linear(n_hid, n_hid))
        
        layer_list.append(nn.ReLU())
        layer_list.append(nn.Linear(n_hid, nclass))
        
        self.energies = []
        self.embeddings = []
        
        self.mlp = nn.Sequential(*layer_list)

    def forward(self, data):
        if not self.training:
            adj = to_dense_
            (data.edge_index)[0].to("cuda")
            self.energies = [dirichlet_energy(data.x, adj)]
            self.embeddings = [data.x]
            for layer in self.mlp:
                if "Linear" in str(layer):
                    self.embeddings.append(layer(self.embeddings[-1]))
                    self.energies.append(dirichlet_energy(self.embeddings[-1], adj))
        
        return self.mlp(data.x)

    def get_emb(self, x):
        return self.mlp[0](x).detach()

                
#ref - https://github.com/GitEventhandler/H2GCN-PyTorch/blob/master/model.py    
class H2GCN_conv(nn.Module):
    def __init__(
            self,
            in_dim: int, out_dim: int,
            neighborhoods: list = [0,1,2]
    ):
        super(H2GCN_conv, self).__init__()
        self.act = F.relu
        self.initialized = False
        self.neighborhoods = neighborhoods
        self.w0 = nn.Linear(in_dim, out_dim - 2 * (out_dim // len(neighborhoods)))
        self.w1 = nn.Linear(in_dim, out_dim // len(neighborhoods))
        self.w2 = nn.Linear(in_dim, out_dim // len(neighborhoods))

    @staticmethod
    def _indicator(sp_tensor: torch.sparse.Tensor) -> torch.sparse.Tensor:
        csp = sp_tensor.coalesce()
        return torch.sparse_coo_tensor(
            indices=csp.indices(),
            values=torch.where(csp.values() > 0, 1, 0),
            size=csp.size(),
            dtype=torch.float
        )

    @staticmethod
    def _spspmm(sp1: torch.sparse.Tensor, sp2: torch.sparse.Tensor) -> torch.sparse.Tensor:
        assert sp1.shape[1] == sp2.shape[0], 'Cannot multiply size %s with %s' % (sp1.shape, sp2.shape)
        sp1, sp2 = sp1.coalesce(), sp2.coalesce()
        index1, value1 = sp1.indices(), sp1.values()
        index2, value2 = sp2.indices(), sp2.values()
        m, n, k = sp1.shape[0], sp1.shape[1], sp2.shape[1]
        indices, values = torch_sparse.spspmm(index1, value1, index2, value2, m, n, k)
        return torch.sparse_coo_tensor(
            indices=indices,
            values=values,
            size=(m, k),
            dtype=torch.float
        )

    @classmethod
    def _adj_norm(cls, adj: torch.sparse.Tensor) -> torch.sparse.Tensor:
        n = adj.size(0)
        d_diag = torch.pow(torch.sparse.sum(adj, dim=1).values(), -0.5)
        d_diag = torch.where(torch.isinf(d_diag), torch.full_like(d_diag, 0), d_diag)
        d_tiled = torch.sparse_coo_tensor(
            indices=[list(range(n)), list(range(n))],
            values=d_diag,
            size=(n, n)
        )
        return cls._spspmm(cls._spspmm(d_tiled, adj), d_tiled)

    def _prepare_prop(self, adj):
        n = adj.size(0)
        device = adj.device
        self.initialized = True
        sp_eye = torch.sparse_coo_tensor(
            indices=[list(range(n)), list(range(n))],
            values=[1.0] * n,
            size=(n, n),
            dtype=torch.float
        ).to(device)
        # initialize A1, A2
        a1 = self._indicator(adj - sp_eye)
        a2 = self._indicator(self._spspmm(adj, adj) - adj - sp_eye)
        # norm A1 A2
        self.a1 = self._adj_norm(a1)
        self.a2 = self._adj_norm(a2)

    def forward(self, x: torch.FloatTensor, adj: torch.sparse.Tensor) -> torch.FloatTensor:
        if not self.initialized:
            self._prepare_prop(adj)
        # H2GCN propagation
        r0 = self.act(self.w0(x))
        
        r1 = torch.spmm(self.a1, x)
        r1 = self.act(self.w1(r1))
        
        r2 = torch.spmm(self.a2, x)
        r2 = self.act(self.w2(r2))

        r_final = torch.cat([r0,r1,r2], dim=1)
        return r_final
        #return torch.softmax(torch.mm(r_final, self.w_classify), dim=1)
    
class GRAFFLayer(nn.Module):
    """GRAFF layer

    Args:
        input_dim (int): Dimensionality of the input feature vectors
        output_dim (int): Dimensionality of the output softmax distribution
        A (torch.Tensor): 2-D adjacency matrix
    """
    def __init__(self, input_dim, output_dim, A, step_size, use_W=True, nonlinear=False, device="cpu"):
        super(GRAFFLayer, self).__init__()
        self.device = device
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_W = use_W
        self.nonlinear = nonlinear
        self.A = A
        self.step_size = step_size
        self.adj_norm = self.sym_norm_adj(A)
        self.linear = nn.Linear(input_dim, output_dim, bias=False)
        

    def sym_norm_adj(self, A):
        #### Create the symmetric normalised adjacency from the dense adj matrix A
        # This should return a sparse adjacency matrix. (torch sparse coo tensor format)
        A_tilde = A + torch.eye(A.shape[0]).to(self.device)
        D_tilde = torch.diag(torch.sum(A_tilde,0))
        D_tilde_inv_sqrt = 1 / torch.sqrt(D_tilde)    
        D_tilde_inv_sqrt[torch.isinf(D_tilde_inv_sqrt)] = 0.0
        A_tilde = A_tilde.to_sparse()
        D_tilde_inv_sqrt = D_tilde_inv_sqrt.to_sparse()
        adj_norm = torch.sparse.mm(torch.sparse.mm(D_tilde_inv_sqrt, A_tilde), D_tilde_inv_sqrt)
        return adj_norm
        
    def forward(self, x):
        w_star = 0.5 * (self.linear.weight + self.linear.weight.T)
        inter = torch.sparse.mm(self.step_size * torch.sparse.mm(self.adj_norm, x), w_star)
        output = x + inter

        return output

class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, nlayers):

        super(GCN, self).__init__()
        self.dropout = dropout
        self.nhid = nhid
        self.nlayers = nlayers
        self.enc = nn.Linear(nfeat,nhid)
        self.conv = GCNConv(nhid, nhid)
        self.dec = nn.Linear(nhid,nclass)
        self.res = nn.Linear(nhid,nhid)

        self.energies = []
        
        self.act_fn = nn.ReLU()
        self.reset_params()
    
    def forward(self, data):
        input = data.x
        edge_index = data.edge_index
        adj = to_dense_adj(data.edge_index)[0].to("cuda")
        
        input = F.dropout(input, self.dropout, training=self.training)
        
        X = self.act_fn(self.enc(input))
        self.energies = [dirichlet_energy(X, adj)]
        
        for i in range(self.nlayers):
            X = self.act_fn(self.conv(X, edge_index) + X) 
            X = F.dropout(X, self.dropout, training=self.training)
            self.energies.append(dirichlet_energy(X, adj))
        X = self.dec(X)
        return X


    def reset_params(self):
        for name, param in self.named_parameters():
            if 'weight' in name and 'emb' not in name and 'out' not in name:
                stdv = 1. / math.sqrt(self.nhid)
                param.data.uniform_(-stdv, stdv)

class GAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, nlayers, dropout, nheads=4):

        super(GAT, self).__init__()
        self.dropout = dropout
        self.nhid = nhid
        self.nlayers = nlayers
        self.res = nn.Linear(nhid, nheads * nhid)
        self.enc = nn.Linear(nfeat,nhid)
        self.conv = GATConv(nhid, nhid, heads=nheads)
        self.dec = nn.Linear(nhid ,nclass)
        self.nheads = nheads
        self.act_fn = nn.ReLU()
        self.reset_params()
        
        self.energies = []
        self.calc_energies = False
    
    def forward(self, data):
        input = data.x
        n_nodes = input.size(0)
        edge_index = data.edge_index
        
        adj = to_dense_adj(data.edge_index)[0].to("cuda")        
        
        input = F.dropout(input, self.dropout, training=self.training)
        X = self.act_fn(self.enc(input))
        if self.calc_energies:
            self.energies = [dirichlet_energy(X, adj)]
        for i in range(self.nlayers):
            X = X + self.act_fn(self.conv(X, edge_index)).view(n_nodes, -1, self.nheads).mean(dim=-1)
            X = F.dropout(X, self.dropout, training=self.training)
            if self.calc_energies:
                self.energies.append(dirichlet_energy(X, adj))
        X = self.dec(X)
        return X


    def reset_params(self):
        for name, param in self.named_parameters():
            if 'weight' in name and 'emb' not in name and 'out' not in name:
                stdv = 1. / math.sqrt(self.nhid)
                param.data.uniform_(-stdv, stdv)

class GRAFF(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, nlayers, A, step_size, device="cpu"):

        super(GRAFF, self).__init__()
        self.dropout = dropout
        self.nhid = nhid
        self.nlayers = nlayers
        self.enc = nn.Linear(nfeat,nhid)
        self.conv = GRAFFLayer(nhid, nhid, A, step_size, device=device)
        self.dec = nn.Linear(nhid,nclass)
        self.res = nn.Linear(nhid,nhid)

        self.act_fn = nn.ReLU()
        self.reset_params()
        
        self.energies = []
    
    def forward(self, data):
        input = data.x
        edge_index = data.edge_index

        adj = to_dense_adj(data.edge_index)[0].to("cuda")
        
        input = F.dropout(input, self.dropout, training=self.training)
        X = self.act_fn(self.enc(input))
        
        self.energies = [dirichlet_energy(X, adj)]
        for i in range(self.nlayers):
            X = X + self.act_fn(self.conv(X))
            X = F.dropout(X, self.dropout, training=self.training)
            self.energies.append(dirichlet_energy(X, adj))
        X = self.dec(X)
        return X

    def reset_params(self):
        for name, param in self.named_parameters():
            if 'weight' in name and 'emb' not in name and 'out' not in name:
                stdv = 1. / math.sqrt(self.nhid)
                param.data.uniform_(-stdv, stdv)


class H2GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, nlayers, adj, neighbors=[0,1,2]):

        super(H2GCN, self).__init__()
        self.dropout = dropout
        self.nhid = nhid
        self.nlayers = nlayers
        self.enc = nn.Linear(nfeat,nhid)
        self.conv = H2GCN_conv(nhid, nhid, neighbors)
        self.dec = nn.Linear(nhid,nclass)
        
        self.adj = adj
        
        self.energies = []

        self.act_fn = nn.ReLU()
        self.reset_params()
    
    def forward(self, data):
        input = data.x
        input = F.dropout(input, self.dropout, training=self.training)
        
        self.energies = []
        #self.embeddings = []
        
        X = self.act_fn(self.enc(input))
        
        for i in range(self.nlayers):
            self.energies.append(dirichlet_energy(X, self.adj))
            #self.embeddings.append(X)
            X = self.conv(X, self.adj) + X
            X = F.dropout(X, self.dropout, training=self.training)
        
        self.energies.append(dirichlet_energy(X, self.adj))
        #self.embeddings.append(X)
        X = self.dec(X)
        return X

    def reset_params(self):
        for name, param in self.named_parameters():
            if 'weight' in name and 'emb' not in name and 'out' not in name:
                stdv = 1. / math.sqrt(self.nhid)
                param.data.uniform_(-stdv, stdv)   

                
class GraphCON_GRAFF(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, nlayers, A, step_size, dt=1., alpha=1., gamma=1., device="cpu"):
        super(GraphCON_GRAFF, self).__init__()
        self.dropout = dropout
        self.nhid = nhid
        self.nlayers = nlayers
        self.enc = nn.Linear(nfeat,nhid)
        self.conv = GRAFFLayer(nhid, nhid, A, step_size, device=device)
        self.dec = nn.Linear(nhid,nclass)
        self.res = nn.Linear(nhid,nhid)
        self.dt = dt
        self.act_fn = nn.ReLU()
        self.alpha = alpha
        self.gamma = gamma
        self.reset_params()

    def reset_params(self):
        for name, param in self.named_parameters():
            if 'weight' in name and 'emb' not in name and 'out' not in name:
                stdv = 1. / math.sqrt(self.nhid)
                param.data.uniform_(-stdv, stdv)

    def forward(self, data):
        input = data.x
        edge_index = data.edge_index
        input = F.dropout(input, self.dropout, training=self.training)
        Y = self.act_fn(self.enc(input))
        X = Y
        Y = F.dropout(Y, self.dropout, training=self.training)
        X = F.dropout(X, self.dropout, training=self.training)

        for i in range(self.nlayers):
            Y = Y + self.dt*(self.act_fn(self.conv(X) + X) - self.alpha*Y - self.gamma*X)
            X = X + self.dt*Y
            Y = F.dropout(Y, self.dropout, training=self.training)
            X = F.dropout(X, self.dropout, training=self.training)
        X = self.dec(X)
        return X

class GraphCON_GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, nlayers, dt=1., alpha=1., gamma=1., res_version=1):
        super(GraphCON_GCN, self).__init__()
        self.dropout = dropout
        self.nhid = nhid
        self.nlayers = nlayers
        self.enc = nn.Linear(nfeat,nhid)
        self.conv = GCNConv(nhid, nhid)
        self.dec = nn.Linear(nhid,nclass)
        self.res = nn.Linear(nhid,nhid)
        if(res_version==1):
            self.residual = self.res_connection_v1
        else:
            self.residual = self.res_connection_v2
        self.dt = dt
        self.act_fn = nn.ReLU()
        self.alpha = alpha
        self.gamma = gamma
        self.reset_params()
        
        self.energies = []

    def reset_params(self):
        for name, param in self.named_parameters():
            if 'weight' in name and 'emb' not in name and 'out' not in name:
                stdv = 1. / math.sqrt(self.nhid)
                param.data.uniform_(-stdv, stdv)

    def res_connection_v1(self, X):
        res = - self.res(self.conv.lin(X))
        return res

    def res_connection_v2(self, X):
        res = - self.conv.lin(X) + self.res(X)
        return res

    def forward(self, data):
        input = data.x
        edge_index = data.edge_index
        input = F.dropout(input, self.dropout, training=self.training)
        Y = self.act_fn(self.enc(input))
        X = Y
        Y = F.dropout(Y, self.dropout, training=self.training)
        X = F.dropout(X, self.dropout, training=self.training)
        
        adj = to_dense_adj(data.edge_index)[0].to("cuda")
        self.energies = [dirichlet_energy(X, adj)]

        for i in range(self.nlayers):
            Y = Y + self.dt*(self.act_fn(self.conv(X,edge_index) + self.residual(X)) - self.alpha*Y - self.gamma*X)
            X = X + self.dt*Y
            
            self.energies.append(dirichlet_energy(X, adj))
            
            Y = F.dropout(Y, self.dropout, training=self.training)
            X = F.dropout(X, self.dropout, training=self.training)

        X = self.dec(X)

        return X


class GraphCON_GAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, nlayers, dropout, dt=1., alpha=1., gamma=1., nheads=4):
        super(GraphCON_GAT, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.dropout = dropout
        self.nheads = nheads
        self.nhid = nhid
        self.nlayers = nlayers
        self.act_fn = nn.ReLU()
        self.res = nn.Linear(nhid, nheads * nhid)
        self.enc = nn.Linear(nfeat,nhid)
        self.conv = GATConv(nhid, nhid, heads=nheads)
        self.dec = nn.Linear(nhid,nclass)
        self.dt = dt

    def forward(self, data):
        input = data.x
        n_nodes = input.size(0)
        edge_index = data.edge_index
        input = F.dropout(input, self.dropout, training=self.training)
        Y = self.act_fn(self.enc(input))
        X = Y
        Y = F.dropout(Y, self.dropout, training=self.training)
        X = F.dropout(X, self.dropout, training=self.training)

        for i in range(self.nlayers):
            Y = Y + self.dt*(F.elu(self.conv(X, edge_index)).view(n_nodes, -1, self.nheads).mean(dim=-1) - self.alpha*Y - self.gamma*X)
            X = X + self.dt*Y
            Y = F.dropout(Y, self.dropout, training=self.training)
            X = F.dropout(X, self.dropout, training=self.training)

        X = self.dec(X)

        return X


class GraphCON_H2GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, nlayers, adj, dt=1., alpha=1., gamma=1., res_version=2):
        super(GraphCON_H2GCN, self).__init__()
        self.dropout = dropout
        self.nhid = nhid
        self.nlayers = nlayers
        self.enc = nn.Linear(nfeat,nhid)
        self.conv = H2GCN_conv(nhid, nhid, [0,1,2])
        self.dec = nn.Linear(nhid,nclass)
        self.res = nn.Linear(nhid,nhid)
        self.adj = adj
        self.dt = dt
        self.act_fn = nn.ReLU()
        self.alpha = alpha
        self.gamma = gamma
        self.reset_params()
        
        self.energies = []
        self.embeddings = []

    def reset_params(self):
        for name, param in self.named_parameters():
            if 'weight' in name and 'emb' not in name and 'out' not in name:
                stdv = 1. / math.sqrt(self.nhid)
                param.data.uniform_(-stdv, stdv)

    def forward(self, data):
        input = data.x
        edge_index = data.edge_index
        input = F.dropout(input, self.dropout, training=self.training)
        Y = self.act_fn(self.enc(input))
        X = Y
        Y = F.dropout(Y, self.dropout, training=self.training)
        X = F.dropout(X, self.dropout, training=self.training)
        
        self.energies = []
        self.embeddings = []

        for i in range(self.nlayers):
            self.energies.append(dirichlet_energy(X, self.adj))
            #self.embeddings.append( torch.cat([X,Y],dim=1) )
            Y = Y + self.dt*(self.act_fn(self.conv(X, self.adj) ) - self.alpha*Y - self.gamma*X)
            X = X + self.dt*Y
            Y = F.dropout(Y, self.dropout, training=self.training)
            X = F.dropout(X, self.dropout, training=self.training)
            
        self.energies.append(dirichlet_energy(X, self.adj))
        #self.embeddings.append( torch.cat([X,Y],dim=1) )
        X = self.dec(X)
        return X