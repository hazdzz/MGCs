import math
import numpy as np
import scipy.sparse as sp

import torch

def calc_sym_renorm_mag_adj(dir_adj, g):
    n_vertex = dir_adj.shape[0]
    id = sp.csc_matrix(sp.identity(n_vertex))

    adj = dir_adj + dir_adj.T.multiply(dir_adj.T > dir_adj) - dir_adj.multiply(dir_adj.T > dir_adj)
    adj_ = adj + id

    row_sum = adj_.sum(axis=1).A1
    row_sum_inv_sqrt = np.power(row_sum, -0.5)
    row_sum_inv_sqrt[np.isinf(row_sum_inv_sqrt)] = 0.
    deg_inv_sqrt_ = sp.diags(row_sum_inv_sqrt)

    sym_renorm_adj = deg_inv_sqrt_.dot(adj_).dot(deg_inv_sqrt_)
    
    if g == 0:
        sym_renorm_mag_adj = sym_renorm_adj
    else:
        trs = np.exp(1j * 2 * math.pi * g * (dir_adj - dir_adj.T).toarray())
        #trs = np.exp(1j * 2 * math.pi * g * (dir_adj.T - dir_adj).toarray())
        sym_renorm_mag_adj = np.multiply(sym_renorm_adj.toarray(), trs)
        sym_renorm_mag_adj = sp.csc_matrix(sym_renorm_mag_adj)

    return sym_renorm_mag_adj

def calc_rw_renorm_mag_adj(dir_adj, g):
    n_vertex = dir_adj.shape[0]
    id = sp.csc_matrix(sp.identity(n_vertex))

    adj = dir_adj + dir_adj.T.multiply(dir_adj.T > dir_adj) - dir_adj.multiply(dir_adj.T > dir_adj)
    adj_ = adj + id

    row_sum = adj_.sum(axis=1).A1
    row_sum_inv = np.power(row_sum, -1)
    row_sum_inv[np.isinf(row_sum_inv)] = 0.
    deg_inv_ = sp.diags(row_sum_inv)

    rw_renorm_adj = deg_inv_.dot(adj_)

    if g == 0:
        rw_renorm_mag_adj = rw_renorm_adj
    else:
        trs = np.exp(1j * 2 * math.pi * g * (dir_adj - dir_adj.T).toarray())
        #trs = np.exp(1j * 2 * math.pi * g * (dir_adj.T - dir_adj).toarray())
        rw_renorm_mag_adj = np.multiply(rw_renorm_adj.toarray(), trs)
        rw_renorm_mag_adj = sp.csc_matrix(rw_renorm_mag_adj)

    return rw_renorm_mag_adj

def calc_mgc_features(renorm_mag_adj, features, alpha, t, K, g):
    n_vertex = renorm_mag_adj.shape[0]
    id = sp.csc_matrix(sp.identity(n_vertex))
    
    a = (1 - alpha) * t / ((1 - alpha) * t + 1)
    b = (1 - alpha * t) / ((1 - alpha) * t + 1)
    c = alpha * t / ((1 - alpha) * t + 1)

    mgc_adj_denom = a * renorm_mag_adj
    mgc_adj_num = b * id + c * renorm_mag_adj
    
    mgc_features = features
    for k in range(1, K):
        mgc_features = mgc_adj_denom.dot(mgc_features) + mgc_adj_num.dot(features)

    if g == 0:
        mgc_features = np.array(mgc_features.toarray(), dtype=np.float32)
    else:
        mgc_features = np.array(mgc_features.toarray(), dtype=np.complex64)
    
    return mgc_features

'''
def cnv_sparse_mat_to_coo_tensor(sp_mat, device):
    # convert a compressed sparse row (csr) or compressed sparse column (csc) matrix to a hybrid sparse coo tensor
    sp_coo_mat = sp_mat.tocoo().astype(np.float32)
    i = torch.from_numpy(np.vstack((sp_coo_mat.row, sp_coo_mat.col)).astype(np.int32))
    v = torch.from_numpy(sp_coo_mat.data)
    s = torch.Size(sp_coo_mat.shape)

    return torch.sparse_coo_tensor(indices=i, values=v, size=s, dtype=torch.float32, device=device, requires_grad=False)
'''

def calc_accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double().sum()
    accuracy = correct / len(labels)

    return accuracy