import os

import numpy as np
import torch
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score, jaccard_score
import torch.nn.functional as F
from utils.log_utils import get_logger

def f1_score_(comm_find, comm):
    lists = [x for x in comm_find if x in comm] #TP
    if len(lists) == 0:
        return 0.0, 0.0, 0.0
    pre = len(lists) * 1.0 / len(comm_find) #pre = TP/(TP+FP) = TP/comm_find
    rec = len(lists) * 1.0 / len(comm) #recall = TP/(TP+FN) = TP/comm
    #ACC= (TP+TN)/(TP+TN+FP+FN)
    f1 = 2 * pre * rec / (pre + rec) # F1=2*P*R/(P+R)
    return f1, pre, rec

def NMI_score(comm_find, comm, n_nodes):

    truthlabel = np.zeros((n_nodes), dtype=int)
    truthlabel[comm] = 1
    prelabel = np.zeros((n_nodes), dtype=int)
    prelabel[comm_find] = 1
    score = normalized_mutual_info_score(truthlabel, prelabel)
    #print("q, nmi:", score)
    return score

def ARI_score(comm_find, comm, n_nodes):

    truthlabel = np.zeros((n_nodes), dtype=int)
    truthlabel[comm] = 1
    prelabel = np.zeros((n_nodes), dtype=int)
    prelabel[comm_find] = 1
    score = adjusted_rand_score(truthlabel, prelabel)
    #print("q, ari:", score)

    return score

def JAC_score(comm_find, comm, n_nodes):
    truthlabel = np.zeros((n_nodes), dtype=int)
    truthlabel[comm] = 1
    prelabel = np.zeros((n_nodes), dtype=int)
    prelabel[comm_find] = 1
    score = jaccard_score(truthlabel, prelabel)
    return score

def get_res_path(resroot,args):
    if args.attack == 'meta':
        return f'{resroot}{args.dataset}/{args.dataset}_{args.aug}_{args.attack}_{args.ptb_rate}_{args.method}_res.txt'
    elif args.attack == 'random':
        return f'{resroot}{args.dataset}/{args.dataset}_{args.aug}_{args.attack}_{args.type}_{args.ptb_rate}_{args.method}_res.txt'
    # elif args.attack =='add':
    #     return f'{resroot}{args.dataset}_{args.aug}_{args.attack}_{args.noise_level}_{args.method}_res.txt'
    elif args.attack in  ['del','gflipm','gdelm','add','gaddm']:
        return f'{resroot}{args.dataset}/{args.dataset}_{args.aug}_{args.attack}_{args.ptb_rate}_{args.method}_res.txt'
    else:
        return f'{resroot}{args.dataset}/{args.dataset}_{args.aug}_{args.method}_res.txt'

def get_model_path(model_dir,args):
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    if args.attack == 'meta':
        return f'{model_dir}{args.dataset}/{args.dataset}_{args.aug}_{args.attack}_{args.ptb_rate}_{args.method}'
    elif args.attack == 'random': #random attack
        return f'{model_dir}{args.dataset}/{args.dataset}_{args.aug}_{args.attack}_{args.type}_{args.ptb_rate}_{args.method}'
    # elif args.attack =='add': #noisy graph
    #     return f'{model_dir}{args.dataset}_{args.attack}_{args.noise_level}_{args.method}.pkl'
    elif args.attack in  ['del','gflipm','gdelm','add','gaddm']: #incomplete graph
        return f'{model_dir}{args.dataset}/{args.dataset}_{args.aug}_{args.attack}_{args.ptb_rate}_{args.method}'
    else:
        return f'{model_dir}{args.dataset}/{args.dataset}_{args.aug}_{args.method}'
def get_comm_path(resroot,args):
    if args.dataset =='cora':
        if args.lam ==0.001 and args.alpha ==0.001:
            newroot = './Case/emb/comm'
        else:
            newroot =f'{resroot}comm'
    elif args.dataset in ['ex-fb','wfb107','facebook']:
        if args.lam == 0.2 and args.alpha == 0.2:
            newroot = './Case/emb/comm'
        else:
            newroot = f'{resroot}comm'
    elif args.dataset in ['cora_gsr','cora_stb','citeseer_gsr','citeseer_stb','fb107_gsr','fb107_stb','facebook_gsr','facebook_stb','cocs_gsr','cocs_stb','photo_gsr','photo_stb','dblp_gsr','dblp_stb']:
        newroot = './Case/emb/comm'
    else:
        newroot = './Case/emb/comm'

    if not os.path.exists(newroot):
        os.makedirs(newroot)

    if args.attack =='none':
        return f'{newroot}/{args.dataset}_{args.method}_res.txt'
    else: #没有攻击
        return f'{newroot}/{args.dataset}_{args.attack}_{args.ptb_rate}_{args.method}_res.txt'