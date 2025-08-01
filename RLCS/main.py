import argparse
import datetime
import os
# import metis
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.utils import to_undirected
from torch_geometric.utils import add_remaining_self_loops

from scipy.sparse import csr_matrix

from models.EmbLearner import EmbLearner
from models.EmbLearnerKnn import EmbLearnerKNN
from models.EmbLearnerCP import EmbLearnerCP
from utils.load_utils import hypergraph_construction, loadQuerys, load_graph
from utils.log_utils import get_logger, get_log_path
from utils.val_utils import f1_score_, NMI_score, ARI_score, JAC_score, get_res_path, get_model_path,get_comm_path
import copy
def validation(val,nodes_feats, model, edge_index, edge_index_aug):
    scorelists = []
    for q, comm in val:
        h = model((q, None, edge_index, edge_index_aug, nodes_feats))
        sim=F.cosine_similarity(h[q].unsqueeze(0),h,dim=1) #(115,)
        simlists = torch.sigmoid(sim.squeeze(0)).to(
            torch.device('cpu')).numpy().tolist()
        scorelists.append([q, comm, simlists])
    s_ = 0.1
    f1_m = 0.0
    s_m = s_
    while(s_<=0.9):
        f1_x = 0.0
        for q, comm, simlists in scorelists:
            comm_find = []
            for i, score in enumerate(simlists):
                if score >=s_ and i not in comm_find:
                    comm_find.append(i)

            comm_find = set(comm_find)
            comm_find = list(comm_find)
            comm = set(comm)
            comm = list(comm)
            f1, pre, rec = f1_score_(comm_find, comm)
            f1_x= f1_x+f1
        f1_x = f1_x/len(val)
        if f1_m<f1_x:
            f1_m = f1_x
            s_m = s_
        s_ = s_+0.05
    logger.info(f'best threshold: {s_m}, validation_set Avg F1: {f1_m}')
    return s_m, f1_m

def validation_pre(val,nodes_feats, model, edge_index, edge_index_aug):
    scorelists = []
    for q, comm in val:
        h = model((q, None, edge_index, edge_index_aug, nodes_feats))

        sim = F.cosine_similarity(h[q].unsqueeze(0), h, dim=1)  # (115,)

        simlists = torch.sigmoid(sim.squeeze(0)).to(
            torch.device('cpu')).numpy().tolist()

        scorelists.append([q, comm, simlists])
    s_ = 0.1
    pre_m = 0.0
    s_m = s_
    while (s_ <= 0.9):
        pre_x = 0.0
        for q, comm, simlists in scorelists:
            comm_find = []
            for i, score in enumerate(simlists):
                if score >= s_ and i not in comm_find:
                    comm_find.append(i)

            comm_find = set(comm_find)
            comm_find = list(comm_find)
            comm = set(comm)
            comm = list(comm)
            f1, pre, rec = f1_score_(comm_find, comm)
            pre_x = pre_x + pre
        pre_x = pre_x / len(val)
        if pre_m < pre_x:
            pre_m = pre_x
            s_m = s_
        s_ = s_ + 0.05
    logger.info(f'best threshold: {s_m}, validation_set Avg Pre: {pre_m}')
    return s_m, pre_m
def construct_augG(aug, nodes_feats, edge_index, n_nodes):
    if aug == 'hy':
        edge_index_aug, egde_attr = hypergraph_construction(edge_index, n_nodes, k=args.k)
    elif aug in ['knn']:
        sim = F.normalize(nodes_feats).mm(F.normalize(nodes_feats).T).fill_diagonal_(0.0)
        dst = sim.topk(10, 1)[1]
        src = torch.arange(nodes_feats.size(0)).unsqueeze(1).expand_as(sim.topk(10, 1)[1])
        device = edge_index.device
        src = src.to(device)
        dst = dst.to(device)
        logger.info(f"src device: {src.device}, dst device: {dst.device}, edge_index device: {edge_index.device}")
        edge_index_aug = torch.stack([src.reshape(-1), dst.reshape(-1)])
        edge_index_aug = to_undirected(edge_index_aug)
        edge_index_aug = add_remaining_self_loops(edge_index_aug, num_nodes=n_nodes)[0]
    else:
        logger.error('error aug type')
        return None
    return edge_index_aug

def load_citations(args):
    if args.dataset.startswith('stb_'):
        dataset = args.dataset[4:]
    else:
        dataset = args.dataset
    '*************loda feature data************'
    if args.dataset in ['cocs','photo','cora','pubmed']:
        with open(f'{args.root}/{args.dataset}/{dataset}.feats', "r") as f:
            nodes_feats = np.array([list(map(float, line.strip().split())) for line in f],dtype=np.float32)
            nodes_feats = torch.from_numpy(nodes_feats)
            node_in_dim = nodes_feats.shape[1]
    elif args.dataset in ['ex-fb']:
        feats_array = np.loadtxt(f'{args.root}/{args.dataset}/{args.dataset}.feat', delimiter=' ', dtype=np.float32)
        print(type(feats_array))
        nodes_feats = torch.from_numpy(feats_array)
        node_in_dim = nodes_feats.shape[1]
    elif args.dataset in ['facebook']:
        feats_array = np.loadtxt(f'{args.root}/{args.dataset}/{args.dataset}.feat', dtype=float, delimiter=' ')
        nodes_feats = torch.tensor(feats_array, dtype=torch.float32)
        node_in_dim = nodes_feats.shape[1]
    else:
        print('loda feature error')
    '''********************load_graph******************************'''
    graphx,n_nodes = load_graph(args.root,args.dataset,args.attack,args.ptb_rate)
    src = []
    dst = []
    for id1, id2 in graphx.edges:
        src.append(id1)
        dst.append(id2)
        src.append(id2)
        dst.append(id1)
    num_nodes = graphx.number_of_nodes()
    adj_matrix = csr_matrix(([1] * len(src), (src, dst)), shape=(num_nodes, num_nodes))
    calhyper_start = datetime.datetime.now()
    edge_index = torch.tensor([src, dst])

    edge_index_aug = construct_augG(args.aug, nodes_feats, edge_index, n_nodes)


    train, val, test = loadQuerys(dataset, args.root, args.train_size, args.val_size, args.test_size,
                                  args.train_path, args.test_path, args.val_path)

    return nodes_feats, train, val, test, node_in_dim, n_nodes, edge_index, edge_index_aug, adj_matrix

def Community_Search(args,logger):

    preprocess_start = datetime.datetime.now()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'device: {device}')

    nodes_feats, train, val, test, node_in_dim, n_nodes, edge_index, edge_index_aug, adj_matrix = load_citations(args)
    logger.info(f'load_time = {datetime.datetime.now() - preprocess_start}, train len = {len(train)}')
    nodes_feats = nodes_feats.to(device)
    edge_index = edge_index.to(device)
    edge_index_aug = edge_index_aug.to(device)

    if args.method == 'EmbLearner':
        embLearner = EmbLearner(node_in_dim, args.hidden_dim, args.num_layers, args.drop_out, args.tau, device,args.alpha, args.lam, args.k)

    elif args.method == 'EmbLearnerKNN':
        embLearner = EmbLearnerKNN(node_in_dim, args.hidden_dim, args.num_layers, args.drop_out, args.tau, device,args.alpha, args.lam, args.k)
        # embLearner = EmbLearnerCP(node_in_dim, args.hidden_dim, args.num_layers, args.drop_out, args.tau, device,args.alpha, args.lam, args.k)

    # elif args.method == 'Nohy':
    #     embLearner = EmbLearnerWithoutHyper(node_in_dim, args.hidden_dim, args.num_layers, args.drop_out, args.tau,device, args.alpha, args.lam, args.k)
    # elif args.method == 'EmbLearnerwithWeights': #将这个作为我的
    #     embLearner = EmbLearnerwithWeights(node_in_dim, args.hidden_dim,args.num_layers,args.drop_out,args.tau,device,args.alpha,args.lam,args.k)
    else:
        raise ValueError(f'method {args.method} not supported')

    logger.info(f'embLearner: {args.method}')

    emb_optim = torch.optim.Adam(embLearner.parameters(), lr=args.lr,weight_decay=args.weight_decay)
    embLearner.to(device)

    pre_process_time = (datetime.datetime.now() - preprocess_start).seconds
    val_best_f1 = 0.
    val_bst_pre = 0.
    val_bst_f1_ep = 0
    val_bst_pre_ep = 0
    val_bst_f1_model = copy.deepcopy(embLearner.state_dict())
    val_bst_pre_model = copy.deepcopy(embLearner.state_dict())
    val_epochs_time=0.0

    train_start = datetime.datetime.now()
    #warm
    for epoch in range(args.warm):
        embLearner.train()
        start = datetime.datetime.now()
        loss_b = 0.0
        i = 0
        for q, pos, comm in train:
            if len(pos) == 0:
                i = i + 1
                continue
            loss,h = embLearner((q, pos, edge_index, edge_index_aug, nodes_feats))

            l2_lambda = 1e-4
            l2_reg = torch.tensor(0.0, device=h.device)
            for param in embLearner.parameters():  # or model.mlp1.parameters() 仅正则 MLP
                if param.requires_grad:
                    l2_reg += torch.norm(param, p=2) ** 2

            loss = loss + l2_lambda * l2_reg

            loss_b = loss_b + loss.item()
            loss.backward()
            if (i + 1) % args.batch_size == 0:
                emb_optim.step()
                emb_optim.zero_grad()
            i = i + 1
        epoch_time = (datetime.datetime.now() - start).seconds
        logger.info(f'epoch_loss = {loss_b}, epoch = {epoch}, epoch_time = {epoch_time}')

    lst_threshold = torch.tensor(0.0,device = device)
    for epoch in range(args.epoch_n):
        embLearner.train()
        start = datetime.datetime.now()
        loss_b = 0.0
        i = 0
        for q, pos, comm in train:
            if len(pos) == 0:
                i = i + 1
                continue
            loss,h = embLearner((q, pos, edge_index, edge_index_aug, nodes_feats))
            l2_lambda = 1e-4
            l2_reg = torch.tensor(0.0, device=h.device)
            for param in embLearner.parameters():  # or model.mlp1.parameters() 仅正则 MLP
                if param.requires_grad:
                    l2_reg += torch.norm(param, p=2) ** 2

            loss = loss + l2_lambda * l2_reg

            loss_b = loss_b + loss.item()
            loss.backward()
            if (i + 1) % args.batch_size == 0:
                emb_optim.step()
                emb_optim.zero_grad()
            i = i + 1
        epoch_time = (datetime.datetime.now() - start).seconds
        logger.info(f'epoch_loss = {loss_b}, epoch = {epoch}, epoch_time = {epoch_time}')

        embLearner.eval()
        with torch.no_grad():
            val_start = datetime.datetime.now()
            s_,f1_ = validation(val,nodes_feats,embLearner,edge_index, edge_index_aug)
            sp_,pre_ = validation_pre(val,nodes_feats,embLearner,edge_index, edge_index_aug)
            val_time = (datetime.datetime.now() - val_start).seconds
            val_epochs_time= val_epochs_time+val_time
            print(f'f1_:{f1_},pre_:{pre_}')
        if f1_ > val_best_f1:
            val_best_f1 = f1_
            val_bst_f1_ep = epoch
            val_bst_f1_model = copy.deepcopy(embLearner.state_dict())
            val_bst_f1_time=(datetime.datetime.now() - train_start).seconds -val_epochs_time
        if pre_ > val_bst_pre:
            val_bst_pre = pre_
            val_bst_pre_ep = epoch
            val_bst_pre_model = copy.deepcopy(embLearner.state_dict())
            val_bst_pre_time=(datetime.datetime.now() - train_start).seconds -val_epochs_time #

        if args.enable_refine:
            with torch.no_grad():
                cos_sim = F.cosine_similarity(h[edge_index[0]], h[edge_index[1]], dim=1)
                edge_probs = torch.sigmoid(cos_sim)  # ∈ (0, 1)，表示边存在的概率
                tau = args.sim_tau
                beta = args.smooth_beta
                quantile_t = torch.quantile(edge_probs,1-tau)
                dynamic_threashold = beta*quantile_t +(1-beta) *lst_threshold
                mask = edge_probs > dynamic_threashold
                old_edge_count = edge_index.size(1)
                edge_index = edge_index[:,mask]
                lst_threshold = dynamic_threashold.detach()
            # logger.info(f'[Refine] epoch={epoch}, kept edges: {mask.sum().item()} / {old_edge_count} with threshold={args.sim_threshold}')


    training_time = (datetime.datetime.now() - train_start).seconds-val_epochs_time
    logger.info(f'===best F1 at epoch {val_bst_f1_ep}, Best F1:{val_best_f1} ===,Best epoch time:{val_bst_f1_time}')
    logger.info(f'===best Pre at epoch {val_bst_pre_ep}, Best Precision:{val_bst_pre} ===,Best epoch time:{val_bst_pre_time}')
    logger.info(f'trainning time = {training_time},validate time ={val_epochs_time}')

    bst_model_path = get_model_path('./results/emb/res_model/',args)
    torch.save(val_bst_f1_model, f'{bst_model_path}_f1.pkl')
    torch.save(val_bst_pre_model,f'{bst_model_path}_pre.pkl')


    logger.info(f'#################### Starting evaluation######################')

    if args.val_type == 'pre':
        embLearner.load_state_dict(torch.load(f'{bst_model_path}_pre.pkl'))
    else:
        embLearner.load_state_dict(torch.load(f'{bst_model_path}_f1.pkl'))
    embLearner.eval()

    F1 = 0.0
    Pre = 0.0
    Rec = 0.0

    nmi_score = 0.0
    ari_score = 0.0
    jac_score = 0.0
    count = 0.0

    eval_start = datetime.datetime.now()

    with torch.no_grad():

        if args.val_type == 'f1':
            s_, f1_ = validation(val, nodes_feats, embLearner, edge_index, edge_index_aug)
            logger.info(f'evaluation time = {datetime.datetime.now() - eval_start}, best s_={s_}, best val f1_={f1_}')
        elif args.val_type == 'pre':
            s_, pre_ = validation_pre(val, nodes_feats, embLearner, edge_index, edge_index_aug)
            logger.info(f'evaluation time = {datetime.datetime.now() - eval_start}, best s_={s_}, best val pre_={pre_}')
        val_running_time = (datetime.datetime.now() - eval_start).seconds  # 结束了测试运行的时间

        logger.info(f'#################### starting test  ####################')
        test_start = datetime.datetime.now()

        comm_path = get_comm_path('./results/emb/', args)
        with open(comm_path, 'a', encoding='utf-8') as f:
            for q, comm in test:
                h = embLearner((q, None, edge_index, edge_index_aug, nodes_feats))
                count = count + 1
                sim = F.cosine_similarity(h[q].unsqueeze(0), h, dim=1)
                simlists = torch.sigmoid(sim.squeeze(0)).to(torch.device('cpu')).numpy().tolist()

                comm_find = []
                for i, score in enumerate(simlists):
                    if score >= s_ and i not in comm_find:
                        comm_find.append(i)

                comm_find = set(comm_find)
                comm_find = list(comm_find)

                line = str(q) + "," + " ".join(str(u) for u in comm_find)
                f.write(line + "\n")
                comm = set(comm)
                comm = list(comm)
                f1, pre, rec = f1_score_(comm_find, comm)
                F1 = F1 + f1
                Pre = Pre + pre
                Rec = Rec + rec
                nmi = NMI_score(comm_find, comm, n_nodes)
                nmi_score = nmi_score + nmi

                ari = ARI_score(comm_find, comm, n_nodes)
                ari_score = ari_score + ari

                jac = JAC_score(comm_find, comm, n_nodes)
                jac_score = jac_score + jac

    test_running_time = (datetime.datetime.now() - test_start).seconds

    F1 = F1 / len((test))
    Pre = Pre / len((test))
    Rec = Rec / len((test))
    nmi_score = nmi_score / len(test)
    ari_score = ari_score / len(test)
    jac_score = jac_score / len(test)
    logger.info(f'Test time = {test_running_time}')
    logger.info(f'Test_set Avg：F1 = {F1}, Pre = {Pre}, Rec = {Rec}, s = {s_}')
    logger.info(f'Test_set Avg NMI = {nmi_score}, ARI = {ari_score}, JAC = {jac_score}')

    # 存储测试结果
    output = get_res_path('./results/emb/', args)
    with open(output, 'a+',encoding='utf-8') as fh:
        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        line = (
            f"args: {args}\n"
            f"val_type:{args.val_type}"
            f"bst_f1_epoch:{val_bst_f1_ep}, bst_ep_time:{val_bst_f1_time}, bst_ep_f1:{val_best_f1}\n"
            f"bst_pre_epoch:{val_bst_pre_ep}, bst_ep_time:{val_bst_pre_time}, bst_ep_f1:{val_bst_pre}\n"
            f"best_comm_threshold: {s_}\n"
            f"pre_process_time: {pre_process_time}\n"
            f"training_time: {training_time}\n"
            f"val_time:{val_running_time}"
            f"test_running_time: {test_running_time}\n"
            f"F1: {F1}\n"
            f"Pre: {Pre}\n"
            f"Rec: {Rec}\n"
            f"nmi_score: {nmi_score}\n"
            f"ari_score: {ari_score}\n"
            f"jac_score: {jac_score}\n"
            f"current_time: {current_time}\n"
            "----------------------------------------\n"
        )
        fh.write(line)
        fh.close()
    return F1, Pre, Rec, nmi_score, ari_score, jac_score, pre_process_time, training_time,val_time, test_running_time


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--count', type=int, default=1)
    parser.add_argument('--root', type=str, default='./data')

    parser.add_argument("--log", type=bool,default=False, help='run prepare_data or not')

    parser.add_argument('--method',type=str,default='EmbLearnerKNN')
    parser.add_argument('--model_path', type=str, default='CS')
    parser.add_argument('--m_model_path', type=str, default='META')


    parser.add_argument('--dataset', type=str, default='cora')

    parser.add_argument('--train_size', type=int, default=300)
    parser.add_argument('--val_size', type=int, default=100)
    parser.add_argument('--test_size', type=int, default=500)
    parser.add_argument('--train_path', type=str, default='pos_train')
    parser.add_argument('--test_path', type=str, default='test')
    parser.add_argument('--val_path', type=str, default='val')
    parser.add_argument('--feats_path', type=str, default='feats.txt')
    parser.add_argument('--val_type', type=str, default='f1',help='pre or f1 to val')


    parser.add_argument('--attack', type=str, default='none')
    parser.add_argument('--type', type=str, default='add', help='random attack type', choices=['add', 'remove', 'flip'])
    parser.add_argument('--noise_level', type=int, default=3, choices=[1, 2, 3], help='noisy level')
    parser.add_argument('--ptb_rate', type=float, default=0.30, help='pertubation rate')


    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--epoch_n', type=int, default=100)
    parser.add_argument('--drop_out', type=float, default=0.1)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=0.0005)

    parser.add_argument('--tau', type=float, default=0.2)
    parser.add_argument('--lam', type=float, default=0.0001)
    parser.add_argument('--warm', type=int, default=2)

    parser.add_argument('--k', type=int, default=2)
    parser.add_argument('--aug', type=str, default='knn')
    parser.add_argument('--alpha', type=float, default=0.0001)
    parser.add_argument('--epsilon', type=float, default=0.05)

    parser.add_argument('--enable_refine', type=bool, default=False)
    parser.add_argument('--sim_threshold', type=float, default=0.3)
    parser.add_argument('--sim_tau', type=float, default=0.8, help='Quantile for edge pruning')
    parser.add_argument('--smooth_beta', type=float, default=0.7, help='Smoothing factor for dynamic threshold')

    args = parser.parse_args()
    if args.log:
        log_path = get_log_path('./log/', args)
        logger = get_logger(log_path)
        print(f'save logger to {log_path}')
    else:
        logger = get_logger('logs/log.log')


    pre_process_time_A, train_model_running_time_A,val_time_A, test_running_time_A = 0.0, 0.0,0.0, 0.0
    count = 0
    F1lists = []
    Prelists = []
    Reclists = []
    nmi_scorelists = []
    ari_scorelists = []
    jac_scorelists = []

    for i in range (args.count):
        count = count + 1

        now = datetime.datetime.now()
        logger.info(f'##{count}: Starting Time: {now.strftime("%Y-%m-%d %H:%M:%S")}')


        F1, Pre, Rec, nmi_score, ari_score, jac_score, pre_process_time, train_model_running_time,val_time,test_running_time = \
            Community_Search(args,logger)


        logger.info(f'##{count}: Finishing Time: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
        running_time = (datetime.datetime.now() - now).seconds

        logger.info(f'##{count}:Running Time(s): {running_time}')
        print('= ' * 20)

        F1lists.append(F1)
        Prelists.append(Pre)
        Reclists.append(Rec)
        nmi_scorelists.append(nmi_score)
        ari_scorelists.append(ari_score)
        jac_scorelists.append(jac_score)

        pre_process_time_A = pre_process_time_A + pre_process_time
        train_model_running_time_A = train_model_running_time_A + train_model_running_time
        val_time_A = val_time_A+val_time
        test_running_time_A = test_running_time_A + test_running_time

    # 计算count次数的各个评价指标的均值和方差
    F1_std = np.std(F1lists)
    F1_mean = np.mean(F1lists)
    Pre_std = np.std(Prelists)
    Pre_mean = np.mean(Prelists)
    Rec_std = np.std(Reclists)
    Rec_mean = np.mean(Reclists)
    nmi_std = np.std(nmi_scorelists)
    nmi_mean = np.mean(nmi_scorelists)
    ari_std = np.std(ari_scorelists)
    ari_mean = np.mean(ari_scorelists)
    jac_std = np.std(jac_scorelists)
    jac_mean = np.mean(jac_scorelists)


    pre_process_time_A = pre_process_time_A / float(args.count)
    train_model_running_time_A = train_model_running_time_A / float(args.count)
    val_time_A = val_time_A / float(args.count)
    test_running_time_A = test_running_time_A / float(args.count)
    single_query_time = test_running_time_A/float(args.test_size)


    output = get_res_path('./results/res/', args)
    with open(output, 'a+',encoding='utf-8') as fh:
        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = (
            f"average {args}\n"
            f"pre_process_time: {pre_process_time_A}\n"
            f"train_model_running_time: {train_model_running_time_A}\n"
            f"val_time_A: {val_time_A}\n"
            f"test_running_time: {test_running_time_A}\n"
            f"single_query_time: {single_query_time}\n"
            f"F1 mean: {F1_mean}\n"
            f"F1 std: {F1_std}\n"
            f"Pre mean: {Pre_mean}\n"
            f"Pre std: {Pre_std}\n"
            f"Rec mean: {Rec_mean}\n"
            f"Rec std: {Rec_std}\n"
            f"nmi_score mean: {nmi_mean}\n"
            f"nmi std: {nmi_std}\n"
            f"ari_score mean: {ari_mean}\n"
            f"ari std: {ari_std}\n"
            f"jac mean: {jac_mean}\n"
            f"jac std: {jac_std}\n"
            f"current_time: {current_time}\n"
            "----------------------------------------\n"
        )
        fh.write(line)
        fh.close()