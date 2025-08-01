import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn.inits import glorot
from torch_scatter import scatter

def get_perturbed_embedding(h_nf, h_kf, epsilon=0.2, mask_prob=0.3):
    """
    h_nf: Tensor [N, d] - representation from original view
    h_kf: Tensor [N, d] - representation from augmented view
    epsilon: float - perturbation strength
    mask_prob: float - probability to keep each dimension
    """
    z = torch.cat([h_nf, h_kf], dim=1)  # [N, 2d]
    z_shuf = z[torch.randperm(z.size(0))]  # shuffle

    mask = (torch.rand_like(z_shuf) > mask_prob).float()  # element-wise mask
    z_masked = z_shuf * mask

    z_perturbed = z + epsilon * z_masked
    return z_perturbed  # shape [N, 2d]

class MLP(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(MLP, self).__init__()
        self.fcs = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, x):
        return self.fcs(x)


class Contra(nn.Module):
    def __init__(self, device):
        super(Contra, self).__init__()
        self.device =device
    def forward(self, h, h_aug, tau, train, alpha, lam, edge_index):
        if self.training ==False:
            return h
        q,pos = train

        h_q = h[q].unsqueeze(0)
        intra_cosine_sim = F.cosine_similarity(h_q, h, dim=1)
        intra_cosine_sim = torch.exp(intra_cosine_sim / tau)


        aug_intra_cosine_sim = F.cosine_similarity(h_aug[q].unsqueeze(0), h_aug, dim=1)  # sim_aug_1=(115,)
        aug_intra_cosine_sim = torch.exp(aug_intra_cosine_sim / tau)

        inter_cosine_sim = F.cosine_similarity(h[q].unsqueeze(0), h_aug, dim=1)
        inter_cosine_sim = torch.exp(inter_cosine_sim / tau)

        aug_inter_cosine_sim = F.cosine_similarity(h_aug[q].unsqueeze(0), h, dim=1)
        aug_inter_cosine_sim = torch.exp(aug_inter_cosine_sim / tau)

        mask_p = [False] * h.shape[0]  # (115,)
        mask_p = torch.tensor(mask_p)
        mask_p.to(self.device)
        mask_p[pos] = True
        mask_p[q] = False

        intra_loss = torch.tensor([0.0]).to(self.device)  # (1,)
        aug_intra_loss = torch.tensor([0.0]).to(self.device)
        inter_loss = torch.tensor([0.0]).to(self.device)
        aug_inter_loss = torch.tensor([0.0]).to(self.device)

        if len(pos) !=0:
            intra_loss = intra_cosine_sim.squeeze(0)[mask_p] / (torch.sum(intra_cosine_sim.squeeze(0)))  # intra_loss=(3,)
            intra_loss = -torch.log(intra_loss).mean()
            aug_intra_loss = aug_intra_cosine_sim.squeeze(0)[mask_p] / (torch.sum(aug_intra_cosine_sim.squeeze(0)))
            aug_intra_loss = -torch.log(aug_intra_loss).mean()

            inter_loss = inter_cosine_sim.squeeze(0)[mask_p] / (torch.sum(inter_cosine_sim.squeeze(0)))
            inter_loss = -torch.log(inter_loss).mean()
            aug_inter_loss = aug_inter_cosine_sim.squeeze(0)[mask_p] / (torch.sum(aug_inter_cosine_sim.squeeze(0)))
            aug_inter_loss = -torch.log(aug_inter_loss).mean()

        total_intra_loss = 0.5*(intra_loss+aug_intra_loss)
        total_inter_loss = 0.5*(inter_loss+aug_inter_loss)


        z_unsup = -torch.log(inter_cosine_sim.squeeze(0)[q] / torch.sum(inter_cosine_sim.squeeze(0)))
        z_aug_unsup = -torch.log(aug_inter_cosine_sim.squeeze(0)[q] / torch.sum(aug_inter_cosine_sim.squeeze(0)))
        loss_unsup = 0.5 * z_unsup + 0.5 * z_aug_unsup

        z_perturb = get_perturbed_embedding(h, h_aug, epsilon=0.2, mask_prob=0.3)
        z_target = torch.cat([h, h_aug], dim=1)
        inter_sim_new = F.cosine_similarity(z_perturb[q].unsqueeze(0), z_target, dim=1)
        unsup_1 = -torch.log(inter_sim_new[q] / inter_sim_new.sum())

        loss = (total_intra_loss + lam * total_inter_loss) + alpha * unsup_1
        return loss

class EmbLearnerKNN(nn.Module):
    def __init__(self, node_in_dim, hidden_dim, num_layers, dropout, tau, device, alpha, lam, k):
        super(EmbLearnerKNN, self).__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        self.tau = tau #
        self.alpha = alpha
        self.lam = lam
        self.k = k
        self.device = device

        self.contra = Contra(device)

        self.query_layers = nn.ModuleList()
        self.query_layers.append(GCNConv(1, hidden_dim))
        for _ in range(num_layers - 1):
            self.query_layers.append(GCNConv(hidden_dim, hidden_dim))

        self.layers = nn.ModuleList()
        self.layers.append(GCNConv(node_in_dim, hidden_dim))
        for _ in range(num_layers - 1):
            self.layers.append(GCNConv(hidden_dim, hidden_dim))


        self.last_fusion_layers = nn.ModuleList()
        self.last_fusion_layers.append(GCNConv(hidden_dim, hidden_dim))
        for _ in range(num_layers - 1):
            self.last_fusion_layers.append(GCNConv(hidden_dim, hidden_dim))


        self.query_atts = []
        self.atts = []
        for _ in range(num_layers):

            query_att = nn.Parameter(torch.empty(size=(hidden_dim, 1)), requires_grad=True)  # 可学习的参数，
            glorot(query_att)
            #graph encoder参数初始化
            att = nn.Parameter(torch.empty(size=(hidden_dim, 1)), requires_grad=True)
            glorot(att)
            self.query_atts.append(query_att)
            self.atts.append(att)

        #初始没有前一层的hf，因此融合x和q作为前一层hf
        self.first_query_att = nn.Parameter(torch.empty(size=(hidden_dim, 1)), requires_grad=True)
        glorot(self.first_query_att)
        self.first_att = nn.Parameter(torch.empty(size=(hidden_dim, 1)), requires_grad=True)
        glorot(self.first_att)



        self.mlp1 = MLP(hidden_dim, hidden_dim)

        #query encoder全连接层
        self.linerquerys = torch.nn.Linear(1, hidden_dim)
        #graph encoder 全连接层
        self.linerfeats = torch.nn.Linear(node_in_dim, hidden_dim)

    def q_att_layer(self, x, layer):
        return torch.matmul(x, self.query_atts[layer].to(self.device))

    def att_layer(self, x, layer):
        return torch.matmul(x, self.atts[layer].to(self.device))

    def first_q_att_layer(self, x):
        return torch.matmul(x, self.first_query_att.to(self.device))
    def first_att_layer(self, x):
        return torch.matmul(x, self.first_att.to(self.device))
    def hyperedge_representation(self, x, edge_index):

        h = x
        edges = h[edge_index[0]]
        nodes = h[edge_index[1]]

        sim = torch.exp(torch.cosine_similarity(edges, nodes))

        denominator = scatter(sim, edge_index[1], dim=0, reduce='sum')
        denominator = denominator[edge_index[1]]
        sim = (sim/denominator).unsqueeze(1)

        edges_ = x[edge_index[0]]
        edges_ = sim * (edges_)

        hyperedge = scatter(edges_, edge_index[1], dim=0, reduce='sum')

        return hyperedge
    def compute_loss(self,train):

        loss = None
        q, pos, edge_index, edge_index_aug, feats = train
        querys = torch.zeros(feats.shape[0], 1).to(self.device)
        querys[q] = 1.0

        hq = F.relu(self.query_layers[0](querys, edge_index)).to(self.device)  # hq=115*256 query_encoder[0]
        h = F.relu(self.layers[0](feats, edge_index)).to(self.device)  # h=115*256  graph_encoder[0]
        atten_co = torch.cat([self.q_att_layer(hq, 0), self.att_layer(h, 0)], 1) #fusion_attention[0]
        atten_co = F.softmax(atten_co, dim=1).unsqueeze(2)
        hf = torch.stack([hq, h], dim=1)
        hf = atten_co * hf
        hf = torch.sum(hf, dim=1)

        #aug encoder
        h_augq = F.relu(self.query_layers[0](querys, edge_index_aug))  # querys(115,1)---GCNConV(1,256)-->h_augq(115,256)
        h_aug = F.relu(self.layers[0](feats, edge_index_aug))  # feats(115,1) --GCNConV(1,256)-->h_aug(115,256)
        atten_coh = torch.cat([self.q_att_layer(h_augq, 0), self.att_layer(h_aug, 0)], 1)  # (115,2)
        atten_coh = F.softmax(atten_coh, dim=1).unsqueeze(2)  # atten_coh=(115,2,1)
        h_augf = torch.stack([h_augq, h_aug], dim=1)  # h_augf(115,2,256)
        h_augf = atten_coh * h_augf
        h_augf = torch.sum(h_augf, dim=1)  # h_augf=(115,256)

        querys = self.linerquerys(querys)  # querys=(115,256)
        feats = self.linerfeats(feats)  # feats=(115,256)
        atten_co_ = torch.cat([self.first_q_att_layer(querys), self.first_att_layer(feats)], 1)
        atten_co_ = F.softmax(atten_co_, dim=1).unsqueeze(2)  # atten_co_=(115,2,1)
        hf_ = torch.stack([querys, feats], dim=1)  # hf_=(115,2,256)
        hf_ = atten_co_ * hf_
        hf_ = torch.sum(hf_, dim=1)  # hf_=(115,256)

        hf = F.relu(hf + self.last_fusion_layers[0](hf_, edge_index))  # fusion_encoder[0]

        atten_coh_ = torch.cat([self.first_q_att_layer(querys), self.first_att_layer(feats)],1)  # atten_coh_=（115，2）
        atten_coh_ = F.softmax(atten_coh_, dim=1).unsqueeze(2)  # atten_coh_=(115,2,1)
        hfh_ = torch.stack([querys, feats], dim=1)  # hfh_=(115,2,256)
        hfh_ = atten_coh_ * hfh_
        hfh_ = torch.sum(hfh_, dim=1)  # hfh_=(115,256)

        h_augf = F.relu(h_augf + self.last_fusion_layers[0](hfh_, edge_index_aug))


        for _ in range(self.num_layers - 2):
            hq = F.dropout(hq, training=self.training, p=self.dropout)
            h = F.dropout(h, training=self.training, p=self.dropout)
            hf = F.dropout(hf, training=self.training, p=self.dropout)
            h_augq = F.dropout(h_augq, training=self.training, p=self.dropout)
            h_aug = F.dropout(h_aug, training=self.training, p=self.dropout)
            h_augf = F.dropout(h_augf, training=self.training, p=self.dropout)

            hq = F.relu(self.query_layers[_ + 1](hq, edge_index)) # query_encoder[_+1]
            h = F.relu(self.layers[_ + 1](h, edge_index)) #graph_encoder[_+1]
            atten_co = torch.cat([self.q_att_layer(hq, _ + 1), self.att_layer(h, _ + 1)], 1)  # (115,2) fusion_attension[l+1]
            atten_co = F.softmax(atten_co, dim=1).unsqueeze(2)  # (115,2,1)
            hfx = torch.stack([hq, h], dim=1)  # hfx=(115,2,256)
            hfx = atten_co * hfx
            hfx = torch.sum(hfx, dim=1)  # hfx=(115,256)

            #   hf[_+1]= hfx[l+1] + (hf[l]-->GCN)
            hf = F.relu(hfx + self.last_fusion_layers[_ + 1](hf, edge_index))

            h_augq = F.relu(self.query_layers[_ + 1](h_augq, edge_index_aug))
            h_aug = F.relu(self.layers[_ + 1](h_aug, edge_index_aug))
            atten_coh = torch.cat([self.q_att_layer(h_augq, _ + 1), self.att_layer(h_aug, _ + 1)], 1)
            atten_coh = F.softmax(atten_coh, dim=1).unsqueeze(2)
            h_augfx = torch.stack([h_augq, h_aug], dim=1)
            h_augfx = atten_coh * h_augfx
            h_augfx = torch.sum(h_augfx, dim=1)
            h_augf = F.relu(h_augfx + self.last_fusion_layers[_ + 1](h_augf, edge_index_aug))

        hq = F.dropout(hq, training=self.training, p=self.dropout)
        h = F.dropout(h, training=self.training, p=self.dropout)
        hf = F.dropout(hf, training=self.training, p=self.dropout)
        h_augq = F.dropout(h_augq, training=self.training, p=self.dropout)
        h_aug = F.dropout(h_aug, training=self.training, p=self.dropout)
        h_augf = F.dropout(h_augf, training=self.training, p=self.dropout)

        hq = self.query_layers[self.num_layers - 1](hq, edge_index)
        h = self.layers[self.num_layers - 1](h, edge_index)
        atten_co = torch.cat([self.q_att_layer(hq, self.num_layers - 1), self.att_layer(h, self.num_layers - 1)], 1) #（115，2）
        atten_co = F.softmax(atten_co, dim=1).unsqueeze(2) #（115，2，1）
        hfx = torch.stack([hq, h], dim=1) #（115，2，256）
        hfx = atten_co * hfx
        hfx = torch.sum(hfx, dim=1) #（115，256）
        hf = hfx + self.last_fusion_layers[self.num_layers - 1](hf, edge_index) #（115，256）

        h_augq = self.query_layers[self.num_layers - 1](h_augq, edge_index_aug)
        h_aug = self.layers[self.num_layers - 1](h_aug, edge_index_aug)
        atten_coh = torch.cat(
            [self.q_att_layer(h_augq, self.num_layers - 1), self.att_layer(h_aug, self.num_layers - 1)],
            1)
        atten_coh = F.softmax(atten_coh, dim=1).unsqueeze(2)
        h_augfx = torch.stack([h_augq, h_aug], dim=1)
        h_augfx = atten_coh * h_augfx
        h_augfx = torch.sum(h_augfx, dim=1)
        h_augf = h_augfx + self.last_fusion_layers[self.num_layers - 1](h_augf, edge_index_aug)


        h_ = self.mlp1(hf)  # h_=(115,256)
        h_auge = self.hyperedge_representation(h_augf, edge_index_aug)  # h_auge=(115,256)
        # h_auge = self.lineraugh(h_auge)#'''
        h_auge = self.mlp1(h_auge)  # h_auge=(115,256)

        if loss is None:
            loss = self.contra(h_, h_auge, self.tau, (q, pos), self.alpha, self.lam, edge_index)
        else:
            loss = loss + self.contra(h_, h_auge, self.tau, (q, pos), self.alpha, self.lam, edge_index)

        return loss,h_

    def valiates(self,train,edge_weight=None):
        q,pos,edge_index,edge_index_aug,feats =train
        querys = torch.zeros(feats.shape[0], 1).to(self.device)
        querys[q] = 1.0


        hq = F.relu(self.query_layers[0](querys, edge_index, edge_weight))
        h = F.relu(self.layers[0](feats, edge_index, edge_weight))
        atten_co = torch.cat([self.q_att_layer(hq, 0), self.att_layer(h, 0)], 1)
        atten_co = F.softmax(atten_co, dim=1).unsqueeze(2)
        hf = torch.stack([hq, h], dim=1)
        hf = atten_co * hf
        hf = torch.sum(hf, dim=1)  # 对h

        querys = self.linerquerys(querys)
        feats = self.linerfeats(feats)
        atten_co_ = torch.cat([self.first_q_att_layer(querys), self.first_att_layer(feats)], 1)
        atten_co_ = F.softmax(atten_co_, dim=1).unsqueeze(2)
        hf_ = torch.stack([querys, feats], dim=1)
        hf_ = atten_co_ * hf_
        hf_ = torch.sum(hf_, dim=1)
        hf = F.relu(hf + self.last_fusion_layers[0](hf_, edge_index))

        for _ in range(self.num_layers - 2):
            hq = F.dropout(hq, training=self.training, p=self.dropout)
            h = F.dropout(h, training=self.training, p=self.dropout)
            hf = F.dropout(hf, training=self.training, p=self.dropout)

            hq = F.relu(self.query_layers[_ + 1](hq, edge_index, edge_weight))
            h = F.relu(self.layers[_ + 1](h, edge_index, edge_weight))

            atten_co = torch.cat([self.q_att_layer(hq, _ + 1), self.att_layer(h, _ + 1)], 1)
            atten_co = F.softmax(atten_co, dim=1).unsqueeze(2)
            hfx = torch.stack([hq, h], dim=1)
            hfx = atten_co * hfx
            hfx = torch.sum(hfx, dim=1)
            hf = F.relu(hfx + self.last_fusion_layers[_ + 1](hf, edge_index))

        hq = F.dropout(hq, training=self.training, p=self.dropout)
        h = F.dropout(h, training=self.training, p=self.dropout)
        hf = F.dropout(hf, training=self.training, p=self.dropout)

        hq = self.query_layers[self.num_layers - 1](hq, edge_index, edge_weight)
        h = self.layers[self.num_layers - 1](h, edge_index, edge_weight)
        atten_co = torch.cat(
            [self.q_att_layer(hq, self.num_layers - 1), self.att_layer(h, self.num_layers - 1)], 1)
        atten_co = F.softmax(atten_co, dim=1).unsqueeze(2)
        hfx = torch.stack([hq, h], dim=1)
        hfx = atten_co * hfx
        hfx = torch.sum(hfx, dim=1)
        hf = hfx + self.last_fusion_layers[self.num_layers - 1](hf, edge_index)

        h_ = self.mlp1(hf)

        return h_

    def forward(self, train):
        if self.training ==False:
            h_ = self.valiates(train)
            return h_
        loss,h_ = self.compute_loss(train)

        return loss,h_









