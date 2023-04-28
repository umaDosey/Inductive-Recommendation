import torch
import torch.nn as nn
import scipy.sparse as sp
import numpy as np
from utils import get_sparse_tensor, graph_rank_nodes, generate_daj_mat
from torch.nn.init import kaiming_uniform_, xavier_normal, normal_, zeros_, ones_
import sys
import torch.nn.functional as F
from sklearn.preprocessing import normalize
from torch.utils.checkpoint import checkpoint
import dgl
import multiprocessing as mp
import time
from info_nce import InfoNCE
from sklearn.metrics.pairwise import cosine_similarity
from torchmetrics.functional import pairwise_cosine_similarity
from utils import generate_aug_daj_mat, generate_drop_daj_mat, graph_aug_rank_nodes, graph_drop_rank_nodes, generate_drop_daj_mat2, generate_drop_daj_mat3
import random
import gc
def get_model(config, dataset):
    config = config.copy()
    config['dataset'] = dataset
    model = getattr(sys.modules['model'], config['name'])
    model = model(config)
    return model


def init_one_layer(in_features, out_features):
    layer = nn.Linear(in_features, out_features)
    kaiming_uniform_(layer.weight)
    zeros_(layer.bias)
    return layer


class BasicModel(nn.Module):
    def __init__(self, model_config):
        super(BasicModel, self).__init__()
        print(model_config)
        self.config = model_config
        self.name = model_config['name']
        self.device = model_config['device']
        self.n_users = model_config['dataset'].n_users
        self.n_items = model_config['dataset'].n_items
        self.trainable = True

    def predict(self, users):
        raise NotImplementedError

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path, map_location=self.device))


class MF(BasicModel):
    def __init__(self, model_config):
        super(MF, self).__init__(model_config)
        self.embedding_size = model_config['embedding_size']
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_size)
        self.item_embedding = nn.Embedding(self.n_items, self.embedding_size)
        normal_(self.user_embedding.weight, std=0.1)
        normal_(self.item_embedding.weight, std=0.1)
        self.to(device=self.device)

    def bpr_forward(self, users, pos_items, neg_items):
        users_e = self.user_embedding(users)
        pos_items_e, neg_items_e = self.item_embedding(pos_items), self.item_embedding(neg_items)
        l2_norm_sq = torch.norm(users_e, p=2, dim=1) ** 2 + torch.norm(pos_items_e, p=2, dim=1) ** 2 \
                     + torch.norm(neg_items_e, p=2, dim=1) ** 2
        return users_e, pos_items_e, neg_items_e, l2_norm_sq

    def predict(self, users):
        user_e = self.user_embedding(users)
        scores = torch.mm(user_e, self.item_embedding.weight.t())
        return scores


class LightGCN(BasicModel):
    def __init__(self, model_config):
        super(LightGCN, self).__init__(model_config)
        self.embedding_size = model_config['embedding_size']
        self.n_layers = model_config['n_layers']
        self.embedding = nn.Embedding(self.n_users + self.n_items, self.embedding_size)
        self.norm_adj = self.generate_graph(model_config['dataset'])
        normal_(self.embedding.weight, std=0.1) # weightの初期化
        self.to(device=self.device)

    def generate_graph(self, dataset):
        adj_mat = generate_daj_mat(dataset)
        degree = np.array(np.sum(adj_mat, axis=1)).squeeze()
        degree = np.maximum(1., degree)
        d_inv = np.power(degree, -0.5) # 累乗
        d_mat = sp.diags(d_inv, format='csr', dtype=np.float32)

        norm_adj = d_mat.dot(adj_mat).dot(d_mat)
        norm_adj = get_sparse_tensor(norm_adj, self.device)
        return norm_adj

    def get_rep(self):
        representations = self.embedding.weight
        all_layer_rep = [representations]
        row, column = self.norm_adj.indices()
        g = dgl.graph((column, row), num_nodes=self.norm_adj.shape[0], device=self.device)
        for _ in range(self.n_layers):
            representations = dgl.ops.gspmm(g, 'mul', 'sum', lhs_data=representations, rhs_data=self.norm_adj.values())
            all_layer_rep.append(representations)
        all_layer_rep = torch.stack(all_layer_rep, dim=0)
        final_rep = all_layer_rep.mean(dim=0)
        return final_rep

    def bpr_forward(self, users, pos_items, neg_items):
        rep = self.get_rep()
        users_e = self.embedding(users)
        pos_items_e, neg_items_e = self.embedding(self.n_users + pos_items), self.embedding(self.n_users + neg_items)
        l2_norm_sq = torch.norm(users_e, p=2, dim=1) ** 2 + torch.norm(pos_items_e, p=2, dim=1) ** 2 \
                     + torch.norm(neg_items_e, p=2, dim=1) ** 2
        users_r = rep[users, :]
        pos_items_r, neg_items_r = rep[self.n_users + pos_items, :], rep[self.n_users + neg_items, :]
        return users_r, pos_items_r, neg_items_r, l2_norm_sq

    def predict(self, users):
        rep = self.get_rep()
        users_r = rep[users, :]
        all_items_r = rep[self.n_users:, :]
        scores = torch.mm(users_r, all_items_r.t())
        return scores


class SGL(BasicModel):
    def __init__(self, model_config):
        super(SGL, self).__init__(model_config)
        self.embedding_size = model_config['embedding_size']
        self.n_layers = model_config['n_layers']
        # self.dropout = model_config['dropout']
        # self.feature_ratio = model_config['feature_ratio']  # Template%
        self.norm_adj = self.generate_graph(model_config['dataset'])

        self.alpha = 1.
        self.delta = model_config.get('delta', 0.99)
        self.taugh = model_config.get('taugh', 0.2)
        self.aug_rate = model_config.get('aug_rate', 0.8)
        # self.aug_num = model_config['aug_num']
        # self.aug_num = int(self.aug_num)
        # self.temper = model_config['temper']
        # self.update_feat_mat()
        # self.norm_aug_adj = generate_aug_graph(model_config['dataset'] )
        self.embedding = nn.Embedding(self.n_users + self.n_items, self.embedding_size)

        # self.w = nn.Parameter(torch.ones([self.embedding_size], dtype=torch.float32, device=self.device))
        normal_(self.embedding.weight, std=0.1)
        self.to(device=self.device)
        self.norm_aug_adj1 = self.generate_drop_graph(model_config['dataset'])
        self.norm_aug_adj2 = self.generate_drop_graph(model_config['dataset'])
        self.times = model_config.get('times', 0)

    def generate_graph(self, dataset):
        adj_mat = generate_daj_mat(dataset)
        degree = np.array(np.sum(adj_mat, axis=1)).squeeze()
        degree = np.maximum(1., degree)
        d_inv = np.power(degree, -0.5) # 累乗
        d_mat = sp.diags(d_inv, format='csr', dtype=np.float32)

        norm_adj = d_mat.dot(adj_mat).dot(d_mat)
        norm_adj = get_sparse_tensor(norm_adj, self.device)
        return norm_adj
    def generate_drop_graph(self, dataset):
        # new graph after adding some interactions
        aug_adj_mat = generate_drop_daj_mat(dataset, aug_rate=self.aug_rate)
        degree = np.array(np.sum(aug_adj_mat, axis=1)).squeeze()
        degree = np.maximum(1., degree)
        d_inv = np.power(degree, -0.5)  # 累乗
        d_mat = sp.diags(d_inv, format='csr', dtype=np.float32)

        norm_aug_adj = d_mat.dot(aug_adj_mat).dot(d_mat)
        norm_aug_adj = get_sparse_tensor(norm_aug_adj, self.device)
        return norm_aug_adj

    def get_rep(self):
        representations = self.embedding.weight
        all_layer_rep = [representations]
        row, column = self.norm_adj.indices()
        g = dgl.graph((column, row), num_nodes=self.norm_adj.shape[0], device=self.device)
        for _ in range(self.n_layers):
            representations = dgl.ops.gspmm(g, 'mul', 'sum', lhs_data=representations, rhs_data=self.norm_adj.values())
            all_layer_rep.append(representations)
        all_layer_rep = torch.stack(all_layer_rep, dim=0)
        final_rep = all_layer_rep.mean(dim=0)
        return final_rep
    def get_aug_rep(self, norm_aug_adj):
        # generate final embedding on aug-graph
        representations = self.embedding.weight
        all_layer_rep = [representations]
        row, column = norm_aug_adj.indices()
        g = dgl.graph((column, row), num_nodes=norm_aug_adj.shape[0], device=self.device)
        for _ in range(self.n_layers):
            representations = dgl.ops.gspmm(g, 'mul', 'sum', lhs_data=representations, rhs_data=norm_aug_adj.values())
            all_layer_rep.append(representations)
        all_layer_rep = torch.stack(all_layer_rep, dim=0)
        final_rep = all_layer_rep.mean(dim=0)
        return final_rep

    def cal_loss(self, users_r, aug_users_r):
        # calcrate ssl-loss(InfoNCE)
        loss = InfoNCE(negative_mode='unpaired')

        query = users_r
        positive_key = aug_users_r
        negative_keys = aug_users_r
        contrasive_loss = loss(query, positive_key, negative_keys)
        return contrasive_loss

    def bpr_forward(self, users, pos_items, neg_items):
        # 普通の埋め込み
        rep = self.get_rep()
        users_r = rep[users, :]
        pos_items_r, neg_items_r = rep[self.n_users + pos_items, :], rep[self.n_users + neg_items, :]
        # AUG後の埋め込み

        aug_rep1 = self.get_aug_rep(self.norm_aug_adj1)
        aug_users_r1 = aug_rep1[users, :]
        aug_rep2 = self.get_aug_rep(self.norm_aug_adj2)
        aug_users_r2 = aug_rep2[users, :]
        l2_norm_sq = torch.norm(users_r, p=2, dim=1) ** 2 + torch.norm(pos_items_r, p=2, dim=1) ** 2 \
                     + torch.norm(neg_items_r, p=2, dim=1) ** 2
        # Contrasive loss
        contrasive_loss = self.cal_loss(aug_users_r1, aug_users_r2)
        return users_r, pos_items_r, neg_items_r, l2_norm_sq, contrasive_loss



    def update_aug_adj(self):
        del self.norm_aug_adj1, self.norm_aug_adj2
        torch.cuda.empty_cache()
        # gc.collect()
        self.norm_aug_adj1 = self.generate_drop_graph(self.config['dataset'])
        self.norm_aug_adj2 = self.generate_drop_graph(self.config['dataset'])

    def predict(self, users):
        rep = self.get_rep()
        users_r = rep[users, :]
        all_items_r = rep[self.n_users:, :]
        scores = torch.mm(users_r, all_items_r.t())
        return scores

class HALF(BasicModel):
    def __init__(self, model_config):
        super(HALF, self).__init__(model_config)
        self.embedding_size = model_config['embedding_size']
        self.n_layers = model_config['n_layers']
        # self.dropout = model_config['dropout']
        # self.feature_ratio = model_config['feature_ratio']  # Template%
        self.norm_adj = self.generate_graph(model_config['dataset'])

        self.alpha = 1.
        self.delta = model_config.get('delta', 0.99)
        self.taugh = model_config.get('taugh', 0.2)
        self.aug_rate = model_config.get('aug_rate', 0.8)
        # self.aug_num = model_config['aug_num']
        # self.aug_num = int(self.aug_num)
        # self.temper = model_config['temper']
        # self.update_feat_mat()
        # self.norm_aug_adj = generate_aug_graph(model_config['dataset'] )
        self.embedding = nn.Embedding(self.n_users + self.n_items, self.embedding_size)

        # self.w = nn.Parameter(torch.ones([self.embedding_size], dtype=torch.float32, device=self.device))
        normal_(self.embedding.weight, std=0.1)
        self.to(device=self.device)
        self.norm_aug_adj1 = self.generate_drop_graph(model_config['dataset'])
        # self.norm_aug_adj2 = self.generate_drop_graph(model_config['dataset'])
        self.times = model_config.get('times', 0)

    def generate_graph(self, dataset):
        adj_mat = generate_daj_mat(dataset)
        degree = np.array(np.sum(adj_mat, axis=1)).squeeze()
        degree = np.maximum(1., degree)
        d_inv = np.power(degree, -0.5) # 累乗
        d_mat = sp.diags(d_inv, format='csr', dtype=np.float32)

        norm_adj = d_mat.dot(adj_mat).dot(d_mat)
        norm_adj = get_sparse_tensor(norm_adj, self.device)
        return norm_adj
    def generate_drop_graph(self, dataset):
        # new graph after adding some interactions
        aug_adj_mat = generate_drop_daj_mat(dataset, aug_rate=self.aug_rate)
        degree = np.array(np.sum(aug_adj_mat, axis=1)).squeeze()
        degree = np.maximum(1., degree)
        d_inv = np.power(degree, -0.5)  # 累乗
        d_mat = sp.diags(d_inv, format='csr', dtype=np.float32)

        norm_aug_adj = d_mat.dot(aug_adj_mat).dot(d_mat)
        norm_aug_adj = get_sparse_tensor(norm_aug_adj, self.device)
        return norm_aug_adj

    def get_rep(self):
        representations = self.embedding.weight
        all_layer_rep = [representations]
        row, column = self.norm_adj.indices()
        g = dgl.graph((column, row), num_nodes=self.norm_adj.shape[0], device=self.device)
        for _ in range(self.n_layers):
            representations = dgl.ops.gspmm(g, 'mul', 'sum', lhs_data=representations, rhs_data=self.norm_adj.values())
            all_layer_rep.append(representations)
        all_layer_rep = torch.stack(all_layer_rep, dim=0)
        final_rep = all_layer_rep.mean(dim=0)
        return final_rep
    def get_aug_rep(self, norm_aug_adj):
        # generate final embedding on aug-graph

        representations = self.embedding.weight
        all_layer_rep = [representations]
        row, column = norm_aug_adj.indices()
        g = dgl.graph((column, row), num_nodes=norm_aug_adj.shape[0], device=self.device)
        for _ in range(self.n_layers):
            representations = dgl.ops.gspmm(g, 'mul', 'sum', lhs_data=representations, rhs_data=norm_aug_adj.values())
            all_layer_rep.append(representations)
        all_layer_rep = torch.stack(all_layer_rep, dim=0)
        final_rep = all_layer_rep.mean(dim=0)
        return final_rep

    def cal_loss(self, users_r, aug_users_r):
        # calcrate ssl-loss(InfoNCE)
        loss = InfoNCE(negative_mode='unpaired')
        # print(users_r.size())
        # print(aug_users_r.size())
        query = users_r
        positive_key = aug_users_r
        negative_keys = aug_users_r
        contrasive_loss = loss(query, positive_key, negative_keys)
        return contrasive_loss

    def bpr_forward(self, users, pos_items, neg_items):
        # 普通の埋め込み
        rep = self.get_rep()
        users_r = rep[users, :]
        pos_items_r, neg_items_r = rep[self.n_users + pos_items, :], rep[self.n_users + neg_items, :]
        # AUG後の埋め込み

        aug_rep1 = self.get_aug_rep(self.norm_aug_adj1)
        aug_users_r1 = aug_rep1[users, :]
        """
        aug_rep2 = self.get_aug_rep(self.norm_aug_adj2)
        aug_users_r2 = aug_rep2[users, :]
        """
        l2_norm_sq = torch.norm(users_r, p=2, dim=1) ** 2 + torch.norm(pos_items_r, p=2, dim=1) ** 2 \
                     + torch.norm(neg_items_r, p=2, dim=1) ** 2
        # Contrasive loss
        contrasive_loss = self.cal_loss(users_r, aug_users_r1)
        return users_r, pos_items_r, neg_items_r, l2_norm_sq, contrasive_loss



    def update_aug_adj(self):
        del self.norm_aug_adj1, # self.norm_aug_adj2
        torch.cuda.empty_cache()
        # gc.collect()
        self.norm_aug_adj1 = self.generate_drop_graph(self.config['dataset'])
        # self.norm_aug_adj2 = self.generate_drop_graph(self.config['dataset'])

    def predict(self, users):
        rep = self.get_rep()
        users_r = rep[users, :]
        all_items_r = rep[self.n_users:, :]
        scores = torch.mm(users_r, all_items_r.t())
        return scores

class DOSE_aug(BasicModel):
    def __init__(self, model_config):
        super(DOSE_aug, self).__init__(model_config)
        self.embedding_size = model_config['embedding_size']
        self.n_layers = model_config['n_layers']
        self.dropout = model_config['dropout']
        self.feature_ratio = model_config['feature_ratio']  # Template%
        self.norm_adj = self.generate_graph(model_config['dataset'])

        self.alpha = 1.
        self.delta = model_config.get('delta', 0.99)
        self.taugh = model_config.get('taugh', 0.2)
        self.aug_num = model_config['aug_num']
        # self.temper = model_config['temper']
        self.feat_mat, self.user_map, self.item_map, self.row_sum = \
            self.generate_feat(model_config['dataset'],
                               ranking_metric=model_config.get('ranking_metric', 'sort'))
        self.update_feat_mat()
        # self.norm_aug_adj = enerate_aug_graph(model_config['dataset'] )
        self.embedding = nn.Embedding(self.feat_mat.shape[1], self.embedding_size)

        self.w = nn.Parameter(torch.ones([self.embedding_size], dtype=torch.float32, device=self.device))
        normal_(self.embedding.weight, std=0.1)
        self.to(device=self.device)
        self.norm_aug_adj = self.generate_aug_graph(model_config['dataset'])
        self.times = model_config.get('times', 0.1)
        # self.aug_num = (self.times) * len(self.feat_mat)

    def update_feat_mat(self):
        row, _ = self.feat_mat.indices()
        edge_values = torch.pow(self.row_sum[row], (self.alpha - 1.) / 2. - 0.5)
        self.feat_mat = torch.sparse.FloatTensor(self.feat_mat.indices(), edge_values, self.feat_mat.shape).coalesce()


    def feat_mat_anneal(self):
        self.alpha *= self.delta
        self.update_feat_mat()

    def generate_graph(self, dataset):
        return LightGCN.generate_graph(self, dataset)

    def generate_aug_graph(self, dataset):
        # new graph after adding some interactions
        aug_idx = self.cal_cos_sim()
        aug_adj_mat = generate_aug_daj_mat(dataset, aug_idx)
        degree = np.array(np.sum(aug_adj_mat, axis=1)).squeeze()
        degree = np.maximum(1., degree)
        d_inv = np.power(degree, -0.5) # 累乗
        d_mat = sp.diags(d_inv, format='csr', dtype=np.float32)

        norm_aug_adj = d_mat.dot(aug_adj_mat).dot(d_mat)
        norm_aug_adj = get_sparse_tensor(norm_aug_adj, self.device)
        return norm_aug_adj

    def generate_feat(self, dataset, is_updating=False, ranking_metric=None):
        # return adj matrix with template
        if not is_updating:
            if self.feature_ratio < 1.:  # ランク付後にスライス

                ranked_users, ranked_items = graph_rank_nodes(dataset, ranking_metric)
                core_users = ranked_users[:int(self.n_users * self.feature_ratio)]
                core_items = ranked_items[:int(self.n_items * self.feature_ratio)]
            else:
                core_users = np.arange(self.n_users, dtype=np.int64)
                core_items = np.arange(self.n_items, dtype=np.int64)

            user_map = dict()
            for idx, user in enumerate(core_users):
                user_map[user] = idx
            item_map = dict()
            for idx, item in enumerate(core_items):
                item_map[item] = idx
        else:
            user_map = self.user_map
            item_map = self.item_map

        user_dim, item_dim = len(user_map), len(item_map)
        indices = []
        for user, item in dataset.train_array:
            if item in item_map:
                indices.append([user, user_dim + item_map[item]])
            if user in user_map:
                indices.append([self.n_users + item, user_map[user]])
        for user in range(self.n_users):
            indices.append([user, user_dim + item_dim])
        for item in range(self.n_items):
            indices.append([self.n_users + item, user_dim + item_dim + 1])
        feat = sp.coo_matrix((np.ones((len(indices),)), np.array(indices).T),
                             shape=(self.n_users + self.n_items, user_dim + item_dim + 2), dtype=np.float32).tocsr()
        row_sum = torch.tensor(np.array(np.sum(feat, axis=1)).squeeze(), dtype=torch.float32, device=self.device)
        feat = get_sparse_tensor(feat, self.device)
        return feat, user_map, item_map, row_sum

    def inductive_rep_layer(self, feat_mat):
        padding_tensor = torch.empty([max(self.feat_mat.shape) - self.feat_mat.shape[1], self.embedding_size],
                                     dtype=torch.float32, device=self.device)
        padding_features = torch.cat([self.embedding.weight, padding_tensor], dim=0)

        row, column = feat_mat.indices()
        g = dgl.graph((column, row), num_nodes=max(self.feat_mat.shape), device=self.device)
        x = dgl.ops.gspmm(g, 'mul', 'sum', lhs_data=padding_features, rhs_data=feat_mat.values())
        x = x[:self.feat_mat.shape[0], :]
        return x


    def get_def_rep(self):
        # generate final embedding
        feat_mat = NGCF.dropout_sp_mat(self, self.feat_mat)
        representations = self.inductive_rep_layer(feat_mat)

        all_layer_rep = [representations]
        row, column = self.norm_adj.indices()
        g = dgl.graph((column, row), num_nodes=self.norm_adj.shape[0], device=self.device)
        for _ in range(self.n_layers):
            representations = dgl.ops.gspmm(g, 'mul', 'sum', lhs_data=representations, rhs_data=self.norm_adj.values())
            all_layer_rep.append(representations)
        all_layer_rep = torch.stack(all_layer_rep, dim=0)
        final_rep = all_layer_rep.mean(dim=0)
        return final_rep

    def get_aug_rep(self, norm_aug_adj):
        # generate final embedding on aug-graph
        feat_mat = NGCF.dropout_sp_mat(self, self.feat_mat)
        representations = self.inductive_rep_layer(feat_mat)
        
        all_layer_rep = [representations]
        row, column = norm_aug_adj.indices()
        g = dgl.graph((column, row), num_nodes=norm_aug_adj.shape[0], device=self.device)
        for _ in range(self.n_layers):
            representations = dgl.ops.gspmm(g, 'mul', 'sum', lhs_data=representations, rhs_data=norm_aug_adj.values())
            all_layer_rep.append(representations)
        all_layer_rep = torch.stack(all_layer_rep, dim=0)
        final_rep = all_layer_rep.mean(dim=0)
        return final_rep

    def cal_cos_sim(self):
        # calculate cosine similarity with user embeddings and item embeddings (on CPU)
        rep = self.get_def_rep()
        all_users_r = rep[:self.n_users, :]
        all_items_r = rep[self.n_users:, :]
        
        all_user_r = all_users_r.to('cpu').detach().numpy().copy()
        all_item_r = all_items_r.to('cpu').detach().numpy().copy()
        
        x = cosine_similarity(all_user_r, all_item_r)

        del all_users_r, all_items_r
        del all_user_r, all_item_r
        torch.cuda.empty_cache()

        cos_mat = torch.from_numpy(x.astype(np.float32)).clone()
        cos_mat.to(device=self.device)
        cos_mat = torch.reshape(cos_mat, (1, -1))
        cos_mat = torch.squeeze(cos_mat)
        cos_mat1 = cos_mat[:len(cos_mat)//2]
        _, idx1 = torch.topk(cos_mat1, self.aug_num//2)
        cos_mat2 = cos_mat[len(cos_mat)//2:]

        del cos_mat
        torch.cuda.empty_cache()
        # gc.collect()
        print(len(cos_mat2))
        _, idx2 = torch.topk(cos_mat2, self.aug_num//2)
        #idx = idx.tolist()S
        aug_idx1 = [[int(torch.div(idx1[i], self.n_items, rounding_mode='floor')),
                     (int(torch.fmod(idx1[i], self.n_items)))] for i in range(self.aug_num//2)]
        # aug_idx.to(device=self.device)
        aug_idx2 = [
            [int(torch.div(idx2[i]+self.aug_num//2, self.n_items, rounding_mode='floor')), (int(torch.fmod(idx2[i]+self.aug_num//2, self.n_items)))] for i
            in range(self.aug_num//2)]
        aug_idx1.extend(aug_idx2)
        return aug_idx1  # return list [user_id, item_id]
    
    def cal_cos_sim_v2(self):
        # calculate cosine similarity with user embeddings and item embeddings (on GPU)
        rep = self.get_def_rep()
        all_users_r = rep[:self.n_users, :]
        all_items_r = rep[self.n_users:, :]
        
        cos_sim = pairwise_cosine_similarity(all_users_r, all_items_r)
        cos_sim = torch.reshape(cos_sim, (1, -1))
        _, idx = torch.topk(cos_sim, self.aug_num)
        aug_idx = [[int(torch.div(idx[0][i], self.n_items, rounding_mode='floor')),
                    (int(torch.fmod(idx[0][i], self.n_items)))] for i in range(self.aug_num)]

        return aug_idx  # return list [user_id, item_id]


    def cal_loss(self, users_r, aug_users_r):
        # calcrate ssl-loss(InfoNCE)
        loss = InfoNCE(negative_mode='unpaired')

        query = users_r
        positive_key = aug_users_r
        negative_keys = aug_users_r
        contrasive_loss = loss(query, positive_key, negative_keys)
        return contrasive_loss

    def bpr_forward(self, users, pos_items, neg_items):
        # 普通の埋め込み
        rep = self.get_def_rep()
        users_r = rep[users, :]
        pos_items_r, neg_items_r = rep[self.n_users + pos_items, :], rep[self.n_users + neg_items, :]
        # AUG後の埋め込み
        
        aug_rep = self.get_aug_rep(self.norm_aug_adj)
        aug_users_r = aug_rep[users, :]
        l2_norm_sq = torch.norm(users_r, p=2, dim=1) ** 2 + torch.norm(pos_items_r, p=2, dim=1) ** 2 \
                     + torch.norm(neg_items_r, p=2, dim=1) ** 2
        # Contrasive loss
        contrasive_loss = self.cal_loss(users_r, aug_users_r)
        return users_r, pos_items_r, neg_items_r, l2_norm_sq, contrasive_loss



    def predict(self, users):
        rep = self.get_def_rep()
        users_r = rep[users, :]
        all_items_r = rep[self.n_users:, :]
        scores = torch.mm(users_r, all_items_r.t())
        return scores
    
    def save(self, path):
        params = {'sate_dict': self.state_dict(), 'user_map': self.user_map,
                  'item_map': self.item_map, 'alpha': self.alpha}
        torch.save(params, path)

    def load(self, path):
        params = torch.load(path, map_location=self.device)
        self.load_state_dict(params['sate_dict'])
        self.user_map = params['user_map']
        self.item_map = params['item_map']
        self.alpha = params['alpha']
        self.feat_mat, _, _, self.row_sum = self.generate_feat(self.config['dataset'], is_updating=True)
        self.update_feat_mat()


class DOSE_aug4(BasicModel):
    def __init__(self, model_config):
        super(DOSE_aug4, self).__init__(model_config)
        self.embedding_size = model_config['embedding_size']
        self.n_layers = model_config['n_layers']
        self.dropout = model_config['dropout']
        self.feature_ratio = model_config['feature_ratio']  # Template%
        self.norm_adj = self.generate_graph(model_config['dataset'])

        self.alpha = 1.
        self.delta = model_config.get('delta', 0.99)
        self.taugh = model_config.get('taugh', 0.2)
        self.pai = model_config.get('pai', 0.1)
        self.aug_num = model_config['aug_num']
        # self.temper = model_config['temper']
        self.feat_mat, self.user_map, self.item_map, self.row_sum = \
            self.generate_feat(model_config['dataset'],
                               ranking_metric=model_config.get('ranking_metric', 'sort'))
        self.update_feat_mat()
        # self.norm_aug_adj = enerate_aug_graph(model_config['dataset'] )
        self.embedding = nn.Embedding(self.feat_mat.shape[1], self.embedding_size)

        self.w = nn.Parameter(torch.ones([self.embedding_size], dtype=torch.float32, device=self.device))
        normal_(self.embedding.weight, std=0.1)
        self.to(device=self.device)
        self.norm_aug_adj = self.generate_aug_graph(model_config['dataset'])
        self.times = model_config.get('times', 0.1)
        # self.aug_num = (self.times) * len(self.feat_mat)

    def update_feat_mat(self):
        row, _ = self.feat_mat.indices()
        edge_values = torch.pow(self.row_sum[row], (self.alpha - 1.) / 2. - 0.5)
        self.feat_mat = torch.sparse.FloatTensor(self.feat_mat.indices(), edge_values, self.feat_mat.shape).coalesce()

    def feat_mat_anneal(self):
        self.alpha *= self.delta
        self.update_feat_mat()

    def generate_graph(self, dataset):
        return LightGCN.generate_graph(self, dataset)

    def generate_aug_graph(self, dataset):
        # new graph after adding some interactions
        aug_idx = self.cal_cos_sim()
        aug_adj_mat = generate_aug_daj_mat(dataset, aug_idx)
        degree = np.array(np.sum(aug_adj_mat, axis=1)).squeeze()
        degree = np.maximum(1., degree)
        d_inv = np.power(degree, -0.5)  # 累乗
        d_mat = sp.diags(d_inv, format='csr', dtype=np.float32)

        norm_aug_adj = d_mat.dot(aug_adj_mat).dot(d_mat)
        norm_aug_adj = get_sparse_tensor(norm_aug_adj, self.device)
        return norm_aug_adj

    def generate_feat(self, dataset, is_updating=False, ranking_metric=None):
        # return adj matrix with template
        if not is_updating:
            if self.feature_ratio < 1.:  # ランク付後にスライス

                ranked_users, ranked_items = graph_rank_nodes(dataset, ranking_metric)
                core_users = ranked_users[:int(self.n_users * self.feature_ratio)]
                core_items = ranked_items[:int(self.n_items * self.feature_ratio)]
            else:
                core_users = np.arange(self.n_users, dtype=np.int64)
                core_items = np.arange(self.n_items, dtype=np.int64)

            user_map = dict()
            for idx, user in enumerate(core_users):
                user_map[user] = idx
            item_map = dict()
            for idx, item in enumerate(core_items):
                item_map[item] = idx
        else:
            user_map = self.user_map
            item_map = self.item_map

        user_dim, item_dim = len(user_map), len(item_map)
        indices = []
        for user, item in dataset.train_array:
            if item in item_map:
                indices.append([user, user_dim + item_map[item]])
            if user in user_map:
                indices.append([self.n_users + item, user_map[user]])
        for user in range(self.n_users):
            indices.append([user, user_dim + item_dim])
        for item in range(self.n_items):
            indices.append([self.n_users + item, user_dim + item_dim + 1])
        feat = sp.coo_matrix((np.ones((len(indices),)), np.array(indices).T),
                             shape=(self.n_users + self.n_items, user_dim + item_dim + 2), dtype=np.float32).tocsr()
        row_sum = torch.tensor(np.array(np.sum(feat, axis=1)).squeeze(), dtype=torch.float32, device=self.device)
        feat = get_sparse_tensor(feat, self.device)
        return feat, user_map, item_map, row_sum

    def inductive_rep_layer(self, feat_mat):
        padding_tensor = torch.empty([max(self.feat_mat.shape) - self.feat_mat.shape[1], self.embedding_size],
                                     dtype=torch.float32, device=self.device)
        padding_features = torch.cat([self.embedding.weight, padding_tensor], dim=0)

        row, column = feat_mat.indices()
        g = dgl.graph((column, row), num_nodes=max(self.feat_mat.shape), device=self.device)
        x = dgl.ops.gspmm(g, 'mul', 'sum', lhs_data=padding_features, rhs_data=feat_mat.values())
        x = x[:self.feat_mat.shape[0], :]
        return x

    def get_def_rep(self):
        # generate final embedding
        feat_mat = NGCF.dropout_sp_mat(self, self.feat_mat)
        representations = self.inductive_rep_layer(feat_mat)

        all_layer_rep = [representations]
        row, column = self.norm_adj.indices()
        g = dgl.graph((column, row), num_nodes=self.norm_adj.shape[0], device=self.device)
        for _ in range(self.n_layers):
            representations = dgl.ops.gspmm(g, 'mul', 'sum', lhs_data=representations, rhs_data=self.norm_adj.values())
            all_layer_rep.append(representations)
        all_layer_rep = torch.stack(all_layer_rep, dim=0)
        final_rep = all_layer_rep.mean(dim=0)
        return final_rep

    def get_aug_rep(self, norm_aug_adj):
        # generate final embedding on aug-graph
        feat_mat = NGCF.dropout_sp_mat(self, self.feat_mat)
        representations = self.inductive_rep_layer(feat_mat)

        all_layer_rep = [representations]
        row, column = norm_aug_adj.indices()
        g = dgl.graph((column, row), num_nodes=norm_aug_adj.shape[0], device=self.device)
        for _ in range(self.n_layers):
            representations = dgl.ops.gspmm(g, 'mul', 'sum', lhs_data=representations, rhs_data=norm_aug_adj.values())
            all_layer_rep.append(representations)
        all_layer_rep = torch.stack(all_layer_rep, dim=0)
        final_rep = all_layer_rep.mean(dim=0)
        return final_rep

    def cal_cos_sim(self):
        # calculate cosine similarity with user embeddings and item embeddings (on CPU)
        rep = self.get_def_rep()
        all_users_r = rep[:self.n_users, :]
        all_items_r = rep[self.n_users:, :]

        cos_mat = pairwise_cosine_similarity(all_users_r, all_items_r)

        del all_users_r, all_items_r
        torch.cuda.empty_cache()

        u_id, i_id = torch.where(cos_mat >= self.pai)

        del cos_mat
        torch.cuda.empty_cache()

        idx = torch.stack([u_id, i_id], dim=1)
        idx = idx.tolist()

        return idx  # return list [user_id, item_id]

    def cal_cos_sim_v2(self):
        # calculate cosine similarity with user embeddings and item embeddings (on GPU)
        rep = self.get_def_rep()
        all_users_r = rep[:self.n_users, :]
        all_items_r = rep[self.n_users:, :]

        cos_sim = pairwise_cosine_similarity(all_users_r, all_items_r)
        cos_sim = torch.reshape(cos_sim, (1, -1))
        _, idx = torch.topk(cos_sim, self.aug_num)
        aug_idx = [[int(torch.div(idx[0][i], self.n_items, rounding_mode='floor')),
                    (int(torch.fmod(idx[0][i], self.n_items)))] for i in range(self.aug_num)]

        return aug_idx  # return list [user_id, item_id]

    def cal_loss(self, users_r, aug_users_r):
        # calcrate ssl-loss(InfoNCE)
        loss = InfoNCE(negative_mode='unpaired')

        query = users_r
        positive_key = aug_users_r
        negative_keys = aug_users_r
        contrasive_loss = loss(query, positive_key, negative_keys)
        return contrasive_loss

    def bpr_forward(self, users, pos_items, neg_items):
        # 普通の埋め込み
        rep = self.get_def_rep()
        users_r = rep[users, :]
        pos_items_r, neg_items_r = rep[self.n_users + pos_items, :], rep[self.n_users + neg_items, :]
        # AUG後の埋め込み

        aug_rep = self.get_aug_rep(self.norm_aug_adj)
        aug_users_r = aug_rep[users, :]
        l2_norm_sq = torch.norm(users_r, p=2, dim=1) ** 2 + torch.norm(pos_items_r, p=2, dim=1) ** 2 \
                     + torch.norm(neg_items_r, p=2, dim=1) ** 2
        # Contrasive loss
        contrasive_loss = self.cal_loss(users_r, aug_users_r)
        return users_r, pos_items_r, neg_items_r, l2_norm_sq, contrasive_loss

    def predict(self, users):
        rep = self.get_def_rep()
        users_r = rep[users, :]
        all_items_r = rep[self.n_users:, :]
        scores = torch.mm(users_r, all_items_r.t())
        return scores

    def save(self, path):
        params = {'sate_dict': self.state_dict(), 'user_map': self.user_map,
                  'item_map': self.item_map, 'alpha': self.alpha}
        torch.save(params, path)

    def load(self, path):
        params = torch.load(path, map_location=self.device)
        self.load_state_dict(params['sate_dict'])
        self.user_map = params['user_map']
        self.item_map = params['item_map']
        self.alpha = params['alpha']
        self.feat_mat, _, _, self.row_sum = self.generate_feat(self.config['dataset'], is_updating=True)
        self.update_feat_mat()

class DOSE_aug2(BasicModel):
    def __init__(self, model_config):
        super(DOSE_aug2, self).__init__(model_config)
        self.embedding_size = model_config['embedding_size']
        self.n_layers = model_config['n_layers']
        self.dropout = model_config['dropout']
        self.feature_ratio = model_config['feature_ratio']  # Template%
        self.norm_adj = self.generate_graph(model_config['dataset'])

        self.alpha = 1.
        self.delta = model_config.get('delta', 0.99)
        self.taugh = model_config.get('taugh', 0.2)
        self.aug_num = model_config['aug_num']
        self.aug_num = int(self.aug_num)
        # self.temper = model_config['temper']
        self.feat_mat, self.user_map, self.item_map, self.row_sum = \
            self.generate_feat(model_config['dataset'],
                               ranking_metric=model_config.get('ranking_metric', 'sort'))
        self.update_feat_mat()
        # self.norm_aug_adj = generate_aug_graph(model_config['dataset'] )
        self.embedding = nn.Embedding(self.feat_mat.shape[1], self.embedding_size)

        self.w = nn.Parameter(torch.ones([self.embedding_size], dtype=torch.float32, device=self.device))
        normal_(self.embedding.weight, std=0.1)
        self.to(device=self.device)
        self.norm_aug_adj = self.generate_aug_graph(model_config['dataset'])
        self.times = model_config.get('times', 0.1)
        self.aug_feat_mat, self.aug_user_map, self.aug_item_map, self.aug_row_sum = \
            self.generate_aug_feat(model_config['dataset'],
                               ranking_metric=model_config.get('ranking_metric', 'sort'))
        # self.aug_num = (self.times) * len(self.feat_mat)

    def update_feat_mat(self):
        row, _ = self.feat_mat.indices()
        edge_values = torch.pow(self.row_sum[row], (self.alpha - 1.) / 2. - 0.5)
        self.feat_mat = torch.sparse.FloatTensor(self.feat_mat.indices(), edge_values, self.feat_mat.shape).coalesce()

    def update_aug_feat_mat(self):
        row, _ = self.aug_feat_mat.indices()
        edge_values = torch.pow(self.row_sum[row], (self.alpha - 1.) / 2. - 0.5)
        self.aug_feat_mat = torch.sparse.FloatTensor(self.aug_feat_mat.indices(), edge_values,
                                                     self.aug_feat_mat.shape).coalesce()

    def feat_mat_anneal(self):
        self.alpha *= self.delta
        self.update_feat_mat()

    def generate_graph(self, dataset):
        return LightGCN.generate_graph(self, dataset)

    def generate_aug_graph(self, dataset):
        # new graph after adding some interactions
        aug_idx = self.cal_cos_sim()
        aug_adj_mat = generate_aug_daj_mat(dataset, aug_idx)
        degree = np.array(np.sum(aug_adj_mat, axis=1)).squeeze()
        degree = np.maximum(1., degree)
        d_inv = np.power(degree, -0.5) # 累乗
        d_mat = sp.diags(d_inv, format='csr', dtype=np.float32)

        norm_aug_adj = d_mat.dot(aug_adj_mat).dot(d_mat)
        norm_aug_adj = get_sparse_tensor(norm_aug_adj, self.device)
        return norm_aug_adj

    def generate_feat(self, dataset, is_updating=False, ranking_metric=None):
        # return adj matrix with template
        if not is_updating:
            if self.feature_ratio < 1.:  # ランク付後にスライス

                ranked_users, ranked_items = graph_rank_nodes(dataset, ranking_metric)
                core_users = ranked_users[:int(self.n_users * self.feature_ratio)]
                core_items = ranked_items[:int(self.n_items * self.feature_ratio)]
            else:
                core_users = np.arange(self.n_users, dtype=np.int64)
                core_items = np.arange(self.n_items, dtype=np.int64)

            user_map = dict()
            for idx, user in enumerate(core_users):
                user_map[user] = idx
            item_map = dict()
            for idx, item in enumerate(core_items):
                item_map[item] = idx
        else:
            user_map = self.user_map
            item_map = self.item_map

        user_dim, item_dim = len(user_map), len(item_map)
        indices = []

        for user, item in dataset.train_array:
            if item in item_map:
                indices.append([user, user_dim + item_map[item]])
            if user in user_map:
                indices.append([self.n_users + item, user_map[user]])
        for user in range(self.n_users):
            indices.append([user, user_dim + item_dim])
        for item in range(self.n_items):
            indices.append([self.n_users + item, user_dim + item_dim + 1])
        feat = sp.coo_matrix((np.ones((len(indices),)), np.array(indices).T),
                             shape=(self.n_users + self.n_items, user_dim + item_dim + 2), dtype=np.float32).tocsr()
        row_sum = torch.tensor(np.array(np.sum(feat, axis=1)).squeeze(), dtype=torch.float32, device=self.device)
        feat = get_sparse_tensor(feat, self.device)
        return feat, user_map, item_map, row_sum

    def generate_aug_feat(self, dataset, is_updating=False, ranking_metric=None):
        # return adj matrix with template
        aug_idx = self.cal_cos_sim()
        if not is_updating:
            if self.feature_ratio < 1.:  # ランク付後にスライス

                ranked_users, ranked_items = graph_aug_rank_nodes(dataset, ranking_metric, aug_idx)
                core_users = ranked_users[:int(self.n_users * self.feature_ratio)]
                core_items = ranked_items[:int(self.n_items * self.feature_ratio)]
            else:
                core_users = np.arange(self.n_users, dtype=np.int64)
                core_items = np.arange(self.n_items, dtype=np.int64)

            user_map = dict()
            for idx, user in enumerate(core_users):
                user_map[user] = idx
            item_map = dict()
            for idx, item in enumerate(core_items):
                item_map[item] = idx
        else:
            user_map = self.user_map
            item_map = self.item_map

        user_dim, item_dim = len(user_map), len(item_map)

        train_array = dataset.train_array
        train_array.extend(aug_idx)
        train_array = list(map(list, set(map(tuple, train_array))))

        indices = []
        for user, item in train_array:
            if item in item_map:
                indices.append([user, user_dim + item_map[item]])
            if user in user_map:
                indices.append([self.n_users + item, user_map[user]])
        for user in range(self.n_users):
            indices.append([user, user_dim + item_dim])
        for item in range(self.n_items):
            indices.append([self.n_users + item, user_dim + item_dim + 1])
        feat = sp.coo_matrix((np.ones((len(indices),)), np.array(indices).T),
                             shape=(self.n_users + self.n_items, user_dim + item_dim + 2), dtype=np.float32).tocsr()
        row_sum = torch.tensor(np.array(np.sum(feat, axis=1)).squeeze(), dtype=torch.float32, device=self.device)
        feat = get_sparse_tensor(feat, self.device)
        return feat, user_map, item_map, row_sum

    def inductive_rep_layer(self, feat_mat):
        # generate embedding by using template
        padding_tensor = torch.empty([max(self.feat_mat.shape) - self.feat_mat.shape[1], self.embedding_size],
                                     dtype=torch.float32, device=self.device)
        padding_features = torch.cat([self.embedding.weight, padding_tensor], dim=0)

        row, column = feat_mat.indices()
        g = dgl.graph((column, row), num_nodes=max(self.feat_mat.shape), device=self.device)
        x = dgl.ops.gspmm(g, 'mul', 'sum', lhs_data=padding_features, rhs_data=feat_mat.values())
        x = x[:self.feat_mat.shape[0], :]
        return x

    def inductive_aug_rep_layer(self, aug_feat_mat):
        # generate embedding by using template
        padding_tensor = torch.empty([max(self.aug_feat_mat.shape) - self.aug_feat_mat.shape[1], self.embedding_size],
                                     dtype=torch.float32, device=self.device)
        padding_features = torch.cat([self.embedding.weight, padding_tensor], dim=0)

        row, column = aug_feat_mat.indices()
        g = dgl.graph((column, row), num_nodes=max(self.aug_feat_mat.shape), device=self.device)
        x = dgl.ops.gspmm(g, 'mul', 'sum', lhs_data=padding_features, rhs_data=aug_feat_mat.values())
        x = x[:self.aug_feat_mat.shape[0], :]
        return x

    def get_def_rep(self):
        # generate final embedding
        feat_mat = NGCF.dropout_sp_mat(self, self.feat_mat)
        representations = self.inductive_rep_layer(feat_mat)

        all_layer_rep = [representations]
        row, column = self.norm_adj.indices()
        g = dgl.graph((column, row), num_nodes=self.norm_adj.shape[0], device=self.device)
        for _ in range(self.n_layers):
            representations = dgl.ops.gspmm(g, 'mul', 'sum', lhs_data=representations, rhs_data=self.norm_adj.values())
            all_layer_rep.append(representations)
        all_layer_rep = torch.stack(all_layer_rep, dim=0)
        final_rep = all_layer_rep.mean(dim=0)
        return final_rep

    def get_aug_rep(self, norm_aug_adj):
        # generate final embedding on aug-graph
        feat_mat = NGCF.dropout_sp_mat(self, self.feat_mat)
        representations = self.inductive_rep_layer(feat_mat)

        all_layer_rep = [representations]
        row, column = norm_aug_adj.indices()
        g = dgl.graph((column, row), num_nodes=norm_aug_adj.shape[0], device=self.device)
        for _ in range(self.n_layers):
            representations = dgl.ops.gspmm(g, 'mul', 'sum', lhs_data=representations, rhs_data=norm_aug_adj.values())
            all_layer_rep.append(representations)
        all_layer_rep = torch.stack(all_layer_rep, dim=0)
        final_rep = all_layer_rep.mean(dim=0)
        return final_rep

    def cal_cos_sim(self):
        # calculate cosine similarity with user embeddings and item embeddings (on CPU)
        rep = self.get_def_rep()
        all_users_r = rep[:self.n_users, :]
        all_items_r = rep[self.n_users:, :]

        all_users_r = all_users_r.to('cpu').detach().numpy().copy()
        all_items_r = all_items_r.to('cpu').detach().numpy().copy()

        x = cosine_similarity(all_users_r, all_items_r)
        cos_mat = torch.from_numpy(x.astype(np.float32)).clone()
        cos_mat.to(device=self.device)
        cos_mat = torch.reshape(cos_mat, (1, -1))
        _, idx = torch.topk(cos_mat, self.aug_num)
        #idx = idx.tolist()S
        aug_idx = [[int(torch.div(idx[0][i], self.n_items, rounding_mode='floor')), (int(torch.fmod(idx[0][i], self.n_items)))] for i in range(self.aug_num)]

        return aug_idx  # return list [user_id, item_id]

    def cal_cos_sim_v2(self):
        # calculate cosine similarity with user embeddings and item embeddings (on GPU)
        rep = self.get_def_rep()
        all_users_r = rep[:self.n_users, :]
        all_items_r = rep[self.n_users:, :]

        cos_sim = pairwise_cosine_similarity(all_users_r, all_items_r)
        cos_sim = torch.reshape(cos_sim, (1, -1))
        _, idx = torch.topk(cos_sim, self.aug_num)
        aug_idx = [[int(torch.div(idx[0][i], self.n_items, rounding_mode='floor')), (int(torch.fmod(idx[0][i], self.n_items)))] for i in range(self.aug_num)]

        return aug_idx  # return list [user_id, item_id]

    def cal_loss(self, users_r, aug_users_r):
        # calcrate ssl-loss(InfoNCE)
        loss = InfoNCE(negative_mode='unpaired')

        query = users_r
        positive_key = aug_users_r
        negative_keys = aug_users_r
        contrasive_loss = loss(query, positive_key, negative_keys)
        return contrasive_loss

    def bpr_forward(self, users, pos_items, neg_items):
        # 普通の埋め込み
        rep = self.get_def_rep()
        users_r = rep[users, :]
        pos_items_r, neg_items_r = rep[self.n_users + pos_items, :], rep[self.n_users + neg_items, :]
        # AUG後の埋め込み

        aug_rep = self.get_aug_rep(self.norm_aug_adj)
        aug_users_r = aug_rep[users, :]
        l2_norm_sq = torch.norm(users_r, p=2, dim=1) ** 2 + torch.norm(pos_items_r, p=2, dim=1) ** 2 \
                     + torch.norm(neg_items_r, p=2, dim=1) ** 2
        # Contrasive loss
        contrasive_loss = self.cal_loss(users_r, aug_users_r)
        return users_r, pos_items_r, neg_items_r, l2_norm_sq, contrasive_loss



    def predict(self, users):
        rep = self.get_def_rep()
        users_r = rep[users, :]
        all_items_r = rep[self.n_users:, :]
        scores = torch.mm(users_r, all_items_r.t())
        return scores

    def save(self, path):
        params = {'sate_dict': self.state_dict(), 'user_map': self.user_map,
                  'item_map': self.item_map, 'alpha': self.alpha}
        torch.save(params, path)

    def load(self, path):
        params = torch.load(path, map_location=self.device)
        self.load_state_dict(params['sate_dict'])
        self.user_map = params['user_map']
        self.item_map = params['item_map']
        self.alpha = params['alpha']
        self.feat_mat, _, _, self.row_sum = self.generate_feat(self.config['dataset'], is_updating=True)
        self.update_feat_mat()


class DOSE_aug3(BasicModel):
    def __init__(self, model_config):
        super(DOSE_aug3, self).__init__(model_config)
        self.embedding_size = model_config['embedding_size']
        self.n_layers = model_config['n_layers']
        self.dropout = model_config['dropout']
        self.feature_ratio = model_config['feature_ratio']  # Template%
        self.norm_adj = self.generate_graph(model_config['dataset'])

        self.alpha = 1.
        self.delta = model_config.get('delta', 0.99)
        self.taugh = model_config.get('taugh', 0.2)
        self.aug_num = model_config['aug_num']
        self.aug_num = int(self.aug_num)
        # self.temper = model_config['temper']
        self.feat_mat, self.user_map, self.item_map, self.row_sum = \
            self.generate_feat(model_config['dataset'],
                               ranking_metric=model_config.get('ranking_metric', 'sort'))
        self.update_feat_mat()
        # self.norm_aug_adj = enerate_aug_graph(model_config['dataset'] )
        self.embedding = nn.Embedding(self.feat_mat.shape[1], self.embedding_size)

        self.w = nn.Parameter(torch.ones([self.embedding_size], dtype=torch.float32, device=self.device))
        normal_(self.embedding.weight, std=0.1)
        self.to(device=self.device)
        self.norm_aug_adj = self.generate_aug_graph(model_config['dataset'])
        self.times = model_config.get('times', 0.1)
        # self.aug_num = (self.times) * len(self.feat_mat)

    def update_feat_mat(self):
        row, _ = self.feat_mat.indices()
        edge_values = torch.pow(self.row_sum[row], (self.alpha - 1.) / 2. - 0.5)
        self.feat_mat = torch.sparse.FloatTensor(self.feat_mat.indices(), edge_values, self.feat_mat.shape).coalesce()

    def feat_mat_anneal(self):
        self.alpha *= self.delta
        self.update_feat_mat()

    def generate_graph(self, dataset):
        return LightGCN.generate_graph(self, dataset)

    def generate_aug_graph(self, dataset):
        # new graph after adding some interactions
        user_id = np.random.randint(0, self.n_users, self.aug_num)
        item_id = np.random.randint(0, self.n_items, self.aug_num)
        aug_idx = np.stack([user_id, item_id], 1)
        aug_idx = aug_idx.tolist()
        aug_adj_mat = generate_aug_daj_mat(dataset, aug_idx)
        degree = np.array(np.sum(aug_adj_mat, axis=1)).squeeze()
        degree = np.maximum(1., degree)
        d_inv = np.power(degree, -0.5)  # 累乗
        d_mat = sp.diags(d_inv, format='csr', dtype=np.float32)

        norm_aug_adj = d_mat.dot(aug_adj_mat).dot(d_mat)
        norm_aug_adj = get_sparse_tensor(norm_aug_adj, self.device)
        return norm_aug_adj

    def generate_feat(self, dataset, is_updating=False, ranking_metric=None):
        # return adj matrix with template
        if not is_updating:
            if self.feature_ratio < 1.:  # ランク付後にスライス

                ranked_users, ranked_items = graph_rank_nodes(dataset, ranking_metric)
                core_users = ranked_users[:int(self.n_users * self.feature_ratio)]
                core_items = ranked_items[:int(self.n_items * self.feature_ratio)]
            else:
                core_users = np.arange(self.n_users, dtype=np.int64)
                core_items = np.arange(self.n_items, dtype=np.int64)

            user_map = dict()
            for idx, user in enumerate(core_users):
                user_map[user] = idx
            item_map = dict()
            for idx, item in enumerate(core_items):
                item_map[item] = idx
        else:
            user_map = self.user_map
            item_map = self.item_map

        user_dim, item_dim = len(user_map), len(item_map)
        indices = []
        for user, item in dataset.train_array:
            if item in item_map:
                indices.append([user, user_dim + item_map[item]])
            if user in user_map:
                indices.append([self.n_users + item, user_map[user]])
        for user in range(self.n_users):
            indices.append([user, user_dim + item_dim])
        for item in range(self.n_items):
            indices.append([self.n_users + item, user_dim + item_dim + 1])
        feat = sp.coo_matrix((np.ones((len(indices),)), np.array(indices).T),
                             shape=(self.n_users + self.n_items, user_dim + item_dim + 2), dtype=np.float32).tocsr()
        row_sum = torch.tensor(np.array(np.sum(feat, axis=1)).squeeze(), dtype=torch.float32, device=self.device)
        feat = get_sparse_tensor(feat, self.device)
        return feat, user_map, item_map, row_sum

    def inductive_rep_layer(self, feat_mat):
        padding_tensor = torch.empty([max(self.feat_mat.shape) - self.feat_mat.shape[1], self.embedding_size],
                                     dtype=torch.float32, device=self.device)
        padding_features = torch.cat([self.embedding.weight, padding_tensor], dim=0)

        row, column = feat_mat.indices()
        g = dgl.graph((column, row), num_nodes=max(self.feat_mat.shape), device=self.device)
        x = dgl.ops.gspmm(g, 'mul', 'sum', lhs_data=padding_features, rhs_data=feat_mat.values())
        x = x[:self.feat_mat.shape[0], :]
        return x

    def get_def_rep(self):
        # generate final embedding
        feat_mat = NGCF.dropout_sp_mat(self, self.feat_mat)
        representations = self.inductive_rep_layer(feat_mat)

        all_layer_rep = [representations]
        row, column = self.norm_adj.indices()
        g = dgl.graph((column, row), num_nodes=self.norm_adj.shape[0], device=self.device)
        for _ in range(self.n_layers):
            representations = dgl.ops.gspmm(g, 'mul', 'sum', lhs_data=representations, rhs_data=self.norm_adj.values())
            all_layer_rep.append(representations)
        all_layer_rep = torch.stack(all_layer_rep, dim=0)
        final_rep = all_layer_rep.mean(dim=0)
        return final_rep

    def get_aug_rep(self, norm_aug_adj):
        # generate final embedding on aug-graph
        feat_mat = NGCF.dropout_sp_mat(self, self.feat_mat)
        representations = self.inductive_rep_layer(feat_mat)

        all_layer_rep = [representations]
        row, column = norm_aug_adj.indices()
        g = dgl.graph((column, row), num_nodes=norm_aug_adj.shape[0], device=self.device)
        for _ in range(self.n_layers):
            representations = dgl.ops.gspmm(g, 'mul', 'sum', lhs_data=representations, rhs_data=norm_aug_adj.values())
            all_layer_rep.append(representations)
        all_layer_rep = torch.stack(all_layer_rep, dim=0)
        final_rep = all_layer_rep.mean(dim=0)
        return final_rep

    def cal_cos_sim(self):
        # calculate cosine similarity with user embeddings and item embeddings (on CPU)
        rep = self.get_def_rep()
        all_users_r = rep[:self.n_users, :]
        all_items_r = rep[self.n_users:, :]

        all_users_r = all_users_r.to('cpu').detach().numpy().copy()
        all_items_r = all_items_r.to('cpu').detach().numpy().copy()

        x = cosine_similarity(all_users_r, all_items_r)

        cos_mat = torch.from_numpy(x.astype(np.float32)).clone()
        cos_mat.to(device=self.device)
        cos_mat = torch.reshape(cos_mat, (1, -1))
        _, idx = torch.topk(cos_mat, self.aug_num)

        del cos_mat
        torch.cuda.empty_cache()
        # gc.collect()
        # idx = idx.tolist()S
        aug_idx = [
            [int(torch.div(idx[0][i], self.n_items, rounding_mode='floor')), (int(torch.fmod(idx[0][i], self.n_items)))]
            for i in range(self.aug_num)]

        return aug_idx  # return list [user_id, item_id]

    def cal_cos_sim_v2(self):
        # calculate cosine similarity with user embeddings and item embeddings (on GPU)
        rep = self.get_def_rep()
        all_users_r = rep[:self.n_users, :]
        all_items_r = rep[self.n_users:, :]

        cos_sim = pairwise_cosine_similarity(all_users_r, all_items_r)
        cos_sim = torch.reshape(cos_sim, (1, -1))
        _, idx = torch.topk(cos_sim, self.aug_num)
        aug_idx = [
            [int(torch.div(idx[0][i], self.n_items, rounding_mode='floor')), (int(torch.fmod(idx[0][i], self.n_items)))]
            for i in range(self.aug_num)]

        return aug_idx  # return list [user_id, item_id]

    def cal_loss(self, users_r, aug_users_r):
        # calcrate ssl-loss(InfoNCE)
        loss = InfoNCE(negative_mode='unpaired')

        query = users_r
        positive_key = aug_users_r
        negative_keys = aug_users_r
        contrasive_loss = loss(query, positive_key, negative_keys)
        return contrasive_loss

    def bpr_forward(self, users, pos_items, neg_items):
        # 普通の埋め込み
        rep = self.get_def_rep()
        users_r = rep[users, :]
        pos_items_r, neg_items_r = rep[self.n_users + pos_items, :], rep[self.n_users + neg_items, :]
        # AUG後の埋め込み

        aug_rep = self.get_aug_rep(self.norm_aug_adj)
        aug_users_r = aug_rep[users, :]
        l2_norm_sq = torch.norm(users_r, p=2, dim=1) ** 2 + torch.norm(pos_items_r, p=2, dim=1) ** 2 \
                     + torch.norm(neg_items_r, p=2, dim=1) ** 2
        # Contrasive loss
        contrasive_loss = self.cal_loss(users_r, aug_users_r)
        return users_r, pos_items_r, neg_items_r, l2_norm_sq, contrasive_loss

    def predict(self, users):
        rep = self.get_def_rep()
        users_r = rep[users, :]
        all_items_r = rep[self.n_users:, :]
        scores = torch.mm(users_r, all_items_r.t())
        return scores

    def save(self, path):
        params = {'sate_dict': self.state_dict(), 'user_map': self.user_map,
                  'item_map': self.item_map, 'alpha': self.alpha}
        torch.save(params, path)

    def load(self, path):
        params = torch.load(path, map_location=self.device)
        self.load_state_dict(params['sate_dict'])
        self.user_map = params['user_map']
        self.item_map = params['item_map']
        self.alpha = params['alpha']
        self.feat_mat, _, _, self.row_sum = self.generate_feat(self.config['dataset'], is_updating=True)
        self.update_feat_mat()


class DOSE_drop(BasicModel):
    def __init__(self, model_config):
        super(DOSE_drop, self).__init__(model_config)
        self.embedding_size = model_config['embedding_size']
        self.n_layers = model_config['n_layers']
        self.dropout = model_config['dropout']
        self.feature_ratio = model_config['feature_ratio']  # Template%
        self.norm_adj = self.generate_graph(model_config['dataset'])

        self.alpha = 1.
        self.delta = model_config.get('delta', 0.99)
        self.taugh = model_config.get('taugh', 0.2)
        self.aug_num = model_config['aug_num']
        self.aug_num = int(self.aug_num)
        # self.temper = model_config['temper']
        self.feat_mat, self.user_map, self.item_map, self.row_sum = \
            self.generate_feat(model_config['dataset'],
                               ranking_metric=model_config.get('ranking_metric', 'sort'))
        self.update_feat_mat()
        # self.norm_aug_adj = generate_aug_graph(model_config['dataset'] )
        self.embedding = nn.Embedding(self.feat_mat.shape[1], self.embedding_size)

        self.w = nn.Parameter(torch.ones([self.embedding_size], dtype=torch.float32, device=self.device))
        normal_(self.embedding.weight, std=0.1)
        self.to(device=self.device)
        self.norm_aug_adj = self.generate_drop_graph(model_config['dataset'])
        self.times = model_config.get('times', 0.1)
        # self.drop_feat_mat, self.drop_user_map, self.drop_item_map, self.drop_row_sum = \
            # self.generate_drop_feat(model_config['dataset'],
                               # ranking_metric=model_config.get('ranking_metric', 'sort'))
        # self.aug_num = (self.times) * len(self.feat_mat)

    def update_feat_mat(self):
        row, _ = self.feat_mat.indices()
        edge_values = torch.pow(self.row_sum[row], (self.alpha - 1.) / 2. - 0.5)
        self.feat_mat = torch.sparse.FloatTensor(self.feat_mat.indices(), edge_values, self.feat_mat.shape).coalesce()

    def update_aug_feat_mat(self):
        row, _ = self.drop_feat_mat.indices()
        edge_values = torch.pow(self.row_sum[row], (self.alpha - 1.) / 2. - 0.5)
        self.drop_feat_mat = torch.sparse.FloatTensor(self.drop_feat_mat.indices(), edge_values,
                                                     self.drop_feat_mat.shape).coalesce()

    def feat_mat_anneal(self):
        self.alpha *= self.delta
        self.update_feat_mat()
    def update_aug_adj(self):
        del self.norm_aug_adj
        torch.cuda.empty_cache()
        gc.collect()
        self.norm_aug_adj = self.generate_drop_graph(self.config['dataset'])

    def generate_graph(self, dataset):
        return LightGCN.generate_graph(self, dataset)

    def generate_drop_graph(self, dataset):
        # new graph after adding some interactions
        aug_idx = self.cal_cos_sim()
        aug_adj_mat = generate_drop_daj_mat3(dataset, aug_idx)
        degree = np.array(np.sum(aug_adj_mat, axis=1)).squeeze()
        degree = np.maximum(1., degree)
        d_inv = np.power(degree, -0.5)  # 累乗
        d_mat = sp.diags(d_inv, format='csr', dtype=np.float32)

        norm_aug_adj = d_mat.dot(aug_adj_mat).dot(d_mat)
        norm_aug_adj = get_sparse_tensor(norm_aug_adj, self.device)
        return norm_aug_adj

    def generate_feat(self, dataset, is_updating=False, ranking_metric=None):
        # return adj matrix with template
        if not is_updating:
            if self.feature_ratio < 1.:  # ランク付後にスライス

                ranked_users, ranked_items = graph_rank_nodes(dataset, ranking_metric)
                core_users = ranked_users[:int(self.n_users * self.feature_ratio)]
                core_items = ranked_items[:int(self.n_items * self.feature_ratio)]
            else:
                core_users = np.arange(self.n_users, dtype=np.int64)
                core_items = np.arange(self.n_items, dtype=np.int64)

            user_map = dict()
            for idx, user in enumerate(core_users):
                user_map[user] = idx
            item_map = dict()
            for idx, item in enumerate(core_items):
                item_map[item] = idx
        else:
            user_map = self.user_map
            item_map = self.item_map

        user_dim, item_dim = len(user_map), len(item_map)
        indices = []
        for user, item in dataset.train_array:
            if item in item_map:
                indices.append([user, user_dim + item_map[item]])
            if user in user_map:
                indices.append([self.n_users + item, user_map[user]])
        for user in range(self.n_users):
            indices.append([user, user_dim + item_dim])
        for item in range(self.n_items):
            indices.append([self.n_users + item, user_dim + item_dim + 1])
        feat = sp.coo_matrix((np.ones((len(indices),)), np.array(indices).T),
                             shape=(self.n_users + self.n_items, user_dim + item_dim + 2), dtype=np.float32).tocsr()
        row_sum = torch.tensor(np.array(np.sum(feat, axis=1)).squeeze(), dtype=torch.float32, device=self.device)
        feat = get_sparse_tensor(feat, self.device)
        return feat, user_map, item_map, row_sum

    def generate_drop_feat(self, dataset, is_updating=False, ranking_metric=None):
        # return adj matrix with template
        if not is_updating:
            if self.feature_ratio < 1.:  # ランク付後にスライス

                ranked_users, ranked_items = graph_drop_rank_nodes(dataset, ranking_metric)
                core_users = ranked_users[:int(self.n_users * self.feature_ratio)]
                core_items = ranked_items[:int(self.n_items * self.feature_ratio)]
            else:
                core_users = np.arange(self.n_users, dtype=np.int64)
                core_items = np.arange(self.n_items, dtype=np.int64)

            user_map = dict()
            for idx, user in enumerate(core_users):
                user_map[user] = idx
            item_map = dict()
            for idx, item in enumerate(core_items):
                item_map[item] = idx
        else:
            user_map = self.user_map
            item_map = self.item_map

        user_dim, item_dim = len(user_map), len(item_map)

        train_array = dataset.train_array
        random.sample(train_array, int(len(train_array) * 0.5))
        indices = []
        for user, item in train_array:
            if item in item_map:
                indices.append([user, user_dim + item_map[item]])
            if user in user_map:
                indices.append([self.n_users + item, user_map[user]])
        for user in range(self.n_users):
            indices.append([user, user_dim + item_dim])
        for item in range(self.n_items):
            indices.append([self.n_users + item, user_dim + item_dim + 1])
        feat = sp.coo_matrix((np.ones((len(indices),)), np.array(indices).T),
                             shape=(self.n_users + self.n_items, user_dim + item_dim + 2), dtype=np.float32).tocsr()
        row_sum = torch.tensor(np.array(np.sum(feat, axis=1)).squeeze(), dtype=torch.float32, device=self.device)
        feat = get_sparse_tensor(feat, self.device)
        return feat, user_map, item_map, row_sum

    def inductive_rep_layer(self, feat_mat):
        # generate embedding by using template
        padding_tensor = torch.empty([max(self.feat_mat.shape) - self.feat_mat.shape[1], self.embedding_size],
                                     dtype=torch.float32, device=self.device)
        padding_features = torch.cat([self.embedding.weight, padding_tensor], dim=0)

        row, column = feat_mat.indices()
        g = dgl.graph((column, row), num_nodes=max(self.feat_mat.shape), device=self.device)
        x = dgl.ops.gspmm(g, 'mul', 'sum', lhs_data=padding_features, rhs_data=feat_mat.values())
        x = x[:self.feat_mat.shape[0], :]
        return x

    def inductive_drop_rep_layer(self, drop_feat_mat):
        # generate embedding by using template
        padding_tensor = torch.empty([max(self.drop_feat_mat.shape) - self.drop_feat_mat.shape[1], self.embedding_size],
                                     dtype=torch.float32, device=self.device)
        padding_features = torch.cat([self.embedding.weight, padding_tensor], dim=0)

        row, column = drop_feat_mat.indices()
        g = dgl.graph((column, row), num_nodes=max(self.drop_feat_mat.shape), device=self.device)
        x = dgl.ops.gspmm(g, 'mul', 'sum', lhs_data=padding_features, rhs_data=drop_feat_mat.values())
        x = x[:self.aug_feat_mat.shape[0], :]
        return x

    def get_def_rep(self):
        # generate final embedding
        feat_mat = NGCF.dropout_sp_mat(self, self.feat_mat)
        representations = self.inductive_rep_layer(feat_mat)

        all_layer_rep = [representations]
        row, column = self.norm_adj.indices()
        g = dgl.graph((column, row), num_nodes=self.norm_adj.shape[0], device=self.device)
        for _ in range(self.n_layers):
            representations = dgl.ops.gspmm(g, 'mul', 'sum', lhs_data=representations, rhs_data=self.norm_adj.values())
            all_layer_rep.append(representations)
        all_layer_rep = torch.stack(all_layer_rep, dim=0)
        final_rep = all_layer_rep.mean(dim=0)
        return final_rep

    def get_aug_rep(self, norm_aug_adj):
        # generate final embedding on aug-graph
        feat_mat = NGCF.dropout_sp_mat(self, self.feat_mat)
        representations = self.inductive_rep_layer(feat_mat)

        all_layer_rep = [representations]
        row, column = norm_aug_adj.indices()
        g = dgl.graph((column, row), num_nodes=norm_aug_adj.shape[0], device=self.device)
        for _ in range(self.n_layers):
            representations = dgl.ops.gspmm(g, 'mul', 'sum', lhs_data=representations, rhs_data=norm_aug_adj.values())
            all_layer_rep.append(representations)
        all_layer_rep = torch.stack(all_layer_rep, dim=0)
        final_rep = all_layer_rep.mean(dim=0)
        return final_rep

    def cal_cos_sim(self):
        # calculate cosine similarity with user embeddings and item embeddings (on CPU)

        rep = self.get_def_rep()
        all_users_r = rep[:self.n_users, :]
        all_items_r = rep[self.n_users:, :]


        all_user_r = all_users_r.to('cpu').detach().numpy().copy()
        all_item_r = all_items_r.to('cpu').detach().numpy().copy()

        x = cosine_similarity(all_user_r, all_item_r)

        del all_users_r, all_items_r
        del all_user_r, all_item_r
        torch.cuda.empty_cache()

        cos_mat = torch.from_numpy(x.astype(np.float32)).clone()
        cos_mat.to(device=self.device)
        cos_mat = torch.reshape(cos_mat, (1, -1))
        cos_mat = torch.squeeze(cos_mat)
        cos_mat1 = cos_mat[:len(cos_mat) // 2]
        _, idx1 = torch.topk(cos_mat1, self.aug_num // 2)
        cos_mat2 = cos_mat[len(cos_mat) // 2:]

        del cos_mat
        torch.cuda.empty_cache()
        # gc.collect()
        print(len(cos_mat2))
        _, idx2 = torch.topk(cos_mat2, self.aug_num // 2)
        # idx = idx.tolist()S
        aug_idx1 = [[int(torch.div(idx1[i], self.n_items, rounding_mode='floor')),
                     (int(torch.fmod(idx1[i], self.n_items)))] for i in range(self.aug_num // 2)]
        # aug_idx.to(device=self.device)
        aug_idx2 = [
            [int(torch.div(idx2[i] + self.aug_num // 2, self.n_items, rounding_mode='floor')),
             (int(torch.fmod(idx2[i] + self.aug_num // 2, self.n_items)))] for i
            in range(self.aug_num // 2)]
        aug_idx1.extend(aug_idx2)
        del aug_idx2, idx1, idx2
        torch.cuda.empty_cache()

        return aug_idx1  # return list [user_id, item_id]

    def cal_cos_sim_v2(self):
        # calculate cosine similarity with user embeddings and item embeddings (on GPU)
        rep = self.get_def_rep()
        all_users_r = rep[:self.n_users, :]
        all_items_r = rep[self.n_users:, :]

        cos_sim = pairwise_cosine_similarity(all_users_r, all_items_r)

        del all_users_r, all_items_r
        torch.cuda.empty_cache()

        cos_sim = torch.reshape(cos_sim, (1, -1))
        _, idx = torch.topk(cos_sim, self.aug_num)
        del cos_sim
        torch.cuda.empty_cache()

        aug_idx = [
            [int(torch.div(idx[0][i], self.n_items, rounding_mode='floor')), (int(torch.fmod(idx[0][i], self.n_items)))]
            for i in range(self.aug_num)]

        return aug_idx  # return list [user_id, item_id]

    def cal_loss(self, users_r, aug_users_r):
        # calcrate ssl-loss(InfoNCE)
        loss = InfoNCE(negative_mode='unpaired')

        query = users_r
        positive_key = aug_users_r
        negative_keys = aug_users_r
        contrasive_loss = loss(query, positive_key, negative_keys)
        return contrasive_loss
    def bpr_forward(self, users, pos_items, neg_items):
        # 普通の埋め込み
        rep = self.get_def_rep()
        users_r = rep[users, :]
        pos_items_r, neg_items_r = rep[self.n_users + pos_items, :], rep[self.n_users + neg_items, :]
        # AUG後の埋め込み

        aug_rep = self.get_aug_rep(self.norm_aug_adj)
        aug_users_r = aug_rep[users, :]
        l2_norm_sq = torch.norm(users_r, p=2, dim=1) ** 2 + torch.norm(pos_items_r, p=2, dim=1) ** 2 \
                     + torch.norm(neg_items_r, p=2, dim=1) ** 2
        # Contrasive loss
        contrasive_loss = self.cal_loss(users_r, aug_users_r)
        return users_r, pos_items_r, neg_items_r, l2_norm_sq, contrasive_loss

    def update_aug_adj(self):
        del self.norm_aug_adj
        torch.cuda.empty_cache()
        # gc.collect()
        self.norm_aug_adj = self.generate_drop_graph(self.config['dataset'])

    def predict(self, users):
        rep = self.get_def_rep()
        users_r = rep[users, :]
        all_items_r = rep[self.n_users:, :]
        scores = torch.mm(users_r, all_items_r.t())
        return scores

    def save(self, path):
        params = {'sate_dict': self.state_dict(), 'user_map': self.user_map,
                  'item_map': self.item_map, 'alpha': self.alpha}
        torch.save(params, path)

    def load(self, path):
        params = torch.load(path, map_location=self.device)
        self.load_state_dict(params['sate_dict'])
        self.user_map = params['user_map']
        self.item_map = params['item_map']
        self.alpha = params['alpha']
        self.feat_mat, _, _, self.row_sum = self.generate_feat(self.config['dataset'], is_updating=True)
        self.update_feat_mat()


class DOSE_drop2(BasicModel):
    def __init__(self, model_config):
        super(DOSE_drop2, self).__init__(model_config)
        self.embedding_size = model_config['embedding_size']
        self.n_layers = model_config['n_layers']
        self.dropout = model_config['dropout']
        self.feature_ratio = model_config['feature_ratio']  # Template%
        self.norm_adj = self.generate_graph(model_config['dataset'])

        self.alpha = 1.
        self.delta = model_config.get('delta', 0.99)
        self.taugh = model_config.get('taugh', 0.2)
        self.aug_rate = model_config.get('aug_rate', 0.2)
        self.aug_num = model_config['aug_num']
        self.aug_num = int(self.aug_num)
        # self.temper = model_config['temper']
        self.feat_mat, self.user_map, self.item_map, self.row_sum = \
            self.generate_feat(model_config['dataset'],
                               ranking_metric=model_config.get('ranking_metric', 'sort'))
        self.update_feat_mat()
        # self.norm_aug_adj = generate_aug_graph(model_config['dataset'] )
        self.embedding = nn.Embedding(self.feat_mat.shape[1], self.embedding_size)

        self.w = nn.Parameter(torch.ones([self.embedding_size], dtype=torch.float32, device=self.device))
        normal_(self.embedding.weight, std=0.1)
        self.to(device=self.device)
        self.norm_aug_adj = self.generate_drop_graph(model_config['dataset'])
        self.times = model_config.get('times', 0.1)
        # self.drop_feat_mat, self.drop_user_map, self.drop_item_map, self.drop_row_sum = \

            # self.generate_drop_feat(model_config['dataset'],
                               # ranking_metric=model_config.get('ranking_metric', 'sort'))
        # self.aug_num = (self.times) * len(self.feat_mat)

    def update_feat_mat(self):
        row, _ = self.feat_mat.indices()
        edge_values = torch.pow(self.row_sum[row], (self.alpha - 1.) / 2. - 0.5)
        self.feat_mat = torch.sparse.FloatTensor(self.feat_mat.indices(), edge_values, self.feat_mat.shape).coalesce()

    def update_aug_feat_mat(self):
        row, _ = self.drop_feat_mat.indices()
        edge_values = torch.pow(self.row_sum[row], (self.alpha - 1.) / 2. - 0.5)
        self.drop_feat_mat = torch.sparse.FloatTensor(self.drop_feat_mat.indices(), edge_values,
                                                     self.drop_feat_mat.shape).coalesce()

    def feat_mat_anneal(self):
        self.alpha *= self.delta
        self.update_feat_mat()

    def generate_graph(self, dataset):
        return LightGCN.generate_graph(self, dataset)

    def generate_drop_graph(self, dataset):
        # new graph after adding some interactions
        aug_adj_mat = generate_drop_daj_mat(dataset, aug_rate=self.aug_rate)
        degree = np.array(np.sum(aug_adj_mat, axis=1)).squeeze()
        degree = np.maximum(1., degree)
        d_inv = np.power(degree, -0.5)  # 累乗
        d_mat = sp.diags(d_inv, format='csr', dtype=np.float32)

        norm_aug_adj = d_mat.dot(aug_adj_mat).dot(d_mat)
        norm_aug_adj = get_sparse_tensor(norm_aug_adj, self.device)
        return norm_aug_adj

    def generate_feat(self, dataset, is_updating=False, ranking_metric=None):
        # return adj matrix with template
        if not is_updating:
            if self.feature_ratio < 1.:  # ランク付後にスライス

                ranked_users, ranked_items = graph_rank_nodes(dataset, ranking_metric)
                core_users = ranked_users[:int(self.n_users * self.feature_ratio)]
                core_items = ranked_items[:int(self.n_items * self.feature_ratio)]
            else:
                core_users = np.arange(self.n_users, dtype=np.int64)
                core_items = np.arange(self.n_items, dtype=np.int64)

            user_map = dict()
            for idx, user in enumerate(core_users):
                user_map[user] = idx
            item_map = dict()
            for idx, item in enumerate(core_items):
                item_map[item] = idx
        else:
            user_map = self.user_map
            item_map = self.item_map

        user_dim, item_dim = len(user_map), len(item_map)
        indices = []
        for user, item in dataset.train_array:
            if item in item_map:
                indices.append([user, user_dim + item_map[item]])
            if user in user_map:
                indices.append([self.n_users + item, user_map[user]])
        for user in range(self.n_users):
            indices.append([user, user_dim + item_dim])
        for item in range(self.n_items):
            indices.append([self.n_users + item, user_dim + item_dim + 1])
        feat = sp.coo_matrix((np.ones((len(indices),)), np.array(indices).T),
                             shape=(self.n_users + self.n_items, user_dim + item_dim + 2), dtype=np.float32).tocsr()
        row_sum = torch.tensor(np.array(np.sum(feat, axis=1)).squeeze(), dtype=torch.float32, device=self.device)
        feat = get_sparse_tensor(feat, self.device)
        return feat, user_map, item_map, row_sum

    def generate_drop_feat(self, dataset, is_updating=False, ranking_metric=None):
        # return adj matrix with template
        if not is_updating:
            if self.feature_ratio < 1.:  # ランク付後にスライス

                ranked_users, ranked_items = graph_drop_rank_nodes(dataset, ranking_metric)
                core_users = ranked_users[:int(self.n_users * self.feature_ratio)]
                core_items = ranked_items[:int(self.n_items * self.feature_ratio)]
            else:
                core_users = np.arange(self.n_users, dtype=np.int64)
                core_items = np.arange(self.n_items, dtype=np.int64)

            user_map = dict()
            for idx, user in enumerate(core_users):
                user_map[user] = idx
            item_map = dict()
            for idx, item in enumerate(core_items):
                item_map[item] = idx
        else:
            user_map = self.user_map
            item_map = self.item_map

        user_dim, item_dim = len(user_map), len(item_map)

        train_array = dataset.train_array
        random.sample(train_array, int(len(train_array) * 0.5))
        indices = []
        for user, item in train_array:
            if item in item_map:
                indices.append([user, user_dim + item_map[item]])
            if user in user_map:
                indices.append([self.n_users + item, user_map[user]])
        for user in range(self.n_users):
            indices.append([user, user_dim + item_dim])
        for item in range(self.n_items):
            indices.append([self.n_users + item, user_dim + item_dim + 1])
        feat = sp.coo_matrix((np.ones((len(indices),)), np.array(indices).T),
                             shape=(self.n_users + self.n_items, user_dim + item_dim + 2), dtype=np.float32).tocsr()
        row_sum = torch.tensor(np.array(np.sum(feat, axis=1)).squeeze(), dtype=torch.float32, device=self.device)
        feat = get_sparse_tensor(feat, self.device)
        return feat, user_map, item_map, row_sum

    def inductive_rep_layer(self, feat_mat):
        # generate embedding by using template
        padding_tensor = torch.empty([max(self.feat_mat.shape) - self.feat_mat.shape[1], self.embedding_size],
                                     dtype=torch.float32, device=self.device)
        padding_features = torch.cat([self.embedding.weight, padding_tensor], dim=0)

        row, column = feat_mat.indices()
        g = dgl.graph((column, row), num_nodes=max(self.feat_mat.shape), device=self.device)
        x = dgl.ops.gspmm(g, 'mul', 'sum', lhs_data=padding_features, rhs_data=feat_mat.values())
        x = x[:self.feat_mat.shape[0], :]
        return x

    def inductive_drop_rep_layer(self, drop_feat_mat):
        # generate embedding by using template
        padding_tensor = torch.empty([max(self.drop_feat_mat.shape) - self.drop_feat_mat.shape[1], self.embedding_size],
                                     dtype=torch.float32, device=self.device)
        padding_features = torch.cat([self.embedding.weight, padding_tensor], dim=0)

        row, column = drop_feat_mat.indices()
        g = dgl.graph((column, row), num_nodes=max(self.drop_feat_mat.shape), device=self.device)
        x = dgl.ops.gspmm(g, 'mul', 'sum', lhs_data=padding_features, rhs_data=drop_feat_mat.values())
        x = x[:self.aug_feat_mat.shape[0], :]
        return x

    def get_def_rep(self):
        # generate final embedding
        feat_mat = NGCF.dropout_sp_mat(self, self.feat_mat)
        representations = self.inductive_rep_layer(feat_mat)

        all_layer_rep = [representations]
        row, column = self.norm_adj.indices()
        g = dgl.graph((column, row), num_nodes=self.norm_adj.shape[0], device=self.device)
        for _ in range(self.n_layers):
            representations = dgl.ops.gspmm(g, 'mul', 'sum', lhs_data=representations, rhs_data=self.norm_adj.values())
            all_layer_rep.append(representations)
        all_layer_rep = torch.stack(all_layer_rep, dim=0)
        final_rep = all_layer_rep.mean(dim=0)
        return final_rep

    def get_aug_rep(self, norm_aug_adj):
        # generate final embedding on aug-graph
        feat_mat = NGCF.dropout_sp_mat(self, self.feat_mat)
        representations = self.inductive_rep_layer(feat_mat)

        all_layer_rep = [representations]
        row, column = norm_aug_adj.indices()
        g = dgl.graph((column, row), num_nodes=norm_aug_adj.shape[0], device=self.device)
        for _ in range(self.n_layers):
            representations = dgl.ops.gspmm(g, 'mul', 'sum', lhs_data=representations, rhs_data=norm_aug_adj.values())
            all_layer_rep.append(representations)
        all_layer_rep = torch.stack(all_layer_rep, dim=0)
        final_rep = all_layer_rep.mean(dim=0)
        return final_rep

    def cal_cos_sim(self):
        # calculate cosine similarity with user embeddings and item embeddings (on CPU)
        rep = self.get_def_rep()
        all_users_r = rep[:self.n_users, :]
        all_items_r = rep[self.n_users:, :]

        all_users_r = all_users_r.to('cpu').detach().numpy().copy()
        all_items_r = all_items_r.to('cpu').detach().numpy().copy()

        x = cosine_similarity(all_users_r, all_items_r)
        cos_mat = torch.from_numpy(x.astype(np.float32)).clone()
        cos_mat.to(device=self.device)
        cos_sim = torch.reshape(cos_mat, (1, -1))
        _, idx = torch.topk(cos_sim, self.aug_num)
        # idx = idx.tolist()S
        aug_idx = [
            [int(torch.div(idx[0][i], self.n_items, rounding_mode='floor')), (int(torch.fmod(idx[0][i], self.n_items)))]
            for i in range(self.aug_num)]

        return aug_idx  # return list [user_id, item_id]

    def cal_cos_sim_v2(self):
        # calculate cosine similarity with user embeddings and item embeddings (on GPU)
        rep = self.get_def_rep()
        all_users_r = rep[:self.n_users, :]
        all_items_r = rep[self.n_users:, :]

        cos_sim = pairwise_cosine_similarity(all_users_r, all_items_r)
        cos_sim = torch.reshape(cos_sim, (1, -1))
        _, idx = torch.topk(cos_sim, self.aug_num)
        aug_idx = [
            [int(torch.div(idx[0][i], self.n_items, rounding_mode='floor')), (int(torch.fmod(idx[0][i], self.n_items)))]
            for i in range(self.aug_num)]

        return aug_idx  # return list [user_id, item_id]

    def cal_loss(self, users_r, aug_users_r):
        # calcrate ssl-loss(InfoNCE)
        loss = InfoNCE(negative_mode='unpaired')

        query = users_r
        positive_key = aug_users_r
        negative_keys = aug_users_r
        contrasive_loss = loss(query, positive_key, negative_keys)
        return contrasive_loss

    def bpr_forward(self, users, pos_items, neg_items):
        # 普通の埋め込み
        rep = self.get_def_rep()
        users_r = rep[users, :]
        pos_items_r, neg_items_r = rep[self.n_users + pos_items, :], rep[self.n_users + neg_items, :]
        # AUG後の埋め込み

        aug_rep = self.get_aug_rep(self.norm_aug_adj)
        aug_users_r = aug_rep[users, :]
        l2_norm_sq = torch.norm(users_r, p=2, dim=1) ** 2 + torch.norm(pos_items_r, p=2, dim=1) ** 2 \
                     + torch.norm(neg_items_r, p=2, dim=1) ** 2
        # Contrasive loss
        contrasive_loss = self.cal_loss(users_r, aug_users_r)
        return users_r, pos_items_r, neg_items_r, l2_norm_sq, contrasive_loss



    def update_aug_adj(self):
        del self.norm_aug_adj
        torch.cuda.empty_cache()
        # gc.collect()
        self.norm_aug_adj = self.generate_drop_graph(self.config['dataset'])

    def predict(self, users):
        rep = self.get_def_rep()
        users_r = rep[users, :]
        all_items_r = rep[self.n_users:, :]
        scores = torch.mm(users_r, all_items_r.t())
        return scores

    def save(self, path):
        params = {'sate_dict': self.state_dict(), 'user_map': self.user_map,
                  'item_map': self.item_map, 'alpha': self.alpha}
        torch.save(params, path)

    def load(self, path):
        params = torch.load(path, map_location=self.device)
        self.load_state_dict(params['sate_dict'])
        self.user_map = params['user_map']
        self.item_map = params['item_map']
        self.alpha = params['alpha']
        self.feat_mat, _, _, self.row_sum = self.generate_feat(self.config['dataset'], is_updating=True)
        self.update_feat_mat()

class TEST(BasicModel):
    def __init__(self, model_config):
        super(TEST, self).__init__(model_config)
        self.embedding_size = model_config['embedding_size']
        self.n_layers = model_config['n_layers']
        self.dropout = model_config['dropout']
        self.feature_ratio = model_config['feature_ratio']  # Template%


        self.alpha = 1.
        self.delta = model_config.get('delta', 0.99)
        self.taugh = model_config.get('taugh', 0.2)
        self.aug_rate = model_config.get('aug_rate', 0.8)
        self.aug_num = model_config['aug_num']
        self.aug_num = int(self.aug_num)
        # self.temper = model_config['temper']
        self.feat_mat, self.user_map, self.item_map, self.row_sum = \
            self.generate_feat(model_config['dataset'],
                               ranking_metric=model_config.get('ranking_metric', 'sort'))
        self.update_feat_mat()
        # self.norm_aug_adj = generate_aug_graph(model_config['dataset'] )
        self.embedding = nn.Embedding(self.feat_mat.shape[1], self.embedding_size)

        self.w = nn.Parameter(torch.ones([self.embedding_size], dtype=torch.float32, device=self.device))
        normal_(self.embedding.weight, std=0.1)
        self.to(device=self.device)
        self.norm_adj = self.generate_drop_graph(model_config['dataset'])
        self.norm_aug_adj = self.generate_drop_graph(model_config['dataset'])
        self.times = model_config.get('times', 0.1)
        # self.drop_feat_mat, self.drop_user_map, self.drop_item_map, self.drop_row_sum = \

            # self.generate_drop_feat(model_config['dataset'],
                               # ranking_metric=model_config.get('ranking_metric', 'sort'))
        # self.aug_num = (self.times) * len(self.feat_mat)

    def update_feat_mat(self):
        row, _ = self.feat_mat.indices()
        edge_values = torch.pow(self.row_sum[row], (self.alpha - 1.) / 2. - 0.5)
        self.feat_mat = torch.sparse.FloatTensor(self.feat_mat.indices(), edge_values, self.feat_mat.shape).coalesce()

    def update_aug_feat_mat(self):
        row, _ = self.drop_feat_mat.indices()
        edge_values = torch.pow(self.row_sum[row], (self.alpha - 1.) / 2. - 0.5)
        self.drop_feat_mat = torch.sparse.FloatTensor(self.drop_feat_mat.indices(), edge_values,
                                                     self.drop_feat_mat.shape).coalesce()

    def feat_mat_anneal(self):
        self.alpha *= self.delta
        self.update_feat_mat()

    def generate_graph(self, dataset):
        return LightGCN.generate_graph(self, dataset)

    def generate_drop_graph(self, dataset):
        # new graph after adding some interactions
        aug_adj_mat = generate_drop_daj_mat(dataset, aug_rate=self.aug_rate)
        degree = np.array(np.sum(aug_adj_mat, axis=1)).squeeze()
        degree = np.maximum(1., degree)
        d_inv = np.power(degree, -0.5)  # 累乗
        d_mat = sp.diags(d_inv, format='csr', dtype=np.float32)

        norm_aug_adj = d_mat.dot(aug_adj_mat).dot(d_mat)
        norm_aug_adj = get_sparse_tensor(norm_aug_adj, self.device)
        return norm_aug_adj

    def generate_feat(self, dataset, is_updating=False, ranking_metric=None):
        # return adj matrix with template
        if not is_updating:
            if self.feature_ratio < 1.:  # ランク付後にスライス

                ranked_users, ranked_items = graph_rank_nodes(dataset, ranking_metric)
                core_users = ranked_users[:int(self.n_users * self.feature_ratio)]
                core_items = ranked_items[:int(self.n_items * self.feature_ratio)]
            else:
                core_users = np.arange(self.n_users, dtype=np.int64)
                core_items = np.arange(self.n_items, dtype=np.int64)

            user_map = dict()
            for idx, user in enumerate(core_users):
                user_map[user] = idx
            item_map = dict()
            for idx, item in enumerate(core_items):
                item_map[item] = idx
        else:
            user_map = self.user_map
            item_map = self.item_map

        user_dim, item_dim = len(user_map), len(item_map)
        indices = []
        for user, item in dataset.train_array:
            if item in item_map:
                indices.append([user, user_dim + item_map[item]])
            if user in user_map:
                indices.append([self.n_users + item, user_map[user]])
        for user in range(self.n_users):
            indices.append([user, user_dim + item_dim])
        for item in range(self.n_items):
            indices.append([self.n_users + item, user_dim + item_dim + 1])
        feat = sp.coo_matrix((np.ones((len(indices),)), np.array(indices).T),
                             shape=(self.n_users + self.n_items, user_dim + item_dim + 2), dtype=np.float32).tocsr()
        row_sum = torch.tensor(np.array(np.sum(feat, axis=1)).squeeze(), dtype=torch.float32, device=self.device)
        feat = get_sparse_tensor(feat, self.device)
        return feat, user_map, item_map, row_sum

    def generate_drop_feat(self, dataset, is_updating=False, ranking_metric=None):
        # return adj matrix with template
        if not is_updating:
            if self.feature_ratio < 1.:  # ランク付後にスライス

                ranked_users, ranked_items = graph_drop_rank_nodes(dataset, ranking_metric)
                core_users = ranked_users[:int(self.n_users * self.feature_ratio)]
                core_items = ranked_items[:int(self.n_items * self.feature_ratio)]
            else:
                core_users = np.arange(self.n_users, dtype=np.int64)
                core_items = np.arange(self.n_items, dtype=np.int64)

            user_map = dict()
            for idx, user in enumerate(core_users):
                user_map[user] = idx
            item_map = dict()
            for idx, item in enumerate(core_items):
                item_map[item] = idx
        else:
            user_map = self.user_map
            item_map = self.item_map

        user_dim, item_dim = len(user_map), len(item_map)

        train_array = dataset.train_array
        random.sample(train_array, int(len(train_array) * 0.5))
        indices = []
        for user, item in train_array:
            if item in item_map:
                indices.append([user, user_dim + item_map[item]])
            if user in user_map:
                indices.append([self.n_users + item, user_map[user]])
        for user in range(self.n_users):
            indices.append([user, user_dim + item_dim])
        for item in range(self.n_items):
            indices.append([self.n_users + item, user_dim + item_dim + 1])
        feat = sp.coo_matrix((np.ones((len(indices),)), np.array(indices).T),
                             shape=(self.n_users + self.n_items, user_dim + item_dim + 2), dtype=np.float32).tocsr()
        row_sum = torch.tensor(np.array(np.sum(feat, axis=1)).squeeze(), dtype=torch.float32, device=self.device)
        feat = get_sparse_tensor(feat, self.device)
        return feat, user_map, item_map, row_sum

    def inductive_rep_layer(self, feat_mat):
        # generate embedding by using template
        padding_tensor = torch.empty([max(self.feat_mat.shape) - self.feat_mat.shape[1], self.embedding_size],
                                     dtype=torch.float32, device=self.device)
        padding_features = torch.cat([self.embedding.weight, padding_tensor], dim=0)

        row, column = feat_mat.indices()
        g = dgl.graph((column, row), num_nodes=max(self.feat_mat.shape), device=self.device)
        x = dgl.ops.gspmm(g, 'mul', 'sum', lhs_data=padding_features, rhs_data=feat_mat.values())
        x = x[:self.feat_mat.shape[0], :]
        return x

    def inductive_drop_rep_layer(self, drop_feat_mat):
        # generate embedding by using template
        padding_tensor = torch.empty([max(self.drop_feat_mat.shape) - self.drop_feat_mat.shape[1], self.embedding_size],
                                     dtype=torch.float32, device=self.device)
        padding_features = torch.cat([self.embedding.weight, padding_tensor], dim=0)

        row, column = drop_feat_mat.indices()
        g = dgl.graph((column, row), num_nodes=max(self.drop_feat_mat.shape), device=self.device)
        x = dgl.ops.gspmm(g, 'mul', 'sum', lhs_data=padding_features, rhs_data=drop_feat_mat.values())
        x = x[:self.aug_feat_mat.shape[0], :]
        return x

    def get_def_rep(self):
        # generate final embedding
        feat_mat = NGCF.dropout_sp_mat(self, self.feat_mat)
        representations = self.inductive_rep_layer(feat_mat)

        all_layer_rep = [representations]
        row, column = self.norm_adj.indices()
        g = dgl.graph((column, row), num_nodes=self.norm_adj.shape[0], device=self.device)
        for _ in range(self.n_layers):
            representations = dgl.ops.gspmm(g, 'mul', 'sum', lhs_data=representations, rhs_data=self.norm_adj.values())
            all_layer_rep.append(representations)
        all_layer_rep = torch.stack(all_layer_rep, dim=0)
        final_rep = all_layer_rep.mean(dim=0)
        return final_rep

    def get_aug_rep(self, norm_aug_adj):
        # generate final embedding on aug-graph
        feat_mat = NGCF.dropout_sp_mat(self, self.feat_mat)
        representations = self.inductive_rep_layer(feat_mat)

        all_layer_rep = [representations]
        row, column = norm_aug_adj.indices()
        g = dgl.graph((column, row), num_nodes=norm_aug_adj.shape[0], device=self.device)
        for _ in range(self.n_layers):
            representations = dgl.ops.gspmm(g, 'mul', 'sum', lhs_data=representations, rhs_data=norm_aug_adj.values())
            all_layer_rep.append(representations)
        all_layer_rep = torch.stack(all_layer_rep, dim=0)
        final_rep = all_layer_rep.mean(dim=0)
        return final_rep

    def cal_cos_sim(self):
        # calculate cosine similarity with user embeddings and item embeddings (on CPU)
        rep = self.get_def_rep()
        all_users_r = rep[:self.n_users, :]
        all_items_r = rep[self.n_users:, :]

        all_users_r = all_users_r.to('cpu').detach().numpy().copy()
        all_items_r = all_items_r.to('cpu').detach().numpy().copy()

        x = cosine_similarity(all_users_r, all_items_r)
        cos_mat = torch.from_numpy(x.astype(np.float32)).clone()
        cos_mat.to(device=self.device)
        cos_sim = torch.reshape(cos_mat, (1, -1))
        _, idx = torch.topk(cos_sim, self.aug_num)
        # idx = idx.tolist()S
        aug_idx = [
            [int(torch.div(idx[0][i], self.n_items, rounding_mode='floor')), (int(torch.fmod(idx[0][i], self.n_items)))]
            for i in range(self.aug_num)]

        return aug_idx  # return list [user_id, item_id]

    def cal_cos_sim_v2(self):
        # calculate cosine similarity with user embeddings and item embeddings (on GPU)
        rep = self.get_def_rep()
        all_users_r = rep[:self.n_users, :]
        all_items_r = rep[self.n_users:, :]

        cos_sim = pairwise_cosine_similarity(all_users_r, all_items_r)
        cos_sim = torch.reshape(cos_sim, (1, -1))
        _, idx = torch.topk(cos_sim, self.aug_num)
        aug_idx = [
            [int(torch.div(idx[0][i], self.n_items, rounding_mode='floor')), (int(torch.fmod(idx[0][i], self.n_items)))]
            for i in range(self.aug_num)]

        return aug_idx  # return list [user_id, item_id]

    def cal_loss(self, users_r, aug_users_r):
        # calcrate ssl-loss(InfoNCE)
        loss = InfoNCE(negative_mode='unpaired')

        query = users_r
        positive_key = aug_users_r
        negative_keys = aug_users_r
        contrasive_loss = loss(query, positive_key, negative_keys)
        return contrasive_loss

    def bpr_forward(self, users, pos_items, neg_items):
        # 普通の埋め込み
        rep = self.get_def_rep()
        users_r = rep[users, :]
        pos_items_r, neg_items_r = rep[self.n_users + pos_items, :], rep[self.n_users + neg_items, :]
        # AUG後の埋め込み

        aug_rep = self.get_aug_rep(self.norm_aug_adj)
        aug_users_r = aug_rep[users, :]
        l2_norm_sq = torch.norm(users_r, p=2, dim=1) ** 2 + torch.norm(pos_items_r, p=2, dim=1) ** 2 \
                     + torch.norm(neg_items_r, p=2, dim=1) ** 2
        # Contrasive loss
        contrasive_loss = self.cal_loss(users_r, aug_users_r)
        return users_r, pos_items_r, neg_items_r, l2_norm_sq, contrasive_loss



    def update_aug_adj(self):
        del self.norm_aug_adj
        torch.cuda.empty_cache()
        # gc.collect()
        self.norm_aug_adj = self.generate_drop_graph(self.config['dataset'])

    def predict(self, users):
        rep = self.get_def_rep()
        users_r = rep[users, :]
        all_items_r = rep[self.n_users:, :]
        scores = torch.mm(users_r, all_items_r.t())
        return scores

    def save(self, path):
        params = {'sate_dict': self.state_dict(), 'user_map': self.user_map,
                  'item_map': self.item_map, 'alpha': self.alpha}
        torch.save(params, path)

    def load(self, path):
        params = torch.load(path, map_location=self.device)
        self.load_state_dict(params['sate_dict'])
        self.user_map = params['user_map']
        self.item_map = params['item_map']
        self.alpha = params['alpha']
        self.feat_mat, _, _, self.row_sum = self.generate_feat(self.config['dataset'], is_updating=True)
        self.update_feat_mat()

class TEST2(BasicModel):
    def __init__(self, model_config):
        super(TEST2, self).__init__(model_config)
        self.embedding_size = model_config['embedding_size']
        self.n_layers = model_config['n_layers']
        self.dropout = model_config['dropout']
        self.feature_ratio = model_config['feature_ratio']  # Template%
        self.norm_adj = self.generate_graph(model_config['dataset'])

        self.alpha = 1.
        self.delta = model_config.get('delta', 0.99)
        self.taugh = model_config.get('taugh', 0.2)
        self.aug_rate = model_config.get('aug_rate', 0.8)
        self.aug_num = model_config['aug_num']
        self.aug_num = int(self.aug_num)
        # self.temper = model_config['temper']
        self.feat_mat, self.user_map, self.item_map, self.row_sum = \
            self.generate_feat(model_config['dataset'],
                               ranking_metric=model_config.get('ranking_metric', 'sort'))
        self.update_feat_mat()
        # self.norm_aug_adj = generate_aug_graph(model_config['dataset'] )
        self.embedding = nn.Embedding(self.feat_mat.shape[1], self.embedding_size)

        self.w = nn.Parameter(torch.ones([self.embedding_size], dtype=torch.float32, device=self.device))
        normal_(self.embedding.weight, std=0.1)
        self.to(device=self.device)
        self.norm_aug_adj1 = self.generate_drop_graph(model_config['dataset'])
        self.norm_aug_adj2 = self.generate_drop_graph(model_config['dataset'])
        self.times = model_config.get('times', 0.1)
        # self.drop_feat_mat, self.drop_user_map, self.drop_item_map, self.drop_row_sum = \

            # self.generate_drop_feat(model_config['dataset'],
                               # ranking_metric=model_config.get('ranking_metric', 'sort'))
        # self.aug_num = (self.times) * len(self.feat_mat)

    def update_feat_mat(self):
        row, _ = self.feat_mat.indices()
        edge_values = torch.pow(self.row_sum[row], (self.alpha - 1.) / 2. - 0.5)
        self.feat_mat = torch.sparse.FloatTensor(self.feat_mat.indices(), edge_values, self.feat_mat.shape).coalesce()

    def update_aug_feat_mat(self):
        row, _ = self.drop_feat_mat.indices()
        edge_values = torch.pow(self.row_sum[row], (self.alpha - 1.) / 2. - 0.5)
        self.drop_feat_mat = torch.sparse.FloatTensor(self.drop_feat_mat.indices(), edge_values,
                                                     self.drop_feat_mat.shape).coalesce()

    def feat_mat_anneal(self):
        self.alpha *= self.delta
        self.update_feat_mat()

    def generate_graph(self, dataset):
        return LightGCN.generate_graph(self, dataset)

    def generate_drop_graph(self, dataset):
        # new graph after adding some interactions
        aug_adj_mat = generate_drop_daj_mat(dataset, aug_rate=self.aug_rate)
        degree = np.array(np.sum(aug_adj_mat, axis=1)).squeeze()
        degree = np.maximum(1., degree)
        d_inv = np.power(degree, -0.5)  # 累乗
        d_mat = sp.diags(d_inv, format='csr', dtype=np.float32)

        norm_aug_adj = d_mat.dot(aug_adj_mat).dot(d_mat)
        norm_aug_adj = get_sparse_tensor(norm_aug_adj, self.device)
        return norm_aug_adj

    def generate_feat(self, dataset, is_updating=False, ranking_metric=None):
        # return adj matrix with template
        if not is_updating:
            if self.feature_ratio < 1.:  # ランク付後にスライス

                ranked_users, ranked_items = graph_rank_nodes(dataset, ranking_metric)
                core_users = ranked_users[:int(self.n_users * self.feature_ratio)]
                core_items = ranked_items[:int(self.n_items * self.feature_ratio)]
            else:
                core_users = np.arange(self.n_users, dtype=np.int64)
                core_items = np.arange(self.n_items, dtype=np.int64)

            user_map = dict()
            for idx, user in enumerate(core_users):
                user_map[user] = idx
            item_map = dict()
            for idx, item in enumerate(core_items):
                item_map[item] = idx
        else:
            user_map = self.user_map
            item_map = self.item_map

        user_dim, item_dim = len(user_map), len(item_map)
        indices = []
        for user, item in dataset.train_array:
            if item in item_map:
                indices.append([user, user_dim + item_map[item]])
            if user in user_map:
                indices.append([self.n_users + item, user_map[user]])
        for user in range(self.n_users):
            indices.append([user, user_dim + item_dim])
        for item in range(self.n_items):
            indices.append([self.n_users + item, user_dim + item_dim + 1])
        feat = sp.coo_matrix((np.ones((len(indices),)), np.array(indices).T),
                             shape=(self.n_users + self.n_items, user_dim + item_dim + 2), dtype=np.float32).tocsr()
        row_sum = torch.tensor(np.array(np.sum(feat, axis=1)).squeeze(), dtype=torch.float32, device=self.device)
        feat = get_sparse_tensor(feat, self.device)
        return feat, user_map, item_map, row_sum

    def generate_drop_feat(self, dataset, is_updating=False, ranking_metric=None):
        # return adj matrix with template
        if not is_updating:
            if self.feature_ratio < 1.:  # ランク付後にスライス

                ranked_users, ranked_items = graph_drop_rank_nodes(dataset, ranking_metric)
                core_users = ranked_users[:int(self.n_users * self.feature_ratio)]
                core_items = ranked_items[:int(self.n_items * self.feature_ratio)]
            else:
                core_users = np.arange(self.n_users, dtype=np.int64)
                core_items = np.arange(self.n_items, dtype=np.int64)

            user_map = dict()
            for idx, user in enumerate(core_users):
                user_map[user] = idx
            item_map = dict()
            for idx, item in enumerate(core_items):
                item_map[item] = idx
        else:
            user_map = self.user_map
            item_map = self.item_map

        user_dim, item_dim = len(user_map), len(item_map)

        train_array = dataset.train_array
        random.sample(train_array, int(len(train_array) * 0.5))
        indices = []
        for user, item in train_array:
            if item in item_map:
                indices.append([user, user_dim + item_map[item]])
            if user in user_map:
                indices.append([self.n_users + item, user_map[user]])
        for user in range(self.n_users):
            indices.append([user, user_dim + item_dim])
        for item in range(self.n_items):
            indices.append([self.n_users + item, user_dim + item_dim + 1])
        feat = sp.coo_matrix((np.ones((len(indices),)), np.array(indices).T),
                             shape=(self.n_users + self.n_items, user_dim + item_dim + 2), dtype=np.float32).tocsr()
        row_sum = torch.tensor(np.array(np.sum(feat, axis=1)).squeeze(), dtype=torch.float32, device=self.device)
        feat = get_sparse_tensor(feat, self.device)
        return feat, user_map, item_map, row_sum

    def inductive_rep_layer(self, feat_mat):
        # generate embedding by using template
        padding_tensor = torch.empty([max(self.feat_mat.shape) - self.feat_mat.shape[1], self.embedding_size],
                                     dtype=torch.float32, device=self.device)
        padding_features = torch.cat([self.embedding.weight, padding_tensor], dim=0)

        row, column = feat_mat.indices()
        g = dgl.graph((column, row), num_nodes=max(self.feat_mat.shape), device=self.device)
        x = dgl.ops.gspmm(g, 'mul', 'sum', lhs_data=padding_features, rhs_data=feat_mat.values())
        x = x[:self.feat_mat.shape[0], :]
        return x

    def inductive_drop_rep_layer(self, drop_feat_mat):
        # generate embedding by using template
        padding_tensor = torch.empty([max(self.drop_feat_mat.shape) - self.drop_feat_mat.shape[1], self.embedding_size],
                                     dtype=torch.float32, device=self.device)
        padding_features = torch.cat([self.embedding.weight, padding_tensor], dim=0)

        row, column = drop_feat_mat.indices()
        g = dgl.graph((column, row), num_nodes=max(self.drop_feat_mat.shape), device=self.device)
        x = dgl.ops.gspmm(g, 'mul', 'sum', lhs_data=padding_features, rhs_data=drop_feat_mat.values())
        x = x[:self.aug_feat_mat.shape[0], :]
        return x

    def get_def_rep(self):
        # generate final embedding
        feat_mat = NGCF.dropout_sp_mat(self, self.feat_mat)
        representations = self.inductive_rep_layer(feat_mat)

        all_layer_rep = [representations]
        row, column = self.norm_adj.indices()
        g = dgl.graph((column, row), num_nodes=self.norm_adj.shape[0], device=self.device)
        for _ in range(self.n_layers):
            representations = dgl.ops.gspmm(g, 'mul', 'sum', lhs_data=representations, rhs_data=self.norm_adj.values())
            all_layer_rep.append(representations)
        all_layer_rep = torch.stack(all_layer_rep, dim=0)
        final_rep = all_layer_rep.mean(dim=0)
        return final_rep

    def get_aug_rep(self, norm_aug_adj):
        # generate final embedding on aug-graph
        feat_mat = NGCF.dropout_sp_mat(self, self.feat_mat)
        representations = self.inductive_rep_layer(feat_mat)

        all_layer_rep = [representations]
        row, column = norm_aug_adj.indices()
        g = dgl.graph((column, row), num_nodes=norm_aug_adj.shape[0], device=self.device)
        for _ in range(self.n_layers):
            representations = dgl.ops.gspmm(g, 'mul', 'sum', lhs_data=representations, rhs_data=norm_aug_adj.values())
            all_layer_rep.append(representations)
        all_layer_rep = torch.stack(all_layer_rep, dim=0)
        final_rep = all_layer_rep.mean(dim=0)
        return final_rep

    def cal_cos_sim(self):
        # calculate cosine similarity with user embeddings and item embeddings (on CPU)
        rep = self.get_def_rep()
        all_users_r = rep[:self.n_users, :]
        all_items_r = rep[self.n_users:, :]

        all_users_r = all_users_r.to('cpu').detach().numpy().copy()
        all_items_r = all_items_r.to('cpu').detach().numpy().copy()

        x = cosine_similarity(all_users_r, all_items_r)
        cos_mat = torch.from_numpy(x.astype(np.float32)).clone()
        cos_mat.to(device=self.device)
        cos_sim = torch.reshape(cos_mat, (1, -1))
        _, idx = torch.topk(cos_sim, self.aug_num)
        # idx = idx.tolist()S
        aug_idx = [
            [int(torch.div(idx[0][i], self.n_items, rounding_mode='floor')), (int(torch.fmod(idx[0][i], self.n_items)))]
            for i in range(self.aug_num)]

        return aug_idx  # return list [user_id, item_id]

    def cal_cos_sim_v2(self):
        # calculate cosine similarity with user embeddings and item embeddings (on GPU)
        rep = self.get_def_rep()
        all_users_r = rep[:self.n_users, :]
        all_items_r = rep[self.n_users:, :]

        cos_sim = pairwise_cosine_similarity(all_users_r, all_items_r)
        cos_sim = torch.reshape(cos_sim, (1, -1))
        _, idx = torch.topk(cos_sim, self.aug_num)
        aug_idx = [
            [int(torch.div(idx[0][i], self.n_items, rounding_mode='floor')), (int(torch.fmod(idx[0][i], self.n_items)))]
            for i in range(self.aug_num)]

        return aug_idx  # return list [user_id, item_id]

    def cal_loss(self, users_r, aug_users_r):
        # calcrate ssl-loss(InfoNCE)
        loss = InfoNCE(negative_mode='unpaired')

        query = users_r
        positive_key = aug_users_r
        negative_keys = aug_users_r
        contrasive_loss = loss(query, positive_key, negative_keys)
        return contrasive_loss

    def bpr_forward(self, users, pos_items, neg_items):
        # 普通の埋め込み
        rep = self.get_def_rep()
        users_r = rep[users, :]
        pos_items_r, neg_items_r = rep[self.n_users + pos_items, :], rep[self.n_users + neg_items, :]
        # AUG後の埋め込み

        aug_rep1 = self.get_aug_rep(self.norm_aug_adj1)
        aug_users_r1 = aug_rep1[users, :]
        aug_rep2 = self.get_aug_rep(self.norm_aug_adj2)
        aug_users_r2 = aug_rep2[users, :]
        l2_norm_sq = torch.norm(users_r, p=2, dim=1) ** 2 + torch.norm(pos_items_r, p=2, dim=1) ** 2 \
                     + torch.norm(neg_items_r, p=2, dim=1) ** 2
        # Contrasive loss
        contrasive_loss = self.cal_loss(aug_users_r1, aug_users_r2)
        return users_r, pos_items_r, neg_items_r, l2_norm_sq, contrasive_loss



    def update_aug_adj(self):
        del self.norm_aug_adj1, self.norm_aug_adj2
        torch.cuda.empty_cache()
        # gc.collect()
        self.norm_aug_adj1 = self.generate_drop_graph(self.config['dataset'])
        self.norm_aug_adj2 = self.generate_drop_graph(self.config['dataset'])
    def predict(self, users):
        rep = self.get_def_rep()
        users_r = rep[users, :]
        all_items_r = rep[self.n_users:, :]
        scores = torch.mm(users_r, all_items_r.t())
        return scores

    def save(self, path):
        params = {'sate_dict': self.state_dict(), 'user_map': self.user_map,
                  'item_map': self.item_map, 'alpha': self.alpha}
        torch.save(params, path)

    def load(self, path):
        params = torch.load(path, map_location=self.device)
        self.load_state_dict(params['sate_dict'])
        self.user_map = params['user_map']
        self.item_map = params['item_map']
        self.alpha = params['alpha']
        self.feat_mat, _, _, self.row_sum = self.generate_feat(self.config['dataset'], is_updating=True)
        self.update_feat_mat()
class DOSE_drop3(BasicModel):
    def __init__(self, model_config):
        super(DOSE_drop3, self).__init__(model_config)
        self.embedding_size = model_config['embedding_size']
        self.n_layers = model_config['n_layers']
        self.dropout = model_config['dropout']
        self.feature_ratio = model_config['feature_ratio']  # Template%
        self.norm_adj = self.generate_graph(model_config['dataset'])

        self.alpha = 1.
        self.delta = model_config.get('delta', 0.99)
        self.taugh = model_config.get('taugh', 0.2)
        self.aug_num = model_config['aug_num']
        self.aug_num = int(self.aug_num)
        # self.temper = model_config['temper']
        self.feat_mat, self.user_map, self.item_map, self.row_sum = \
            self.generate_feat(model_config['dataset'],
                               ranking_metric=model_config.get('ranking_metric', 'sort'))
        self.update_feat_mat()
        # self.norm_aug_adj = generate_aug_graph(model_config['dataset'] )
        self.embedding = nn.Embedding(self.feat_mat.shape[1], self.embedding_size)

        self.w = nn.Parameter(torch.ones([self.embedding_size], dtype=torch.float32, device=self.device))
        normal_(self.embedding.weight, std=0.1)
        self.to(device=self.device)
        self.norm_aug_adj = self.generate_drop_graph(model_config['dataset'])
        self.times = model_config.get('times', 0.1)
        # self.drop_feat_mat, self.drop_user_map, self.drop_item_map, self.drop_row_sum = \
            # self.generate_drop_feat(model_config['dataset'],
                               # ranking_metric=model_config.get('ranking_metric', 'sort'))
        # self.aug_num = (self.times) * len(self.feat_mat)

    def update_feat_mat(self):
        row, _ = self.feat_mat.indices()
        edge_values = torch.pow(self.row_sum[row], (self.alpha - 1.) / 2. - 0.5)
        self.feat_mat = torch.sparse.FloatTensor(self.feat_mat.indices(), edge_values, self.feat_mat.shape).coalesce()

    def update_aug_feat_mat(self):
        row, _ = self.drop_feat_mat.indices()
        edge_values = torch.pow(self.row_sum[row], (self.alpha - 1.) / 2. - 0.5)
        self.drop_feat_mat = torch.sparse.FloatTensor(self.drop_feat_mat.indices(), edge_values,
                                                     self.drop_feat_mat.shape).coalesce()

    def feat_mat_anneal(self):
        self.alpha *= self.delta
        self.update_feat_mat()
    def update_aug_adj(self):
        del self.norm_aug_adj
        torch.cuda.empty_cache()
        gc.collect()
        self.norm_aug_adj = self.generate_drop_graph(self.config['dataset'])

    def generate_graph(self, dataset):
        return LightGCN.generate_graph(self, dataset)

    def generate_drop_graph(self, dataset):
        # new graph after adding some interactions
        aug_idx = self.cal_cos_sim()
        aug_adj_mat = generate_drop_daj_mat3(dataset, aug_idx)
        degree = np.array(np.sum(aug_adj_mat, axis=1)).squeeze()
        degree = np.maximum(1., degree)
        d_inv = np.power(degree, -0.5)  # 累乗
        d_mat = sp.diags(d_inv, format='csr', dtype=np.float32)

        norm_aug_adj = d_mat.dot(aug_adj_mat).dot(d_mat)
        norm_aug_adj = get_sparse_tensor(norm_aug_adj, self.device)
        return norm_aug_adj

    def generate_feat(self, dataset, is_updating=False, ranking_metric=None):
        # return adj matrix with template
        if not is_updating:
            if self.feature_ratio < 1.:  # ランク付後にスライス

                ranked_users, ranked_items = graph_rank_nodes(dataset, ranking_metric)
                core_users = ranked_users[:int(self.n_users * self.feature_ratio)]
                core_items = ranked_items[:int(self.n_items * self.feature_ratio)]
            else:
                core_users = np.arange(self.n_users, dtype=np.int64)
                core_items = np.arange(self.n_items, dtype=np.int64)

            user_map = dict()
            for idx, user in enumerate(core_users):
                user_map[user] = idx
            item_map = dict()
            for idx, item in enumerate(core_items):
                item_map[item] = idx
        else:
            user_map = self.user_map
            item_map = self.item_map

        user_dim, item_dim = len(user_map), len(item_map)
        indices = []
        for user, item in dataset.train_array:
            if item in item_map:
                indices.append([user, user_dim + item_map[item]])
            if user in user_map:
                indices.append([self.n_users + item, user_map[user]])
        for user in range(self.n_users):
            indices.append([user, user_dim + item_dim])
        for item in range(self.n_items):
            indices.append([self.n_users + item, user_dim + item_dim + 1])
        feat = sp.coo_matrix((np.ones((len(indices),)), np.array(indices).T),
                             shape=(self.n_users + self.n_items, user_dim + item_dim + 2), dtype=np.float32).tocsr()
        row_sum = torch.tensor(np.array(np.sum(feat, axis=1)).squeeze(), dtype=torch.float32, device=self.device)
        feat = get_sparse_tensor(feat, self.device)
        return feat, user_map, item_map, row_sum

    def generate_drop_feat(self, dataset, is_updating=False, ranking_metric=None):
        # return adj matrix with template
        if not is_updating:
            if self.feature_ratio < 1.:  # ランク付後にスライス

                ranked_users, ranked_items = graph_drop_rank_nodes(dataset, ranking_metric)
                core_users = ranked_users[:int(self.n_users * self.feature_ratio)]
                core_items = ranked_items[:int(self.n_items * self.feature_ratio)]
            else:
                core_users = np.arange(self.n_users, dtype=np.int64)
                core_items = np.arange(self.n_items, dtype=np.int64)

            user_map = dict()
            for idx, user in enumerate(core_users):
                user_map[user] = idx
            item_map = dict()
            for idx, item in enumerate(core_items):
                item_map[item] = idx
        else:
            user_map = self.user_map
            item_map = self.item_map

        user_dim, item_dim = len(user_map), len(item_map)

        train_array = dataset.train_array
        random.sample(train_array, int(len(train_array) * 0.5))
        indices = []
        for user, item in train_array:
            if item in item_map:
                indices.append([user, user_dim + item_map[item]])
            if user in user_map:
                indices.append([self.n_users + item, user_map[user]])
        for user in range(self.n_users):
            indices.append([user, user_dim + item_dim])
        for item in range(self.n_items):
            indices.append([self.n_users + item, user_dim + item_dim + 1])
        feat = sp.coo_matrix((np.ones((len(indices),)), np.array(indices).T),
                             shape=(self.n_users + self.n_items, user_dim + item_dim + 2), dtype=np.float32).tocsr()
        row_sum = torch.tensor(np.array(np.sum(feat, axis=1)).squeeze(), dtype=torch.float32, device=self.device)
        feat = get_sparse_tensor(feat, self.device)
        return feat, user_map, item_map, row_sum

    def inductive_rep_layer(self, feat_mat):
        # generate embedding by using template
        padding_tensor = torch.empty([max(self.feat_mat.shape) - self.feat_mat.shape[1], self.embedding_size],
                                     dtype=torch.float32, device=self.device)
        padding_features = torch.cat([self.embedding.weight, padding_tensor], dim=0)

        row, column = feat_mat.indices()
        g = dgl.graph((column, row), num_nodes=max(self.feat_mat.shape), device=self.device)
        x = dgl.ops.gspmm(g, 'mul', 'sum', lhs_data=padding_features, rhs_data=feat_mat.values())
        x = x[:self.feat_mat.shape[0], :]
        return x

    def inductive_drop_rep_layer(self, drop_feat_mat):
        # generate embedding by using template
        padding_tensor = torch.empty([max(self.drop_feat_mat.shape) - self.drop_feat_mat.shape[1], self.embedding_size],
                                     dtype=torch.float32, device=self.device)
        padding_features = torch.cat([self.embedding.weight, padding_tensor], dim=0)

        row, column = drop_feat_mat.indices()
        g = dgl.graph((column, row), num_nodes=max(self.drop_feat_mat.shape), device=self.device)
        x = dgl.ops.gspmm(g, 'mul', 'sum', lhs_data=padding_features, rhs_data=drop_feat_mat.values())
        x = x[:self.aug_feat_mat.shape[0], :]
        return x

    def get_def_rep(self):
        # generate final embedding
        feat_mat = NGCF.dropout_sp_mat(self, self.feat_mat)
        representations = self.inductive_rep_layer(feat_mat)

        all_layer_rep = [representations]
        row, column = self.norm_adj.indices()
        g = dgl.graph((column, row), num_nodes=self.norm_adj.shape[0], device=self.device)
        for _ in range(self.n_layers):
            representations = dgl.ops.gspmm(g, 'mul', 'sum', lhs_data=representations, rhs_data=self.norm_adj.values())
            all_layer_rep.append(representations)
        all_layer_rep = torch.stack(all_layer_rep, dim=0)
        final_rep = all_layer_rep.mean(dim=0)
        return final_rep

    def get_aug_rep(self, norm_aug_adj):
        # generate final embedding on aug-graph
        feat_mat = NGCF.dropout_sp_mat(self, self.feat_mat)
        representations = self.inductive_rep_layer(feat_mat)

        all_layer_rep = [representations]
        row, column = norm_aug_adj.indices()
        g = dgl.graph((column, row), num_nodes=norm_aug_adj.shape[0], device=self.device)
        for _ in range(self.n_layers):
            representations = dgl.ops.gspmm(g, 'mul', 'sum', lhs_data=representations, rhs_data=norm_aug_adj.values())
            all_layer_rep.append(representations)
        all_layer_rep = torch.stack(all_layer_rep, dim=0)
        final_rep = all_layer_rep.mean(dim=0)
        return final_rep

    def cal_cos_sim(self):
        # calculate cosine similarity with user embeddings and item embeddings (on CPU)

        rep = self.get_def_rep()
        all_users_r = rep[:self.n_users, :]
        all_items_r = rep[self.n_users:, :]
        all_items_r *= -1

        all_user_r = all_users_r.to('cpu').detach().numpy().copy()
        all_item_r = all_items_r.to('cpu').detach().numpy().copy()

        x = cosine_similarity(all_user_r, all_item_r)

        del all_users_r, all_items_r
        del all_user_r, all_item_r
        torch.cuda.empty_cache()

        cos_mat = torch.from_numpy(x.astype(np.float32)).clone()
        cos_mat.to(device=self.device)
        cos_mat = torch.reshape(cos_mat, (1, -1))
        cos_mat = torch.squeeze(cos_mat)
        cos_mat1 = cos_mat[:len(cos_mat) // 2]
        _, idx1 = torch.topk(cos_mat1, self.aug_num // 2)
        cos_mat2 = cos_mat[len(cos_mat) // 2:]

        del cos_mat
        torch.cuda.empty_cache()
        # gc.collect()
        print(len(cos_mat2))
        _, idx2 = torch.topk(cos_mat2, self.aug_num // 2)
        # idx = idx.tolist()S
        aug_idx1 = [[int(torch.div(idx1[i], self.n_items, rounding_mode='floor')),
                     (int(torch.fmod(idx1[i], self.n_items)))] for i in range(self.aug_num // 2)]
        # aug_idx.to(device=self.device)
        aug_idx2 = [
            [int(torch.div(idx2[i] + self.aug_num // 2, self.n_items, rounding_mode='floor')),
             (int(torch.fmod(idx2[i] + self.aug_num // 2, self.n_items)))] for i
            in range(self.aug_num // 2)]
        aug_idx1.extend(aug_idx2)
        del aug_idx2, idx1, idx2
        torch.cuda.empty_cache()

        return aug_idx1  # return list [user_id, item_id]

    def cal_cos_sim_v2(self):
        # calculate cosine similarity with user embeddings and item embeddings (on GPU)
        rep = self.get_def_rep()
        all_users_r = rep[:self.n_users, :]
        all_items_r = rep[self.n_users:, :]

        cos_sim = pairwise_cosine_similarity(all_users_r, all_items_r)

        del all_users_r, all_items_r
        torch.cuda.empty_cache()

        cos_sim = torch.reshape(cos_sim, (1, -1))
        _, idx = torch.topk(cos_sim, self.aug_num)
        del cos_sim
        torch.cuda.empty_cache()

        aug_idx = [
            [int(torch.div(idx[0][i], self.n_items, rounding_mode='floor')), (int(torch.fmod(idx[0][i], self.n_items)))]
            for i in range(self.aug_num)]

        return aug_idx  # return list [user_id, item_id]

    def cal_loss(self, users_r, aug_users_r):
        # calcrate ssl-loss(InfoNCE)
        loss = InfoNCE(negative_mode='unpaired')

        query = users_r
        positive_key = aug_users_r
        negative_keys = aug_users_r
        contrasive_loss = loss(query, positive_key, negative_keys)
        return contrasive_loss
    def bpr_forward(self, users, pos_items, neg_items):
        # 普通の埋め込み
        rep = self.get_def_rep()
        users_r = rep[users, :]
        pos_items_r, neg_items_r = rep[self.n_users + pos_items, :], rep[self.n_users + neg_items, :]
        # AUG後の埋め込み

        aug_rep = self.get_aug_rep(self.norm_aug_adj)
        aug_users_r = aug_rep[users, :]
        l2_norm_sq = torch.norm(users_r, p=2, dim=1) ** 2 + torch.norm(pos_items_r, p=2, dim=1) ** 2 \
                     + torch.norm(neg_items_r, p=2, dim=1) ** 2
        # Contrasive loss
        contrasive_loss = self.cal_loss(users_r, aug_users_r)
        return users_r, pos_items_r, neg_items_r, l2_norm_sq, contrasive_loss

    def update_aug_adj(self):
        del self.norm_aug_adj
        torch.cuda.empty_cache()
        # gc.collect()
        self.norm_aug_adj = self.generate_drop_graph(self.config['dataset'])

    def predict(self, users):
        rep = self.get_def_rep()
        users_r = rep[users, :]
        all_items_r = rep[self.n_users:, :]
        scores = torch.mm(users_r, all_items_r.t())
        return scores

    def save(self, path):
        params = {'sate_dict': self.state_dict(), 'user_map': self.user_map,
                  'item_map': self.item_map, 'alpha': self.alpha}
        torch.save(params, path)

    def load(self, path):
        params = torch.load(path, map_location=self.device)
        self.load_state_dict(params['sate_dict'])
        self.user_map = params['user_map']
        self.item_map = params['item_map']
        self.alpha = params['alpha']
        self.feat_mat, _, _, self.row_sum = self.generate_feat(self.config['dataset'], is_updating=True)
        self.update_feat_mat()


class DOSE_aug_drop(BasicModel):
    def __init__(self, model_config):
        super(DOSE_aug_drop, self).__init__(model_config)
        self.embedding_size = model_config['embedding_size']
        self.n_layers = model_config['n_layers']
        self.dropout = model_config['dropout']
        self.feature_ratio = model_config['feature_ratio']  # Template%
        self.norm_adj = self.generate_graph(model_config['dataset'])

        self.alpha = 1.
        self.delta = model_config.get('delta', 0.99)
        self.taugh = model_config.get('taugh', 0.2)
        self.aug_rate = model_config.get('aug_rate', 0.8)
        self.aug_num = model_config['aug_num']
        # self.temper = model_config['temper']
        self.feat_mat, self.user_map, self.item_map, self.row_sum = \
            self.generate_feat(model_config['dataset'],
                               ranking_metric=model_config.get('ranking_metric', 'sort'))
        self.update_feat_mat()
        # self.norm_aug_adj = generate_aug_graph(model_config['dataset'] )
        self.embedding = nn.Embedding(self.feat_mat.shape[1], self.embedding_size)

        self.w = nn.Parameter(torch.ones([self.embedding_size], dtype=torch.float32, device=self.device))
        normal_(self.embedding.weight, std=0.1)
        self.to(device=self.device)
        self.norm_aug_adj = self.generate_aug_graph(model_config['dataset'])
        self.norm_drop_adj = self.generate_drop_graph(model_config['dataset'])
        self.times = model_config.get('times', 0.1)
        # self.aug_num = (self.times) * len(self.feat_mat)

    def update_feat_mat(self):
        row, _ = self.feat_mat.indices()
        edge_values = torch.pow(self.row_sum[row], (self.alpha - 1.) / 2. - 0.5)
        self.feat_mat = torch.sparse.FloatTensor(self.feat_mat.indices(), edge_values, self.feat_mat.shape).coalesce()

    def feat_mat_anneal(self):
        self.alpha *= self.delta
        self.update_feat_mat()

    def generate_graph(self, dataset):
        return LightGCN.generate_graph(self, dataset)

    def generate_aug_graph(self, dataset):
        # new graph after adding some interactions
        user_id = np.random.randint(0, self.n_users, self.aug_num)
        item_id = np.random.randint(0, self.n_items, self.aug_num)
        aug_idx = np.stack([user_id, item_id], 1)
        aug_idx = aug_idx.tolist()
        aug_adj_mat = generate_aug_daj_mat(dataset, aug_idx)
        degree = np.array(np.sum(aug_adj_mat, axis=1)).squeeze()
        degree = np.maximum(1., degree)
        d_inv = np.power(degree, -0.5)  # 累乗
        d_mat = sp.diags(d_inv, format='csr', dtype=np.float32)

        norm_aug_adj = d_mat.dot(aug_adj_mat).dot(d_mat)
        norm_aug_adj = get_sparse_tensor(norm_aug_adj, self.device)

        return norm_aug_adj



    def generate_drop_graph(self, dataset):
        # new graph after adding some interactions
        aug_adj_mat = generate_drop_daj_mat(dataset, aug_rate=self.aug_rate)
        degree = np.array(np.sum(aug_adj_mat, axis=1)).squeeze()
        degree = np.maximum(1., degree)
        d_inv = np.power(degree, -0.5)  # 累乗
        d_mat = sp.diags(d_inv, format='csr', dtype=np.float32)

        norm_aug_adj = d_mat.dot(aug_adj_mat).dot(d_mat)
        norm_aug_adj = get_sparse_tensor(norm_aug_adj, self.device)
        return norm_aug_adj


    def generate_feat(self, dataset, is_updating=False, ranking_metric=None):
        # return adj matrix with template
        if not is_updating:
            if self.feature_ratio < 1.:  # ランク付後にスライス

                ranked_users, ranked_items = graph_rank_nodes(dataset, ranking_metric)
                core_users = ranked_users[:int(self.n_users * self.feature_ratio)]
                core_items = ranked_items[:int(self.n_items * self.feature_ratio)]
            else:
                core_users = np.arange(self.n_users, dtype=np.int64)
                core_items = np.arange(self.n_items, dtype=np.int64)

            user_map = dict()
            for idx, user in enumerate(core_users):
                user_map[user] = idx
            item_map = dict()
            for idx, item in enumerate(core_items):
                item_map[item] = idx
        else:
            user_map = self.user_map
            item_map = self.item_map

        user_dim, item_dim = len(user_map), len(item_map)
        indices = []
        for user, item in dataset.train_array:
            if item in item_map:
                indices.append([user, user_dim + item_map[item]])
            if user in user_map:
                indices.append([self.n_users + item, user_map[user]])
        for user in range(self.n_users):
            indices.append([user, user_dim + item_dim])
        for item in range(self.n_items):
            indices.append([self.n_users + item, user_dim + item_dim + 1])
        feat = sp.coo_matrix((np.ones((len(indices),)), np.array(indices).T),
                             shape=(self.n_users + self.n_items, user_dim + item_dim + 2), dtype=np.float32).tocsr()
        row_sum = torch.tensor(np.array(np.sum(feat, axis=1)).squeeze(), dtype=torch.float32, device=self.device)
        feat = get_sparse_tensor(feat, self.device)
        return feat, user_map, item_map, row_sum

    def cal_cos_sim(self):
        # calculate cosine similarity with user embeddings and item embeddings (on CPU)

        rep = self.get_def_rep()
        all_users_r = rep[:self.n_users, :]
        all_items_r = rep[self.n_users:, :]

        all_user_r = all_users_r.to('cpu').detach().numpy().copy()
        all_item_r = all_items_r.to('cpu').detach().numpy().copy()

        x = cosine_similarity(all_user_r, all_item_r)

        del all_users_r, all_items_r
        del all_user_r, all_item_r
        torch.cuda.empty_cache()

        cos_mat = torch.from_numpy(x.astype(np.float32)).clone()
        cos_mat.to(device=self.device)
        cos_mat = torch.reshape(cos_mat, (1, -1))
        cos_mat = torch.squeeze(cos_mat)
        cos_mat1 = cos_mat[:len(cos_mat) // 2]
        _, idx1 = torch.topk(cos_mat1, self.aug_num // 2)
        cos_mat2 = cos_mat[len(cos_mat) // 2:]

        del cos_mat
        torch.cuda.empty_cache()
        # gc.collect()
        print(len(cos_mat2))
        _, idx2 = torch.topk(cos_mat2, self.aug_num // 2)
        # idx = idx.tolist()S
        aug_idx1 = [[int(torch.div(idx1[i], self.n_items, rounding_mode='floor')),
                     (int(torch.fmod(idx1[i], self.n_items)))] for i in range(self.aug_num // 2)]
        # aug_idx.to(device=self.device)
        aug_idx2 = [
            [int(torch.div(idx2[i] + self.aug_num // 2, self.n_items, rounding_mode='floor')),
             (int(torch.fmod(idx2[i] + self.aug_num // 2, self.n_items)))] for i
            in range(self.aug_num // 2)]
        aug_idx1.extend(aug_idx2)
        del aug_idx2, idx1, idx2
        torch.cuda.empty_cache()

        return aug_idx1  # return list [user_id, item_id]


    def cal_cos_simv2(self):
        # calculate cosine similarity with user embeddings and item embeddings (on CPU)

        rep = self.get_def_rep()
        all_users_r = rep[:self.n_users, :]
        all_items_r = rep[self.n_users:, :]
        all_items_r *= -1

        all_user_r = all_users_r.to('cpu').detach().numpy().copy()
        all_item_r = all_items_r.to('cpu').detach().numpy().copy()

        x = cosine_similarity(all_user_r, all_item_r)

        del all_users_r, all_items_r
        del all_user_r, all_item_r
        torch.cuda.empty_cache()

        cos_mat = torch.from_numpy(x.astype(np.float32)).clone()
        cos_mat.to(device=self.device)
        cos_mat = torch.reshape(cos_mat, (1, -1))
        cos_mat = torch.squeeze(cos_mat)
        cos_mat1 = cos_mat[:len(cos_mat) // 2]
        _, idx1 = torch.topk(cos_mat1, self.aug_num // 2)
        cos_mat2 = cos_mat[len(cos_mat) // 2:]

        del cos_mat
        torch.cuda.empty_cache()
        # gc.collect()
        print(len(cos_mat2))
        _, idx2 = torch.topk(cos_mat2, self.aug_num // 2)
        # idx = idx.tolist()S
        aug_idx1 = [[int(torch.div(idx1[i], self.n_items, rounding_mode='floor')),
                     (int(torch.fmod(idx1[i], self.n_items)))] for i in range(self.aug_num // 2)]
        # aug_idx.to(device=self.device)
        aug_idx2 = [
            [int(torch.div(idx2[i] + self.aug_num // 2, self.n_items, rounding_mode='floor')),
             (int(torch.fmod(idx2[i] + self.aug_num // 2, self.n_items)))] for i
            in range(self.aug_num // 2)]
        aug_idx1.extend(aug_idx2)
        del aug_idx2, idx1, idx2
        torch.cuda.empty_cache()

        return aug_idx1  # return list [user_id, item_id]

    def inductive_rep_layer(self, feat_mat):
        # generate embedding by using template
        padding_tensor = torch.empty([max(self.feat_mat.shape) - self.feat_mat.shape[1], self.embedding_size],
                                     dtype=torch.float32, device=self.device)
        padding_features = torch.cat([self.embedding.weight, padding_tensor], dim=0)

        row, column = feat_mat.indices()
        g = dgl.graph((column, row), num_nodes=max(self.feat_mat.shape), device=self.device)
        x = dgl.ops.gspmm(g, 'mul', 'sum', lhs_data=padding_features, rhs_data=feat_mat.values())
        x = x[:self.feat_mat.shape[0], :]
        return x

    def get_def_rep(self):
        # generate final embedding
        feat_mat = NGCF.dropout_sp_mat(self, self.feat_mat)
        representations = self.inductive_rep_layer(feat_mat)

        all_layer_rep = [representations]
        row, column = self.norm_adj.indices()
        g = dgl.graph((column, row), num_nodes=self.norm_adj.shape[0], device=self.device)
        for _ in range(self.n_layers):
            representations = dgl.ops.gspmm(g, 'mul', 'sum', lhs_data=representations, rhs_data=self.norm_adj.values())
            all_layer_rep.append(representations)
        all_layer_rep = torch.stack(all_layer_rep, dim=0)
        final_rep = all_layer_rep.mean(dim=0)
        return final_rep
    def get_aug_rep(self, norm_aug_adj):
        # generate final embedding
        feat_mat = NGCF.dropout_sp_mat(self, self.feat_mat)
        representations = self.inductive_rep_layer(feat_mat)

        all_layer_rep = [representations]
        row, column = norm_aug_adj.indices()
        g = dgl.graph((column, row), num_nodes=norm_aug_adj.shape[0], device=self.device)
        for _ in range(self.n_layers):
            representations = dgl.ops.gspmm(g, 'mul', 'sum', lhs_data=representations, rhs_data=norm_aug_adj.values())
            all_layer_rep.append(representations)
        all_layer_rep = torch.stack(all_layer_rep, dim=0)
        final_rep = all_layer_rep.mean(dim=0)
        return final_rep

    def get_drop_rep(self, norm_drop_adj):
        # generate final embedding on aug-graph
        feat_mat = NGCF.dropout_sp_mat(self, self.feat_mat)
        representations = self.inductive_rep_layer(feat_mat)

        all_layer_rep = [representations]
        row, column = norm_drop_adj.indices()
        g = dgl.graph((column, row), num_nodes=norm_drop_adj.shape[0], device=self.device)
        for _ in range(self.n_layers):
            representations = dgl.ops.gspmm(g, 'mul', 'sum', lhs_data=representations, rhs_data=norm_drop_adj.values())
            all_layer_rep.append(representations)
        all_layer_rep = torch.stack(all_layer_rep, dim=0)
        final_rep = all_layer_rep.mean(dim=0)
        return final_rep

    def cal_loss(self, users_r, aug_users_r):
        # calcrate ssl-loss(InfoNCE)
        loss = InfoNCE(negative_mode='unpaired')

        query = users_r
        positive_key = aug_users_r
        negative_keys = aug_users_r
        contrasive_loss = loss(query, positive_key, negative_keys)
        return contrasive_loss

    def bpr_forward(self, users, pos_items, neg_items):
        # 普通の埋め込み
        rep = self.get_def_rep()
        users_r = rep[users, :]
        pos_items_r, neg_items_r = rep[self.n_users + pos_items, :], rep[self.n_users + neg_items, :]
        # AUG後の埋め込み
        drop_rep = self.get_drop_rep(self.norm_aug_adj)
        drop_users_r = drop_rep[users, :]
        aug_rep = self.get_aug_rep(self.norm_aug_adj)
        aug_users_r = aug_rep[users, :]
        l2_norm_sq = torch.norm(users_r, p=2, dim=1) ** 2 + torch.norm(pos_items_r, p=2, dim=1) ** 2 \
                     + torch.norm(neg_items_r, p=2, dim=1) ** 2
        # Contrasive loss
        contrasive_loss1 = self.cal_loss(users_r, aug_users_r)
        contrasive_loss2 = self.cal_loss(users_r, drop_users_r)

        contrasive_loss = contrasive_loss1 + contrasive_loss2
        return users_r, pos_items_r, neg_items_r, l2_norm_sq, contrasive_loss

    def predict(self, users):
        rep = self.get_def_rep()
        users_r = rep[users, :]
        all_items_r = rep[self.n_users:, :]
        scores = torch.mm(users_r, all_items_r.t())
        return scores
    def update_aug_adj(self):
        del self.norm_aug_adj,  self.norm_drop_adj
        torch.cuda.empty_cache()
        # gc.collect()
        # self.norm_adj = self.generate_drop_graph()
        self.norm_aug_adj = self.generate_aug_graph(self.config['dataset'])
        self.norm_drop_adj = self.generate_drop_graph(self.config['dataset'])

    def save(self, path):
        params = {'sate_dict': self.state_dict(), 'user_map': self.user_map,
                  'item_map': self.item_map, 'alpha': self.alpha}
        torch.save(params, path)

    def load(self, path):
        params = torch.load(path, map_location=self.device)
        self.load_state_dict(params['sate_dict'])
        self.user_map = params['user_map']
        self.item_map = params['item_map']
        self.alpha = params['alpha']
        self.feat_mat, _, _, self.row_sum = self.generate_feat(self.config['dataset'], is_updating=True)
        self.update_feat_mat()



class DOSE_aug_drop2(BasicModel):
    def __init__(self, model_config):
        super(DOSE_aug_drop2, self).__init__(model_config)
        self.embedding_size = model_config['embedding_size']
        self.n_layers = model_config['n_layers']
        self.dropout = model_config['dropout']
        self.feature_ratio = model_config['feature_ratio']  # Template%
        self.norm_adj = self.generate_graph(model_config['dataset'])

        self.alpha = 1.
        self.delta = model_config.get('delta', 0.99)
        self.taugh = model_config.get('taugh', 0.2)
        self.aug_ratio = model_config.get('aug_ratio', 0.2)
        self.aug_num = model_config['aug_num']
        # self.temper = model_config['temper']
        self.feat_mat, self.user_map, self.item_map, self.row_sum = \
            self.generate_feat(model_config['dataset'],
                               ranking_metric=model_config.get('ranking_metric', 'sort'))
        self.update_feat_mat()
        # self.norm_aug_adj = generate_aug_graph(model_config['dataset'] )
        self.embedding = nn.Embedding(self.feat_mat.shape[1], self.embedding_size)

        self.w = nn.Parameter(torch.ones([self.embedding_size], dtype=torch.float32, device=self.device))
        normal_(self.embedding.weight, std=0.1)
        self.to(device=self.device)
        self.norm_aug_adj = self.generate_aug_graph(model_config['dataset'])
        self.norm_drop_adj = self.generate_drop_graph(model_config['dataset'])
        self.times = model_config.get('times', 0.1)
        # self.aug_num = (self.times) * len(self.feat_mat)

    def update_feat_mat(self):
        row, _ = self.feat_mat.indices()
        edge_values = torch.pow(self.row_sum[row], (self.alpha - 1.) / 2. - 0.5)
        self.feat_mat = torch.sparse.FloatTensor(self.feat_mat.indices(), edge_values, self.feat_mat.shape).coalesce()

    def feat_mat_anneal(self):
        self.alpha *= self.delta
        self.update_feat_mat()

    def generate_graph(self, dataset):
        return LightGCN.generate_graph(self, dataset)

    def generate_aug_graph(self, dataset):
        # new graph after adding some interactions
        aug_idx = self.cal_cos_sim(dataset)
        aug_adj_mat = generate_aug_daj_mat(dataset, aug_idx)
        degree = np.array(np.sum(aug_adj_mat, axis=1)).squeeze()
        degree = np.maximum(1., degree)
        d_inv = np.power(degree, -0.5) # 累乗
        d_mat = sp.diags(d_inv, format='csr', dtype=np.float32)

        norm_aug_adj = d_mat.dot(aug_adj_mat).dot(d_mat)
        norm_aug_adj = get_sparse_tensor(norm_aug_adj, self.device)
        return norm_aug_adj



    def generate_drop_graph(self, dataset):
        # new graph after adding some interactions
        aug_idx = self.cal_cos_sim(dataset)
        aug_adj_mat = generate_drop_daj_mat2(dataset, aug_idx)
        degree = np.array(np.sum(aug_adj_mat, axis=1)).squeeze()
        degree = np.maximum(1., degree)
        d_inv = np.power(degree, -0.5) # 累乗
        d_mat = sp.diags(d_inv, format='csr', dtype=np.float32)

        norm_aug_adj = d_mat.dot(aug_adj_mat).dot(d_mat)
        norm_aug_adj = get_sparse_tensor(norm_aug_adj, self.device)
        return norm_aug_adj

    def generate_feat(self, dataset, is_updating=False, ranking_metric=None):
        # return adj matrix with template
        if not is_updating:
            if self.feature_ratio < 1.:  # ランク付後にスライス

                ranked_users, ranked_items = graph_rank_nodes(dataset, ranking_metric)
                core_users = ranked_users[:int(self.n_users * self.feature_ratio)]
                core_items = ranked_items[:int(self.n_items * self.feature_ratio)]
            else:
                core_users = np.arange(self.n_users, dtype=np.int64)
                core_items = np.arange(self.n_items, dtype=np.int64)

            user_map = dict()
            for idx, user in enumerate(core_users):
                user_map[user] = idx
            item_map = dict()
            for idx, item in enumerate(core_items):
                item_map[item] = idx
        else:
            user_map = self.user_map
            item_map = self.item_map

        user_dim, item_dim = len(user_map), len(item_map)
        indices = []
        for user, item in dataset.train_array:
            if item in item_map:
                indices.append([user, user_dim + item_map[item]])
            if user in user_map:
                indices.append([self.n_users + item, user_map[user]])
        for user in range(self.n_users):
            indices.append([user, user_dim + item_dim])
        for item in range(self.n_items):
            indices.append([self.n_users + item, user_dim + item_dim + 1])
        feat = sp.coo_matrix((np.ones((len(indices),)), np.array(indices).T),
                             shape=(self.n_users + self.n_items, user_dim + item_dim + 2), dtype=np.float32).tocsr()
        row_sum = torch.tensor(np.array(np.sum(feat, axis=1)).squeeze(), dtype=torch.float32, device=self.device)
        feat = get_sparse_tensor(feat, self.device)
        return feat, user_map, item_map, row_sum

    def cal_cos_sim(self, dataset):
        # calculate cosine similarity with user embeddings and item embeddings (on CPU)
        ranking_metric = 'degree'
        ranked_users, ranked_items = graph_rank_nodes(dataset, ranking_metric)
        core_users = ranked_users[int(self.n_users * self.aug_ratio):]
        core_items = ranked_items[int(self.n_items * self.aug_ratio):]

        user_map = dict()
        item_map = dict()
        n_core_user = len(core_users)
        n_core_item = len(core_items)
        for idx, user in enumerate(core_users):
            user_map[idx] = user
        for idx, item in enumerate(core_items):
            item_map[idx] = item
        rep = self.get_def_rep()
        aug_users_r = rep[core_users, :]
        aug_items_r = rep[self.n_users + core_items, :]

        aug_users_r = aug_users_r.to('cpu').detach().numpy().copy()
        aug_items_r = aug_items_r.to('cpu').detach().numpy().copy()

        x = cosine_similarity(aug_users_r, aug_items_r)
        cos_mat = torch.from_numpy(x.astype(np.float32)).clone()
        cos_mat.to(device=self.device)
        cos_mat = torch.reshape(cos_mat, (1, -1))
        _, idx = torch.topk(cos_mat, self.aug_num)
        #print(user_map)
        #print(item_map)
        # idx = idx.tolist()S
        aug_idx = [
            [user_map[int(torch.div(idx[0][i], n_core_item, rounding_mode='floor'))], item_map[(int(torch.fmod(idx[0][i], n_core_item)))]]
            for i in range(self.aug_num)]

        return aug_idx  # return list [user_id, item_id]

    def inductive_rep_layer(self, feat_mat):
        # generate embedding by using template
        padding_tensor = torch.empty([max(self.feat_mat.shape) - self.feat_mat.shape[1], self.embedding_size],
                                     dtype=torch.float32, device=self.device)
        padding_features = torch.cat([self.embedding.weight, padding_tensor], dim=0)

        row, column = feat_mat.indices()
        g = dgl.graph((column, row), num_nodes=max(self.feat_mat.shape), device=self.device)
        x = dgl.ops.gspmm(g, 'mul', 'sum', lhs_data=padding_features, rhs_data=feat_mat.values())
        x = x[:self.feat_mat.shape[0], :]
        return x

    def get_def_rep(self):
        # generate final embedding
        feat_mat = NGCF.dropout_sp_mat(self, self.feat_mat)
        representations = self.inductive_rep_layer(feat_mat)

        all_layer_rep = [representations]
        row, column = self.norm_adj.indices()
        g = dgl.graph((column, row), num_nodes=self.norm_adj.shape[0], device=self.device)
        for _ in range(self.n_layers):
            representations = dgl.ops.gspmm(g, 'mul', 'sum', lhs_data=representations, rhs_data=self.norm_adj.values())
            all_layer_rep.append(representations)
        all_layer_rep = torch.stack(all_layer_rep, dim=0)
        final_rep = all_layer_rep.mean(dim=0)
        return final_rep
    
    def get_aug_rep(self, norm_aug_adj):
        # generate final embedding
        feat_mat = NGCF.dropout_sp_mat(self, self.feat_mat)
        representations = self.inductive_rep_layer(feat_mat)

        all_layer_rep = [representations]
        row, column = norm_aug_adj.indices()
        g = dgl.graph((column, row), num_nodes=norm_aug_adj.shape[0], device=self.device)
        for _ in range(self.n_layers):
            representations = dgl.ops.gspmm(g, 'mul', 'sum', lhs_data=representations, rhs_data=norm_aug_adj.values())
            all_layer_rep.append(representations)
        all_layer_rep = torch.stack(all_layer_rep, dim=0)
        final_rep = all_layer_rep.mean(dim=0)
        return final_rep

    def get_drop_rep(self, norm_drop_adj):
        # generate final embedding on aug-graph
        feat_mat = NGCF.dropout_sp_mat(self, self.feat_mat)
        representations = self.inductive_rep_layer(feat_mat)

        all_layer_rep = [representations]
        row, column = norm_drop_adj.indices()
        g = dgl.graph((column, row), num_nodes=norm_drop_adj.shape[0], device=self.device)
        for _ in range(self.n_layers):
            representations = dgl.ops.gspmm(g, 'mul', 'sum', lhs_data=representations, rhs_data=norm_drop_adj.values())
            all_layer_rep.append(representations)
        all_layer_rep = torch.stack(all_layer_rep, dim=0)
        final_rep = all_layer_rep.mean(dim=0)
        return final_rep

    def cal_loss(self, users_r, aug_users_r):
        # calcrate ssl-loss(InfoNCE)
        loss = InfoNCE(negative_mode='unpaired')

        query = users_r
        positive_key = aug_users_r
        negative_keys = aug_users_r
        contrasive_loss = loss(query, positive_key, negative_keys)
        return contrasive_loss

    def bpr_forward(self, users, pos_items, neg_items):
        # 普通の埋め込み
        rep = self.get_def_rep()
        users_r = rep[users, :]
        pos_items_r, neg_items_r = rep[self.n_users + pos_items, :], rep[self.n_users + neg_items, :]
        # AUG後の埋め込み

        aug_rep = self.get_drop_rep(self.norm_drop_adj)
        aug_users_r = aug_rep[users, :]
        l2_norm_sq = torch.norm(users_r, p=2, dim=1) ** 2 + torch.norm(pos_items_r, p=2, dim=1) ** 2 \
                     + torch.norm(neg_items_r, p=2, dim=1) ** 2
        # Contrasive loss
        contrasive_loss = self.cal_loss(users_r, aug_users_r)
        return users_r, pos_items_r, neg_items_r, l2_norm_sq, contrasive_loss

    def predict(self, users):
        rep = self.get_def_rep()
        users_r = rep[users, :]
        all_items_r = rep[self.n_users:, :]
        scores = torch.mm(users_r, all_items_r.t())
        return scores

    def save(self, path):
        params = {'sate_dict': self.state_dict(), 'user_map': self.user_map,
                  'item_map': self.item_map, 'alpha': self.alpha}
        torch.save(params, path)

    def load(self, path):
        params = torch.load(path, map_location=self.device)
        self.load_state_dict(params['sate_dict'])
        self.user_map = params['user_map']
        self.item_map = params['item_map']
        self.alpha = params['alpha']
        self.feat_mat, _, _, self.row_sum = self.generate_feat(self.config['dataset'], is_updating=True)
        self.update_feat_mat()


class DOSE_aug_drop3(BasicModel):
    def __init__(self, model_config):
        super(DOSE_aug_drop3, self).__init__(model_config)
        self.embedding_size = model_config['embedding_size']
        self.n_layers = model_config['n_layers']
        self.dropout = model_config['dropout']
        self.feature_ratio = model_config['feature_ratio']  # Template%
        self.norm_adj = self.generate_graph(model_config['dataset'])

        self.alpha = 1.
        self.delta = model_config.get('delta', 0.99)
        self.taugh = model_config.get('taugh', 0.2)
        self.aug_ratio = model_config.get('aug_ratio', 0.2)
        self.aug_num = model_config['aug_num']
        # self.temper = model_config['temper']
        self.feat_mat, self.user_map, self.item_map, self.row_sum = \
            self.generate_feat(model_config['dataset'],
                               ranking_metric=model_config.get('ranking_metric', 'sort'))
        self.update_feat_mat()
        # self.norm_aug_adj = generate_aug_graph(model_config['dataset'] )
        self.embedding = nn.Embedding(self.feat_mat.shape[1], self.embedding_size)

        self.w = nn.Parameter(torch.ones([self.embedding_size], dtype=torch.float32, device=self.device))
        normal_(self.embedding.weight, std=0.1)
        self.to(device=self.device)
        self.norm_aug_adj = self.generate_aug_graph(model_config['dataset'])
        self.norm_drop_adj = self.generate_drop_graph(model_config['dataset'])
        self.times = model_config.get('times', 0.1)
        # self.aug_num = (self.times) * len(self.feat_mat)

    def update_feat_mat(self):
        row, _ = self.feat_mat.indices()
        edge_values = torch.pow(self.row_sum[row], (self.alpha - 1.) / 2. - 0.5)
        self.feat_mat = torch.sparse.FloatTensor(self.feat_mat.indices(), edge_values, self.feat_mat.shape).coalesce()

    def feat_mat_anneal(self):
        self.alpha *= self.delta
        self.update_feat_mat()

    def generate_graph(self, dataset):
        return LightGCN.generate_graph(self, dataset)

    def generate_aug_graph(self, dataset):
        # new graph after adding some interactions
        aug_idx = self.cal_cos_sim()
        aug_adj_mat = generate_aug_daj_mat(dataset, aug_idx)
        degree = np.array(np.sum(aug_adj_mat, axis=1)).squeeze()
        degree = np.maximum(1., degree)
        d_inv = np.power(degree, -0.5)  # 累乗
        d_mat = sp.diags(d_inv, format='csr', dtype=np.float32)

        norm_aug_adj = d_mat.dot(aug_adj_mat).dot(d_mat)
        norm_aug_adj = get_sparse_tensor(norm_aug_adj, self.device)
        return norm_aug_adj

    def generate_drop_graph(self, dataset):
        # new graph after adding some interactions
        aug_idx = self.cal_cos_sim()
        aug_adj_mat = generate_drop_daj_mat3(dataset, aug_idx)
        degree = np.array(np.sum(aug_adj_mat, axis=1)).squeeze()
        degree = np.maximum(1., degree)
        d_inv = np.power(degree, -0.5)  # 累乗
        d_mat = sp.diags(d_inv, format='csr', dtype=np.float32)

        norm_aug_adj = d_mat.dot(aug_adj_mat).dot(d_mat)
        norm_aug_adj = get_sparse_tensor(norm_aug_adj, self.device)
        return norm_aug_adj

    def generate_feat(self, dataset, is_updating=False, ranking_metric=None):
        # return adj matrix with template
        if not is_updating:
            if self.feature_ratio < 1.:  # ランク付後にスライス

                ranked_users, ranked_items = graph_rank_nodes(dataset, ranking_metric)
                core_users = ranked_users[:int(self.n_users * self.feature_ratio)]
                core_items = ranked_items[:int(self.n_items * self.feature_ratio)]
            else:
                core_users = np.arange(self.n_users, dtype=np.int64)
                core_items = np.arange(self.n_items, dtype=np.int64)

            user_map = dict()
            for idx, user in enumerate(core_users):
                user_map[user] = idx
            item_map = dict()
            for idx, item in enumerate(core_items):
                item_map[item] = idx
        else:
            user_map = self.user_map
            item_map = self.item_map

        user_dim, item_dim = len(user_map), len(item_map)
        indices = []
        for user, item in dataset.train_array:
            if item in item_map:
                indices.append([user, user_dim + item_map[item]])
            if user in user_map:
                indices.append([self.n_users + item, user_map[user]])
        for user in range(self.n_users):
            indices.append([user, user_dim + item_dim])
        for item in range(self.n_items):
            indices.append([self.n_users + item, user_dim + item_dim + 1])
        feat = sp.coo_matrix((np.ones((len(indices),)), np.array(indices).T),
                             shape=(self.n_users + self.n_items, user_dim + item_dim + 2), dtype=np.float32).tocsr()
        row_sum = torch.tensor(np.array(np.sum(feat, axis=1)).squeeze(), dtype=torch.float32, device=self.device)
        feat = get_sparse_tensor(feat, self.device)
        return feat, user_map, item_map, row_sum

    def cal_cos_sim(self):
        # calculate cosine similarity with user embeddings and item embeddings (on CPU)
        rep = self.get_def_rep()
        all_users_r = rep[:self.n_users, :]
        all_items_r = rep[self.n_users:, :]

        all_users_r = all_users_r.to('cpu').detach().numpy().copy()
        all_items_r = all_items_r.to('cpu').detach().numpy().copy()

        x = cosine_similarity(all_users_r, all_items_r)
        cos_mat = torch.from_numpy(x.astype(np.float32)).clone()
        cos_mat.to(device=self.device)
        cos_sim = torch.reshape(cos_mat, (1, -1))
        _, idx = torch.topk(cos_sim, self.aug_num)
        # idx = idx.tolist()S
        aug_idx = [
            [int(torch.div(idx[0][i], self.n_items, rounding_mode='floor')), (int(torch.fmod(idx[0][i], self.n_items)))]
            for i in range(self.aug_num)]

        return aug_idx  # return list [user_id, item_id]

    def inductive_rep_layer(self, feat_mat):
        # generate embedding by using template
        padding_tensor = torch.empty([max(self.feat_mat.shape) - self.feat_mat.shape[1], self.embedding_size],
                                     dtype=torch.float32, device=self.device)
        padding_features = torch.cat([self.embedding.weight, padding_tensor], dim=0)

        row, column = feat_mat.indices()
        g = dgl.graph((column, row), num_nodes=max(self.feat_mat.shape), device=self.device)
        x = dgl.ops.gspmm(g, 'mul', 'sum', lhs_data=padding_features, rhs_data=feat_mat.values())
        x = x[:self.feat_mat.shape[0], :]
        return x

    def get_def_rep(self):
        # generate final embedding
        feat_mat = NGCF.dropout_sp_mat(self, self.feat_mat)
        representations = self.inductive_rep_layer(feat_mat)

        all_layer_rep = [representations]
        row, column = self.norm_adj.indices()
        g = dgl.graph((column, row), num_nodes=self.norm_adj.shape[0], device=self.device)
        for _ in range(self.n_layers):
            representations = dgl.ops.gspmm(g, 'mul', 'sum', lhs_data=representations, rhs_data=self.norm_adj.values())
            all_layer_rep.append(representations)
        all_layer_rep = torch.stack(all_layer_rep, dim=0)
        final_rep = all_layer_rep.mean(dim=0)
        return final_rep

    def get_aug_rep(self, norm_aug_adj):
        # generate final embedding
        feat_mat = NGCF.dropout_sp_mat(self, self.feat_mat)
        representations = self.inductive_rep_layer(feat_mat)

        all_layer_rep = [representations]
        row, column = norm_aug_adj.indices()
        g = dgl.graph((column, row), num_nodes=norm_aug_adj.shape[0], device=self.device)
        for _ in range(self.n_layers):
            representations = dgl.ops.gspmm(g, 'mul', 'sum', lhs_data=representations, rhs_data=norm_aug_adj.values())
            all_layer_rep.append(representations)
        all_layer_rep = torch.stack(all_layer_rep, dim=0)
        final_rep = all_layer_rep.mean(dim=0)
        return final_rep

    def get_drop_rep(self, norm_drop_adj):
        # generate final embedding on aug-graph
        feat_mat = NGCF.dropout_sp_mat(self, self.feat_mat)
        representations = self.inductive_rep_layer(feat_mat)

        all_layer_rep = [representations]
        row, column = norm_drop_adj.indices()
        g = dgl.graph((column, row), num_nodes=norm_drop_adj.shape[0], device=self.device)
        for _ in range(self.n_layers):
            representations = dgl.ops.gspmm(g, 'mul', 'sum', lhs_data=representations, rhs_data=norm_drop_adj.values())
            all_layer_rep.append(representations)
        all_layer_rep = torch.stack(all_layer_rep, dim=0)
        final_rep = all_layer_rep.mean(dim=0)
        return final_rep

    def cal_loss(self, users_r, aug_users_r):
        # calcrate ssl-loss(InfoNCE)
        loss = InfoNCE(negative_mode='unpaired')

        query = users_r
        positive_key = aug_users_r
        negative_keys = aug_users_r
        contrasive_loss = loss(query, positive_key, negative_keys)
        return contrasive_loss

    def bpr_forward(self, users, pos_items, neg_items):
        # 普通の埋め込み
        rep = self.get_def_rep()
        users_r = rep[users, :]
        pos_items_r, neg_items_r = rep[self.n_users + pos_items, :], rep[self.n_users + neg_items, :]
        # AUG後の埋め込み

        aug_rep = self.get_drop_rep(self.norm_drop_adj)
        aug_users_r = aug_rep[users, :]
        l2_norm_sq = torch.norm(users_r, p=2, dim=1) ** 2 + torch.norm(pos_items_r, p=2, dim=1) ** 2 \
                     + torch.norm(neg_items_r, p=2, dim=1) ** 2
        # Contrasive loss
        contrasive_loss = self.cal_loss(users_r, aug_users_r)
        return users_r, pos_items_r, neg_items_r, l2_norm_sq, contrasive_loss

    def predict(self, users):
        rep = self.get_def_rep()
        users_r = rep[users, :]
        all_items_r = rep[self.n_users:, :]
        scores = torch.mm(users_r, all_items_r.t())
        return scores

    def save(self, path):
        params = {'sate_dict': self.state_dict(), 'user_map': self.user_map,
                  'item_map': self.item_map, 'alpha': self.alpha}
        torch.save(params, path)

    def load(self, path):
        params = torch.load(path, map_location=self.device)
        self.load_state_dict(params['sate_dict'])
        self.user_map = params['user_map']
        self.item_map = params['item_map']
        self.alpha = params['alpha']
        self.feat_mat, _, _, self.row_sum = self.generate_feat(self.config['dataset'], is_updating=True)
        self.update_feat_mat()


class DOSE_test(BasicModel):
    def __init__(self, model_config):
        super(DOSE_test, self).__init__(model_config)
        self.embedding_size = model_config['embedding_size']
        self.n_layers = model_config['n_layers']
        self.dropout = model_config['dropout']
        self.feature_ratio = model_config['feature_ratio']  # Template%
        self.norm_adj = self.generate_graph(model_config['dataset'])

        self.alpha = 1.
        self.delta = model_config.get('delta', 0.99)
        self.taugh = model_config.get('taugh', 0.2)
        self.aug_num = model_config['aug_num']
        # self.temper = model_config['temper']
        self.feat_mat, self.user_map, self.item_map, self.row_sum = \
            self.generate_feat(model_config['dataset'],
                               ranking_metric=model_config.get('ranking_metric', 'sort'))
        self.update_feat_mat()
        # self.norm_aug_adj = enerate_aug_graph(model_config['dataset'] )
        self.embedding = nn.Embedding(self.feat_mat.shape[1], self.embedding_size)

        self.w = nn.Parameter(torch.ones([self.embedding_size], dtype=torch.float32, device=self.device))
        normal_(self.embedding.weight, std=0.1)
        self.to(device=self.device)
        self.norm_aug_adj = self.generate_aug_graph(model_config['dataset'])
        self.times = model_config.get('times', 0.1)
        # self.aug_num = (self.times) * len(self.feat_mat)

    def update_feat_mat(self):
        row, _ = self.feat_mat.indices()
        edge_values = torch.pow(self.row_sum[row], (self.alpha - 1.) / 2. - 0.5)
        self.feat_mat = torch.sparse.FloatTensor(self.feat_mat.indices(), edge_values, self.feat_mat.shape).coalesce()

    def feat_mat_anneal(self):
        self.alpha *= self.delta
        self.update_feat_mat()

    def generate_graph(self, dataset):
        return LightGCN.generate_graph(self, dataset)

    def generate_aug_graph(self, dataset):
        # new graph after adding some interactions
        aug_idx = self.cal_cos_sim()
        aug_adj_mat = generate_aug_daj_mat(dataset, aug_idx)
        degree = np.array(np.sum(aug_adj_mat, axis=1)).squeeze()
        degree = np.maximum(1., degree)
        d_inv = np.power(degree, -0.5)  # 累乗
        d_mat = sp.diags(d_inv, format='csr', dtype=np.float32)

        norm_aug_adj = d_mat.dot(aug_adj_mat).dot(d_mat)
        norm_aug_adj = get_sparse_tensor(norm_aug_adj, self.device)
        return norm_aug_adj

    def generate_feat(self, dataset, is_updating=False, ranking_metric=None):
        # return adj matrix with template
        if not is_updating:
            if self.feature_ratio < 1.:  # ランク付後にスライス

                ranked_users, ranked_items = graph_rank_nodes(dataset, ranking_metric)
                core_users = ranked_users[:int(self.n_users * self.feature_ratio)]
                core_items = ranked_items[:int(self.n_items * self.feature_ratio)]
            else:
                core_users = np.arange(self.n_users, dtype=np.int64)
                core_items = np.arange(self.n_items, dtype=np.int64)

            user_map = dict()
            for idx, user in enumerate(core_users):
                user_map[user] = idx
            item_map = dict()
            for idx, item in enumerate(core_items):
                item_map[item] = idx
        else:
            user_map = self.user_map
            item_map = self.item_map

        user_dim, item_dim = len(user_map), len(item_map)
        indices = []
        for user, item in dataset.train_array:
            if item in item_map:
                indices.append([user, user_dim + item_map[item]])
            if user in user_map:
                indices.append([self.n_users + item, user_map[user]])
        for user in range(self.n_users):
            indices.append([user, user_dim + item_dim])
        for item in range(self.n_items):
            indices.append([self.n_users + item, user_dim + item_dim + 1])
        feat = sp.coo_matrix((np.ones((len(indices),)), np.array(indices).T),
                             shape=(self.n_users + self.n_items, user_dim + item_dim + 2), dtype=np.float32).tocsr()
        row_sum = torch.tensor(np.array(np.sum(feat, axis=1)).squeeze(), dtype=torch.float32, device=self.device)
        feat = get_sparse_tensor(feat, self.device)
        return feat, user_map, item_map, row_sum

    def inductive_rep_layer(self, feat_mat):
        padding_tensor = torch.empty([max(self.feat_mat.shape) - self.feat_mat.shape[1], self.embedding_size],
                                     dtype=torch.float32, device=self.device)
        padding_features = torch.cat([self.embedding.weight, padding_tensor], dim=0)

        row, column = feat_mat.indices()
        g = dgl.graph((column, row), num_nodes=max(self.feat_mat.shape), device=self.device)
        x = dgl.ops.gspmm(g, 'mul', 'sum', lhs_data=padding_features, rhs_data=feat_mat.values())
        x = x[:self.feat_mat.shape[0], :]
        return x

    def get_def_rep(self):
        # generate final embedding
        feat_mat = NGCF.dropout_sp_mat(self, self.feat_mat)
        representations = self.inductive_rep_layer(feat_mat)

        all_layer_rep = [representations]
        row, column = self.norm_adj.indices()
        g = dgl.graph((column, row), num_nodes=self.norm_adj.shape[0], device=self.device)
        for _ in range(self.n_layers):
            representations = dgl.ops.gspmm(g, 'mul', 'sum', lhs_data=representations, rhs_data=self.norm_adj.values())
            all_layer_rep.append(representations)
        all_layer_rep = torch.stack(all_layer_rep, dim=0)
        final_rep = all_layer_rep.mean(dim=0)
        return final_rep

    def get_aug_rep(self, norm_aug_adj):
        # generate final embedding on aug-graph
        feat_mat = NGCF.dropout_sp_mat(self, self.feat_mat)
        representations = self.inductive_rep_layer(feat_mat)

        all_layer_rep = [representations]
        row, column = norm_aug_adj.indices()
        g = dgl.graph((column, row), num_nodes=norm_aug_adj.shape[0], device=self.device)
        for _ in range(self.n_layers):
            representations = dgl.ops.gspmm(g, 'mul', 'sum', lhs_data=representations, rhs_data=norm_aug_adj.values())
            all_layer_rep.append(representations)
        all_layer_rep = torch.stack(all_layer_rep, dim=0)
        final_rep = all_layer_rep.mean(dim=0)
        return final_rep

    def cal_cos_sim(self):
        # calculate cosine similarity with user embeddings and item embeddings (on CPU)
        rep = self.get_def_rep()
        all_users_r = rep[:self.n_users, :]
        all_items_r = rep[self.n_users:, :]

        all_users_r = all_users_r.to('cpu').detach().numpy().copy()
        all_items_r = all_items_r.to('cpu').detach().numpy().copy()

        x = cosine_similarity(all_users_r, all_items_r)
        cos_mat = torch.from_numpy(x.astype(np.float32)).clone()
        cos_mat.to(device=self.device)
        cos_sim = torch.reshape(cos_mat, (1, -1))
        _, idx = torch.topk(cos_sim, self.aug_num)
        # idx = idx.tolist()S
        aug_idx = [
            [int(torch.div(idx[0][i], self.n_items, rounding_mode='floor')), (int(torch.fmod(idx[0][i], self.n_items)))]
            for i in range(self.aug_num)]

        return aug_idx  # return list [user_id, item_id]

    def cal_cos_sim_v2(self):
        # calculate cosine similarity with user embeddings and item embeddings (on GPU)
        rep = self.get_def_rep()
        all_users_r = rep[:self.n_users, :]
        all_items_r = rep[self.n_users:, :]

        cos_sim = pairwise_cosine_similarity(all_users_r, all_items_r)
        cos_sim = torch.reshape(cos_sim, (1, -1))
        _, idx = torch.topk(cos_sim, self.aug_num)
        aug_idx = [
            [int(torch.div(idx[0][i], self.n_items, rounding_mode='floor')), (int(torch.fmod(idx[0][i], self.n_items)))]
            for i in range(self.aug_num)]

        return aug_idx  # return list [user_id, item_id]

    def cal_loss(self, users_r, aug_users_r):
        # calcrate ssl-loss(InfoNCE)
        loss = InfoNCE(negative_mode='paired')

        query = users_r
        positive_key = aug_users_r
        negative_keys = aug_users_r.unsqueeze(1)

        contrasive_loss = loss(query, positive_key, negative_keys)
        return contrasive_loss

    def bpr_forward(self, users, pos_items, neg_items):
        # 普通の埋め込み
        rep = self.get_def_rep()
        users_r = rep[users, :]
        aug_rep = self.get_aug_rep(self.norm_aug_adj)
        aug_users_r = aug_rep[users, :]
        pos_items_r, neg_items_r = rep[self.n_users + pos_items, :], rep[self.n_users + neg_items, :]
        # AUG後の埋め込み

        l2_norm_sq = torch.norm(users_r, p=2, dim=1) ** 2 + torch.norm(pos_items_r, p=2, dim=1) ** 2 \
                     + torch.norm(neg_items_r, p=2, dim=1) ** 2

        return users_r, pos_items_r, neg_items_r, l2_norm_sq, aug_users_r

    def predict(self, users):
        rep = self.get_def_rep()
        users_r = rep[users, :]
        all_items_r = rep[self.n_users:, :]
        scores = torch.mm(users_r, all_items_r.t())
        return scores

    def save(self, path):
        params = {'sate_dict': self.state_dict(), 'user_map': self.user_map,
                  'item_map': self.item_map, 'alpha': self.alpha}
        torch.save(params, path)

    def load(self, path):
        params = torch.load(path, map_location=self.device)
        self.load_state_dict(params['sate_dict'])
        self.user_map = params['user_map']
        self.item_map = params['item_map']
        self.alpha = params['alpha']
        self.feat_mat, _, _, self.row_sum = self.generate_feat(self.config['dataset'], is_updating=True)
        self.update_feat_mat()


class RelationGAT(nn.Module):
    def __init__(self, in_size, out_size):
        super(RelationGAT, self).__init__()
        self.wq = init_one_layer(in_size, out_size)
        self.wk = init_one_layer(in_size, out_size)
        self.wv = init_one_layer(in_size, out_size)

    def forward(self, x, neighbors):
        x = self.wq(x).unsqueeze(1)
        key = self.wk(neighbors).unsqueeze(0)
        gat_input = torch.sum(x * key, dim=2)
        attn = F.softmax(gat_input, dim=1)
        gat_output = self.wv(torch.matmul(attn, neighbors))
        return gat_output


class IDCF_LGCN(BasicModel):
    def __init__(self, model_config):
        super(IDCF_LGCN, self).__init__(model_config)
        self.embedding_size = model_config['embedding_size']
        self.n_layers = model_config['n_layers']
        self.n_headers = model_config['n_headers']
        self.n_samples = model_config.get('n_samples', 50)
        self.n_old_users = self.n_users
        self.n_old_items = self.n_items
        self.embedding = nn.Embedding(self.n_users + self.n_items, self.embedding_size)
        with torch.no_grad():
            lgcn_path = model_config['lgcn_path']
            self.embedding.weight.data = torch.load(lgcn_path, map_location=self.device)['embedding.weight']
            self.embedding.weight.requires_grad = False
        self.gat_units = []
        for _ in range(self.n_headers):
            self.gat_units.append(RelationGAT(self.embedding_size, self.embedding_size))
        self.gat_units = nn.ModuleList(self.gat_units)
        self.w_out = init_one_layer(self.embedding_size * self.n_headers, self.embedding_size)
        self.norm_adj = self.generate_graph(model_config['dataset'])
        self.feat_mat = self.generate_feat(model_config['dataset'])
        self.to(device=self.device)

    def generate_graph(self, dataset):
        return LightGCN.generate_graph(self, dataset)

    def generate_feat(self, dataset):
        feat_mat = generate_daj_mat(dataset)
        feat_mat = sp.hstack([feat_mat[:, :self.n_old_users], feat_mat[:, self.n_users:self.n_users + self.n_old_items]])
        feat_mat = get_sparse_tensor(feat_mat, self.device)
        return feat_mat

    def get_rep(self, contrastive=False):
        padding_tensor = torch.empty([self.feat_mat.shape[0] - self.feat_mat.shape[1], self.embedding_size],
                                     dtype=torch.float32, device=self.device)
        padding_features = torch.cat([self.embedding.weight, padding_tensor], dim=0)

        row, column = self.feat_mat.indices()
        g = dgl.graph((column, row), num_nodes=self.feat_mat.shape[0], device=self.device)
        x_q = dgl.ops.gspmm(g, 'mul', 'sum', lhs_data=padding_features, rhs_data=self.feat_mat.values())
        gat_outputs = []
        for i in range(self.n_headers):
            sampled_users = np.random.randint(0, self.n_old_users, size=self.n_samples)
            sampled_items = np.random.randint(0, self.n_old_items, size=self.n_samples)
            sampled_user_embs = self.embedding.weight[sampled_users]
            sampled_item_embs = self.embedding.weight[self.n_old_users + sampled_items]
            user_reps = self.gat_units[i](x_q[:self.n_users], sampled_user_embs)
            item_reps = self.gat_units[i](x_q[self.n_users:], sampled_item_embs)
            gat_outputs.append(torch.cat([user_reps, item_reps], dim=0))
        gat_outputs = torch.cat(gat_outputs, dim=1)
        representations = self.w_out(gat_outputs)
        if contrastive:
            user_similarities = torch.sum(representations[:self.n_users].unsqueeze(1) * sampled_user_embs.unsqueeze(0),
                                          dim=2)
            user_self = torch.sum(representations[:self.n_users] * self.embedding.weight[:self.n_old_users], dim=1)
            user_loss = torch.logsumexp(user_similarities, dim=1) - user_self
            item_similarities = torch.sum(representations[self.n_users:].unsqueeze(1) * sampled_item_embs.unsqueeze(0),
                                          dim=2)
            item_self = torch.sum(representations[self.n_users:] * self.embedding.weight[self.n_old_users:], dim=1)
            item_loss = torch.logsumexp(item_similarities, dim=1) - item_self
            contrastive_loss = torch.cat([user_loss, item_loss], dim=0)

        all_layer_rep = [representations]
        row, column = self.norm_adj.indices()
        g = dgl.graph((column, row), num_nodes=self.norm_adj.shape[0], device=self.device)
        for _ in range(self.n_layers):
            representations = dgl.ops.gspmm(g, 'mul', 'sum', lhs_data=representations, rhs_data=self.norm_adj.values())
            all_layer_rep.append(representations)
        all_layer_rep = torch.stack(all_layer_rep, dim=0)
        final_rep = all_layer_rep.mean(dim=0)
        if contrastive:
            return final_rep, contrastive_loss
        return final_rep

    def bpr_forward(self, users, pos_items, neg_items):
        rep, contrastive_loss = self.get_rep(contrastive=True)
        contrastive_loss = contrastive_loss[users] + contrastive_loss[self.n_users + pos_items] \
                           + contrastive_loss[self.n_users + neg_items]
        users_r = rep[users, :]
        pos_items_r, neg_items_r = rep[self.n_users + pos_items, :], rep[self.n_users + neg_items, :]
        l2_norm_sq = torch.norm(users_r, p=2, dim=1) ** 2 + torch.norm(pos_items_r, p=2, dim=1) ** 2 \
                     + torch.norm(neg_items_r, p=2, dim=1) ** 2
        for i in range(self.n_headers):
            l2_norm_sq += torch.norm(self.gat_units[i].wq.weight, p=2) ** 2
            l2_norm_sq += torch.norm(self.gat_units[i].wk.weight, p=2) ** 2
        return users_r, pos_items_r, neg_items_r, l2_norm_sq, contrastive_loss

    def predict(self, users):
        return LightGCN.predict(self, users)


class NGCF(BasicModel):
    def __init__(self, model_config):
        super(NGCF, self).__init__(model_config)
        self.dropout = model_config['dropout']
        self.embedding_size = model_config['embedding_size']
        self.layer_sizes = model_config['layer_sizes'].copy()
        self.embedding = nn.Embedding(self.n_users + self.n_items, self.embedding_size)
        kaiming_uniform_(self.embedding.weight)
        self.n_layers = len(self.layer_sizes)
        self.layer_sizes.insert(0, self.embedding_size)
        self.gc_layers = []
        self.bi_layers = []
        for layer_idx in range(1, self.n_layers + 1):
            dense_layer = init_one_layer(self.layer_sizes[layer_idx - 1], self.layer_sizes[layer_idx])
            self.gc_layers.append(dense_layer)
            dense_layer = init_one_layer(self.layer_sizes[layer_idx - 1], self.layer_sizes[layer_idx])
            self.bi_layers.append(dense_layer)
        self.gc_layers = nn.ModuleList(self.gc_layers)
        self.bi_layers = nn.ModuleList(self.bi_layers)
        self.norm_adj = self.generate_graph(model_config['dataset'])
        self.to(device=self.device)

    def generate_graph(self, dataset):
        adj_mat = generate_daj_mat(dataset)
        adj_mat = adj_mat + sp.eye(adj_mat.shape[0], format='csr')

        norm_adj = normalize(adj_mat, axis=1, norm='l1')
        norm_adj = get_sparse_tensor(norm_adj, self.device)
        return norm_adj

    def dropout_sp_mat(self, mat):
        if not self.training:
            return mat
        random_tensor = 1 - self.dropout
        random_tensor += torch.rand(mat._nnz()).to(self.device)
        dropout_mask = torch.floor(random_tensor).type(torch.bool)
        i = mat.indices()
        v = mat.values()

        i = i[:, dropout_mask]
        v = v[dropout_mask] / (1. - self.dropout)
        out = torch.sparse.FloatTensor(i, v, mat.shape).coalesce()
        return out

    def get_rep(self):
        representations = self.embedding.weight
        all_layer_rep = [representations]
        dropped_adj = self.dropout_sp_mat(self.norm_adj)
        row, column = dropped_adj.indices()
        g = dgl.graph((column, row), num_nodes=dropped_adj.shape[0], device=self.device)
        for layer_idx in range(self.n_layers):
            messages_0 = dgl.ops.gspmm(g, 'mul', 'sum', lhs_data=representations, rhs_data=dropped_adj.values())
            messages_1 = torch.mul(representations, messages_0)
            messages_0, messages_1 = self.gc_layers[layer_idx](messages_0), self.bi_layers[layer_idx](messages_1)
            representations = F.leaky_relu(messages_0 + messages_1, negative_slope=0.2)
            representations = F.dropout(representations, p=self.dropout, training=self.training)
            all_layer_rep.append(F.normalize(representations, p=2, dim=1))
        final_rep = torch.cat(all_layer_rep, dim=1)
        return final_rep

    def bpr_forward(self, users, pos_items, neg_items):
        rep = self.get_rep()
        users_r = rep[users, :]
        pos_items_r, neg_items_r = rep[self.n_users + pos_items, :], rep[self.n_users + neg_items, :]
        l2_norm_sq = torch.norm(users_r, p=2, dim=1) ** 2 + torch.norm(pos_items_r, p=2, dim=1) ** 2 \
                     + torch.norm(neg_items_r, p=2, dim=1) ** 2
        return users_r, pos_items_r, neg_items_r, l2_norm_sq

    def predict(self, users):
        return LightGCN.predict(self, users)


class ItemKNN(BasicModel):
    def __init__(self, model_config):
        super(ItemKNN, self).__init__(model_config)
        self.k = model_config['k']
        self.data_mat, self.sim_mat = self.calculate_similarity(model_config['dataset'])
        self.trainable = False

    def calculate_similarity(self, dataset):
        data_mat = sp.coo_matrix((np.ones((len(dataset.train_array),)), np.array(dataset.train_array).T),
                                 shape=(self.n_users, self.n_items), dtype=np.float32).tocsr()
        item_degree = np.array(np.sum(data_mat, axis=0)).squeeze()
        row = []
        col = []
        data = []
        for item in range(dataset.n_items):
            intersections = data_mat.T.dot(data_mat[:, item]).toarray().squeeze()
            with np.errstate(invalid='ignore'):
                sims = intersections / (item_degree + item_degree[item] - intersections)
            sims[np.isnan(sims)] = 0.
            row.extend([item] * self.k)
            topk_items = np.argsort(sims)[-self.k:]
            col.extend(topk_items.tolist())
            data.extend(sims[topk_items].tolist())
        sim_mat = sp.coo_matrix((data, (row, col)), shape=(self.n_items, self.n_items), dtype=np.float32).tocsr()
        return data_mat, sim_mat

    def predict(self, users):
        users = users.cpu().numpy()
        profiles = self.data_mat[users, :]
        scores = torch.tensor(profiles.dot(self.sim_mat).toarray(), dtype=torch.float32, device=self.device)
        return scores


class Popularity(BasicModel):
    def __init__(self, model_config):
        super(Popularity, self).__init__(model_config)
        self.item_degree = self.calculate_degree(model_config['dataset'])
        self.trainable = False

    def calculate_degree(self, dataset):
        data_mat = sp.coo_matrix((np.ones((len(dataset.train_array),)), np.array(dataset.train_array).T),
                                 shape=(self.n_users, self.n_items), dtype=np.float32).tocsr()
        item_degree = np.array(np.sum(data_mat, axis=0)).squeeze()
        return torch.tensor(item_degree, dtype=torch.float32, device=self.device)

    def predict(self, users):
        return self.item_degree[None, :].repeat(users.shape[0], 1)


class IGCN(BasicModel):
    def __init__(self, model_config):
        super(IGCN, self).__init__(model_config)
        self.embedding_size = model_config['embedding_size']
        self.n_layers = model_config['n_layers']
        self.dropout = model_config['dropout']
        self.feature_ratio = model_config['feature_ratio']  # Template%
        self.norm_adj = self.generate_graph(model_config['dataset'])
        self.alpha = 1.
        self.delta = model_config.get('delta', 0.99)
        self.feat_mat, self.user_map, self.item_map, self.row_sum = \
            self.generate_feat(model_config['dataset'],
                               ranking_metric=model_config.get('ranking_metric', 'sort'))
        self.update_feat_mat()

        self.embedding = nn.Embedding(self.feat_mat.shape[1], self.embedding_size)
        self.w = nn.Parameter(torch.ones([self.embedding_size], dtype=torch.float32, device=self.device))
        normal_(self.embedding.weight, std=0.1)
        self.to(device=self.device)

    def update_feat_mat(self):
        row, _ = self.feat_mat.indices()
        edge_values = torch.pow(self.row_sum[row], (self.alpha - 1.) / 2. - 0.5)
        self.feat_mat = torch.sparse.FloatTensor(self.feat_mat.indices(), edge_values, self.feat_mat.shape).coalesce()

    def feat_mat_anneal(self):
        self.alpha *= self.delta
        self.update_feat_mat()

    def generate_graph(self, dataset):
        return LightGCN.generate_graph(self, dataset)

    def generate_feat(self, dataset, is_updating=False, ranking_metric=None):
        if not is_updating:
            if self.feature_ratio < 1.:  # ランク付後にスライス

                ranked_users, ranked_items = graph_rank_nodes(dataset, ranking_metric)
                core_users = ranked_users[:int(self.n_users * self.feature_ratio)]
                core_items = ranked_items[:int(self.n_items * self.feature_ratio)]
            else:
                core_users = np.arange(self.n_users, dtype=np.int64)
                core_items = np.arange(self.n_items, dtype=np.int64)

            user_map = dict()
            for idx, user in enumerate(core_users):
                user_map[user] = idx
            item_map = dict()
            for idx, item in enumerate(core_items):
                item_map[item] = idx
        else:
            user_map = self.user_map
            item_map = self.item_map

        user_dim, item_dim = len(user_map), len(item_map)
        indices = []
        for user, item in dataset.train_array:
            if item in item_map:
                indices.append([user, user_dim + item_map[item]])
            if user in user_map:
                indices.append([self.n_users + item, user_map[user]])
        for user in range(self.n_users):
            indices.append([user, user_dim + item_dim])
        for item in range(self.n_items):
            indices.append([self.n_users + item, user_dim + item_dim + 1])
        feat = sp.coo_matrix((np.ones((len(indices),)), np.array(indices).T),
                             shape=(self.n_users + self.n_items, user_dim + item_dim + 2), dtype=np.float32).tocsr()
        row_sum = torch.tensor(np.array(np.sum(feat, axis=1)).squeeze(), dtype=torch.float32, device=self.device)
        feat = get_sparse_tensor(feat, self.device)
        return feat, user_map, item_map, row_sum

    def inductive_rep_layer(self, feat_mat):
        padding_tensor = torch.empty([max(self.feat_mat.shape) - self.feat_mat.shape[1], self.embedding_size],
                                     dtype=torch.float32, device=self.device)
        padding_features = torch.cat([self.embedding.weight, padding_tensor], dim=0)

        row, column = feat_mat.indices()
        g = dgl.graph((column, row), num_nodes=max(self.feat_mat.shape), device=self.device)
        x = dgl.ops.gspmm(g, 'mul', 'sum', lhs_data=padding_features, rhs_data=feat_mat.values())
        x = x[:self.feat_mat.shape[0], :]
        return x

    def get_rep(self):
        feat_mat = NGCF.dropout_sp_mat(self, self.feat_mat)
        representations = self.inductive_rep_layer(feat_mat)

        all_layer_rep = [representations]
        row, column = self.norm_adj.indices()
        g = dgl.graph((column, row), num_nodes=self.norm_adj.shape[0], device=self.device)
        for _ in range(self.n_layers):
            representations = dgl.ops.gspmm(g, 'mul', 'sum', lhs_data=representations, rhs_data=self.norm_adj.values())
            all_layer_rep.append(representations)
        all_layer_rep = torch.stack(all_layer_rep, dim=0)
        final_rep = all_layer_rep.mean(dim=0)
        return final_rep

    def bpr_forward(self, users, pos_items, neg_items):
        return NGCF.bpr_forward(self, users, pos_items, neg_items)

    def predict(self, users):
        return LightGCN.predict(self, users)

    def save(self, path):
        params = {'sate_dict': self.state_dict(), 'user_map': self.user_map,
                  'item_map': self.item_map, 'alpha': self.alpha}
        torch.save(params, path)

    def load(self, path):
        params = torch.load(path, map_location=self.device)
        self.load_state_dict(params['sate_dict'])
        self.user_map = params['user_map']
        self.item_map = params['item_map']
        self.alpha = params['alpha']
        self.feat_mat, _, _, self.row_sum = self.generate_feat(self.config['dataset'], is_updating=True)
        self.update_feat_mat()


'''
class AttIGCN(IGCN):
    def __init__(self, model_config):
        BasicModel.__init__(self, model_config)
        self.embedding_size = model_config['embedding_size']
        self.n_layers = model_config['n_layers']
        self.n_heads = 4
        self.dropout = model_config['dropout']
        self.feature_ratio = 1.
        self.alpha = 0.
        self.size_chunk = int(1e5)
        self.norm_adj = self.generate_graph(model_config['dataset'])
        self.feat_mat, self.user_map, self.item_map, self.row_sum = \
            self.generate_feat(model_config['dataset'],
                               ranking_metric=model_config.get('ranking_metric', 'normalized_degree'))
        self.update_feat_mat()
        
        self.embedding = nn.Embedding(self.feat_mat.shape[1], self.embedding_size)
        kaiming_uniform_(self.embedding.weight)
        self.weight_q = init_one_layer(self.embedding_size, self.embedding_size * self.n_heads)
        self.weight_k = init_one_layer(self.embedding_size, self.embedding_size * self.n_heads)
        self.to(device=self.device)

    def masked_mm(self, x_q, x_k, row, column):
        alpha = []
        n_non_zeros = row.shape[0]
        end_indices = list(np.arange(0, n_non_zeros, self.size_chunk, dtype=np.int64)) + [n_non_zeros]
        for i_chunk in range(1, len(end_indices)):
            t_q = torch.index_select(x_q, 0, row[end_indices[i_chunk - 1]:end_indices[i_chunk]])
            t_k = torch.index_select(x_k, 0, column[end_indices[i_chunk - 1]:end_indices[i_chunk]])
            alpha.append((t_q * t_k).sum(2))
        alpha = torch.cat(alpha, dim=0)
        return alpha

    def inductive_rep_layer(self, feat_mat, return_alpha=False):
        x_k = self.embedding.weight.detach()
        x_k = self.weight_k(x_k).view(-1, self.n_heads, self.embedding_size)

        row, column = feat_mat.indices()
        g = dgl.graph((column, row), num_nodes=self.feat_mat.shape[1], device=self.device)
        x_q = dgl.ops.gspmm(g, 'mul', 'sum', lhs_data=self.embedding.weight.detach(), rhs_data=feat_mat.values())
        x_q = self.weight_q(x_q).view(-1, self.n_heads, self.embedding_size)
        if self.training:
            alpha = checkpoint(self.masked_mm, x_q, x_k, row, column, preserve_rng_state=False)
        else:
            alpha = self.masked_mm(x_q, x_k, row, column)

        row_max_alpha = dgl.ops.gspmm(g, 'copy_rhs', 'max', lhs_data=None, rhs_data=alpha)
        alpha = alpha - row_max_alpha[row, :]
        alpha = torch.exp(alpha / np.sqrt(self.embedding_size) / 10.)
        row_sum_alpha = dgl.ops.gspmm(g, 'copy_rhs', 'sum', lhs_data=None, rhs_data=alpha)
        alpha = alpha / row_sum_alpha[row, :]
        alpha = alpha.mean(-1)
        if return_alpha:
            return alpha

        out = dgl.ops.gspmm(g, 'mul', 'sum', lhs_data=self.embedding.weight, rhs_data=alpha)
        out = out[:self.feat_mat.shape[0], :]
        return out

    def bpr_forward(self, users, pos_items, neg_items):
        users_r, pos_items_r, neg_items_r, l2_norm_sq = NGCF.bpr_forward(self, users, pos_items, neg_items)
        l2_norm_sq += torch.norm(self.weight_q.weight, p=2) ** 2 + torch.norm(self.weight_k.weight, p=2) ** 2
        return users_r, pos_items_r, neg_items_r, l2_norm_sq
'''


class IMF(IGCN):
    def __init__(self, model_config):
        super(IMF, self).__init__(model_config)

    def get_rep(self):
        feat_mat = NGCF.dropout_sp_mat(self, self.feat_mat)
        representations = IGCN.inductive_rep_layer(self, feat_mat)
        return representations


class IMCGAE(BasicModel):
    def __init__(self, model_config):
        super(IMCGAE, self).__init__(model_config)
        self.embedding_size = model_config['embedding_size']
        self.n_layers = model_config['n_layers']
        self.dropout = model_config['dropout']
        self.embedding = nn.Embedding(self.n_users + self.n_items + 3, self.embedding_size)
        self.norm_adj = self.generate_graph(model_config['dataset'])
        normal_(self.embedding.weight, std=0.1)
        self.to(device=self.device)

    def generate_graph(self, dataset):
        return LightGCN.generate_graph(self, dataset)

    def get_rep(self):
        personal_user_embedding = self.embedding.weight[:self.n_users, :]
        personal_item_embedding = self.embedding.weight[self.n_users:self.n_users + self.n_items, :]
        identical_embedding = self.embedding.weight[self.n_users + self.n_items, :]
        general_user_embedding = self.embedding.weight[self.n_users + self.n_items + 1, :]
        general_item_embedding = self.embedding.weight[self.n_users + self.n_items + 2, :]
        u_representations = torch.cat([personal_user_embedding,
                                       general_user_embedding[None, :].expand(personal_user_embedding.shape),
                                       identical_embedding[None, :].expand(personal_user_embedding.shape)], dim=1)
        i_representations = torch.cat([personal_item_embedding,
                                       general_item_embedding[None, :].expand(personal_item_embedding.shape),
                                       identical_embedding[None, :].expand(personal_item_embedding.shape)], dim=1)

        representations = torch.cat([u_representations, i_representations], dim=0)
        all_layer_rep = [representations]
        row, column = self.norm_adj.indices()
        g = dgl.graph((column, row), num_nodes=self.norm_adj.shape[0], device=self.device)
        for i in range(self.n_layers):
            node_dropout_masks = torch.ones(self.n_users + self.n_items, dtype=torch.float32, device=self.device)
            node_dropout_masks = F.dropout(node_dropout_masks, p=self.dropout - 0.1 * i, training=self.training)
            representations = representations * node_dropout_masks[:, None]
            representations = dgl.ops.gspmm(g, 'mul', 'sum', lhs_data=representations, rhs_data=self.norm_adj.values())
            all_layer_rep.append(representations / float(i + 2))
        all_layer_rep = torch.stack(all_layer_rep, dim=0)
        final_rep = all_layer_rep.sum(dim=0)
        return final_rep

    def bpr_forward(self, users, pos_items, neg_items):
        return NGCF.bpr_forward(self, users, pos_items, neg_items)

    def predict(self, users):
        return LightGCN.predict(self, users)


class MultiVAE(BasicModel):
    def __init__(self, model_config):
        super(MultiVAE, self).__init__(model_config)
        self.dropout = model_config['dropout']
        self.normalized_data_mat = self.get_data_mat(model_config['dataset'])

        self.e_layer_sizes = model_config['layer_sizes'].copy()
        self.e_layer_sizes.insert(0, self.normalized_data_mat.shape[1])
        self.d_layer_sizes = self.e_layer_sizes[::-1].copy()
        self.mid_size = self.e_layer_sizes[-1]
        self.e_layer_sizes[-1] = self.mid_size * 2
        self.encoder_layers = []
        self.decoder_layers = []
        for layer_idx in range(1, len(self.e_layer_sizes)):
            encoder_layer = init_one_layer(self.e_layer_sizes[layer_idx - 1], self.e_layer_sizes[layer_idx])
            self.encoder_layers.append(encoder_layer)
            decoder_layer = init_one_layer(self.d_layer_sizes[layer_idx - 1], self.d_layer_sizes[layer_idx])
            self.decoder_layers.append(decoder_layer)
        self.encoder_layers = nn.ModuleList(self.encoder_layers)
        self.decoder_layers = nn.ModuleList(self.decoder_layers)
        self.to(device=self.device)

    def get_data_mat(self, dataset):
        data_mat = sp.coo_matrix((np.ones((len(dataset.train_array),)), np.array(dataset.train_array).T),
                                 shape=(self.n_users, self.n_items), dtype=np.float32).tocsr()

        normalized_data_mat = normalize(data_mat, axis=1, norm='l2')
        return normalized_data_mat

    def ml_forward(self, users):
        users = users.cpu().numpy()
        profiles = self.normalized_data_mat[users, :]
        representations = get_sparse_tensor(profiles, self.device)

        representations = NGCF.dropout_sp_mat(self, representations)
        representations = torch.sparse.mm(representations, self.encoder_layers[0].weight.t())
        representations += self.encoder_layers[0].bias[None, :]
        l2_norm_sq = torch.norm(self.encoder_layers[0].weight, p=2)[None] ** 2
        for layer in self.encoder_layers[1:]:
            representations = layer(torch.tanh(representations))
            l2_norm_sq += torch.norm(layer.weight, p=2)[None] ** 2

        mean, log_var = representations[:, :self.mid_size], representations[:, -self.mid_size:]
        std = torch.exp(0.5 * log_var)
        kl = torch.sum(-log_var + torch.exp(log_var) + mean ** 2, dim=1)
        epsilon = torch.randn(mean.shape[0], mean.shape[1], device=self.device)
        representations = mean + float(self.training) * epsilon * std

        for layer in self.decoder_layers[:-1]:
            representations = torch.tanh(layer(representations))
            l2_norm_sq += torch.norm(layer.weight, p=2)[None] ** 2
        scores = self.decoder_layers[-1](representations)
        l2_norm_sq += torch.norm(self.decoder_layers[-1].weight, p=2)[None] ** 2
        return scores, kl, l2_norm_sq

    def predict(self, users):
        scores, _, _ = self.ml_forward(users)
        if scores.shape[1] < self.n_items:
            padding = torch.full([scores.shape[0], self.n_items - scores.shape[1]], -np.inf, device=self.device)
            scores = torch.cat([scores, padding], dim=1)
        return scores


class NeuMF(BasicModel):
    def __init__(self, model_config):
        super(NeuMF, self).__init__(model_config)
        self.embedding_size = model_config['embedding_size']
        self.layer_sizes = model_config['layer_sizes']
        self.mf_user_embedding = nn.Embedding(self.n_users, self.embedding_size)
        self.mf_item_embedding = nn.Embedding(self.n_items, self.embedding_size)
        self.mlp_user_embedding = nn.Embedding(self.n_users, self.layer_sizes[0] // 2)
        self.mlp_item_embedding = nn.Embedding(self.n_items, self.layer_sizes[0] // 2)
        self.mlp_layers = []
        for layer_idx in range(1, len(self.layer_sizes)):
            dense_layer = nn.Linear(self.layer_sizes[layer_idx - 1], self.layer_sizes[layer_idx])
            self.mlp_layers.append(dense_layer)
        self.mlp_layers = nn.ModuleList(self.mlp_layers)
        self.output_layer = nn.Linear(self.layer_sizes[-1] + self.embedding_size, 1, bias=False)

        kaiming_uniform_(self.mf_user_embedding.weight)
        kaiming_uniform_(self.mf_item_embedding.weight)
        kaiming_uniform_(self.mlp_user_embedding.weight)
        kaiming_uniform_(self.mlp_item_embedding.weight)
        self.init_mlp_layers()
        self.arch = 'gmf'
        self.to(device=self.device)

    def init_mlp_layers(self):
        for layer in self.mlp_layers:
            kaiming_uniform_(layer.weight)
            zeros_(layer.bias)
        ones_(self.output_layer.weight)

    def bce_forward(self, users, items):
        users_mf_e, items_mf_e = self.mf_user_embedding(users), self.mf_item_embedding(items)
        users_mlp_e, items_mlp_e = self.mlp_user_embedding(users), self.mlp_item_embedding(items)

        mf_vectors = users_mf_e * items_mf_e
        mlp_vectors = torch.cat([users_mlp_e, items_mlp_e], dim=1)
        for layer in self.mlp_layers:
            mlp_vectors = F.leaky_relu(layer(mlp_vectors))

        if self.arch == 'gmf':
            vectors = [mf_vectors, torch.zeros_like(mlp_vectors, device=self.device, dtype=torch.float32)]
        elif self.arch == 'mlp':
            vectors = [torch.zeros_like(mf_vectors, device=self.device, dtype=torch.float32), mlp_vectors]
        else:
            vectors = [mf_vectors, mlp_vectors]
        predict_vectors = torch.cat(vectors, dim=1)
        scores = predict_vectors * self.output_layer.weight
        l2_norm_sq = torch.norm(scores, p=2, dim=1) ** 2
        scores = scores.sum(dim=1)
        return scores, l2_norm_sq

    def predict(self, users):
        items = torch.arange(self.n_items, dtype=torch.int64, device=self.device).repeat(users.shape[0])
        users = users[:, None].repeat(1, self.n_items).flatten()
        scores, _ = self.bce_forward(users, items)
        scores = scores.reshape(-1, self.n_items)
        return scores
