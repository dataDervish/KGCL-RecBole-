# -*- coding: utf-8 -*-
# @Time   : 2024/05/20
# @Author : Your Name
# @Email  : your_email@domain.com

r"""
KGCL
##################################################
Reference:
    Yuhao Yang et al. "Knowledge Graph Contrastive Learning for Recommendation." in SIGIR 2022.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.sparse as sp

from recbole.model.abstract_recommender import KnowledgeRecommender
from recbole.model.init import xavier_uniform_initialization
from recbole.model.layers import SparseDropout
from recbole.model.loss import BPRLoss, EmbLoss
from recbole.utils import InputType


class KGCL(KnowledgeRecommender):
    input_type = InputType.PAIRWISE

    def __init__(self, config, dataset):
        super(KGCL, self).__init__(config, dataset)

        # 1. 核心超参数加载
        self.embedding_size = config["embedding_size"]
        self.ui_layers = config["ui_layers"] if "ui_layers" in config else 3      
        self.kg_layers = config["kg_layers"] if "kg_layers" in config else 2      
        self.ssl_temp = config["ssl_temp"] if "ssl_temp" in config else 0.2      
        self.ssl_reg = config["ssl_reg"] if "ssl_reg" in config else 0.1        
        self.reg_weight = config["reg_weight"] if "reg_weight" in config else 1e-4 
        self.edge_drop_rate = config["edge_drop_rate"] if "edge_drop_rate" in config else 0.1

        # 2. 数据与图结构加载
        self.inter_matrix = dataset.inter_matrix(form="coo").astype(np.float32)
        # 构建 UI 交互的归一化拉普拉斯矩阵 (LightGCN 用)
        self.norm_ui_matrix = self.get_norm_ui_matrix().to(self.device)
        
        # 构建 KG 图谱邻接表 (保留实体间关联)
        self.kg_graph = dataset.kg_graph(form="coo", value_field="relation_id")
        self.kg_edge_index, self.kg_edge_type = self.get_edges(self.kg_graph)

     # 3. 定义 Embedding 层
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_size)
        self.entity_embedding = nn.Embedding(self.n_entities, self.embedding_size)
        
        # 加入关系 Embedding
        self.relation_embedding = nn.Embedding(self.n_relations, self.embedding_size) 
        
        self.W_kg = nn.Linear(self.embedding_size, self.embedding_size)

        # 4. 损失函数与 Dropout
        self.mf_loss = BPRLoss()
        self.reg_loss = EmbLoss()
        self.edge_dropout = SparseDropout(p=self.edge_drop_rate)
        
        self.restore_user_e = None
        self.restore_entity_e = None

        self.apply(xavier_uniform_initialization)

    def get_norm_ui_matrix(self):
        """构建 UI 二分图的归一化邻接矩阵用于 LightGCN 传播"""
        A = sp.dok_matrix((self.n_users + self.n_items, self.n_users + self.n_items), dtype=np.float32)
        inter_M = self.inter_matrix
        inter_M_t = self.inter_matrix.transpose()
        data_dict = dict(zip(zip(inter_M.row, inter_M.col + self.n_users), [1] * inter_M.nnz))
        data_dict.update(dict(zip(zip(inter_M_t.row + self.n_users, inter_M_t.col), [1] * inter_M_t.nnz)))
        A._update(data_dict)
        
        # D^{-1/2} A D^{-1/2}
        rowsum = np.array(A.sum(1))
        d_inv_sqrt = np.power(rowsum, -0.5).flatten()
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
        norm_adj = d_mat_inv_sqrt.dot(A).dot(d_mat_inv_sqrt).tocoo()
        
        indices = torch.LongTensor(np.array([norm_adj.row, norm_adj.col]))
        values = torch.FloatTensor(norm_adj.data)
        return torch.sparse.FloatTensor(indices, values, norm_adj.shape)

    def get_edges(self, graph):
        index = torch.LongTensor(np.array([graph.row, graph.col]))
        type = torch.LongTensor(np.array(graph.data))
        return index.to(self.device), type.to(self.device)

    def lightgcn_forward(self, norm_matrix):
        """UI 交互图上的 LightGCN 传播"""
        ego_embeddings = torch.cat([self.user_embedding.weight, self.entity_embedding.weight[:self.n_items]], dim=0)
        all_embeddings = [ego_embeddings]
        for _ in range(self.ui_layers):
            ego_embeddings = torch.sparse.mm(norm_matrix, ego_embeddings)
            all_embeddings.append(ego_embeddings)
        all_embeddings = torch.stack(all_embeddings, dim=1).mean(dim=1) # Layer均值聚合
        user_emb, item_emb = torch.split(all_embeddings, [self.n_users, self.n_items])
        return user_emb, item_emb

    def kg_forward(self, edge_index, edge_type):
        """带关系感知的 KG 传播"""
        from torch_scatter import scatter_mean
        
        entity_emb = self.entity_embedding.weight
        relation_emb = self.relation_embedding.weight
        all_entity_embeddings = [entity_emb]
        
        head, tail = edge_index
        
        for _ in range(self.kg_layers):
            # 获取尾节点特征和对应的边关系特征
            neigh_entity_emb = entity_emb[tail]
            edge_relation_emb = relation_emb[edge_type]
            
            # 将尾节点与边的关系进行哈达玛乘积（融合语义）
            neigh_emb = neigh_entity_emb * edge_relation_emb
            
            # 使用 torch_scatter 进行按头节点分组的均值聚合
            entity_agg = scatter_mean(src=neigh_emb, index=head, dim_size=self.n_entities, dim=0)
            
            # 残差连接与归一化
            entity_emb = entity_emb + self.W_kg(entity_agg) 
            entity_emb = F.normalize(entity_emb, p=2, dim=1)
            all_entity_embeddings.append(entity_emb)
            
        all_entity_embeddings = torch.stack(all_entity_embeddings, dim=1).mean(dim=1)
        return all_entity_embeddings

    def forward(self, perturbed=False):
        # 如果是训练阶段 (需要计算对比损失)
        if perturbed:
            # 1. UI 视图增强：丢弃用户物品交互边
            perturbed_ui_matrix = self.edge_dropout(self.norm_ui_matrix)
            ui_user_emb, ui_item_emb = self.lightgcn_forward(perturbed_ui_matrix)
            
            # 2. KG 视图增强：随机丢弃图谱关系边
            # 这里的 drop_rate 可以和 ui 保持一致，也可以单独设置，通常 0.1 ~ 0.2
            pert_kg_edge_index, pert_kg_edge_type = self.drop_kg_edges(
                self.kg_edge_index, self.kg_edge_type, self.edge_drop_rate
            )
            # 使用之前写好的带软剪枝/关系感知的 kg_forward
            kg_entity_emb = self.kg_forward(pert_kg_edge_index, pert_kg_edge_type)
            
        # 如果是测试/推理阶段 (使用完整原图)
        else:
            ui_user_emb, ui_item_emb = self.lightgcn_forward(self.norm_ui_matrix)
            kg_entity_emb = self.kg_forward(self.kg_edge_index, self.kg_edge_type)
            
        kg_item_emb = kg_entity_emb[:self.n_items]

        return ui_user_emb, ui_item_emb, kg_item_emb
    
    
    def calc_ssl_loss(self, ui_emb, kg_emb, batch_items):
        """修复版：强制去重，防止同一个 Item 在 batch 内互斥"""
        # 1. 提取 batch 内不重复的 item 集合
        unique_items = torch.unique(batch_items)
        
        # 2. 截取对应的 Embedding 并归一化
        z1 = F.normalize(ui_emb[unique_items], dim=1)
        z2 = F.normalize(kg_emb[unique_items], dim=1)
        
        # 3. 计算相似度矩阵
        tot_score = torch.matmul(z1, z2.transpose(0, 1)) / self.ssl_temp
        
        # 4. 对角线为正样本，计算交叉熵
        labels = torch.arange(z1.shape[0]).to(self.device)
        ssl_loss = F.cross_entropy(tot_score, labels)
        
        return ssl_loss


    def calculate_loss(self, interaction):
        if self.restore_user_e is not None:
            self.restore_user_e, self.restore_entity_e = None, None

        user = interaction[self.USER_ID]
        pos_item = interaction[self.ITEM_ID]
        neg_item = interaction[self.NEG_ITEM_ID]

        # 前向传播提取表征
        ui_user_emb, ui_item_emb, kg_item_emb = self.forward(perturbed=True)
        
        # 1. BPR 推荐主损失 (基于 UI 协同信号)
        u_embeddings = ui_user_emb[user]
        pos_embeddings = ui_item_emb[pos_item]
        neg_embeddings = ui_item_emb[neg_item]

        pos_scores = torch.mul(u_embeddings, pos_embeddings).sum(dim=1)
        neg_scores = torch.mul(u_embeddings, neg_embeddings).sum(dim=1)
        bpr_loss = self.mf_loss(pos_scores, neg_scores)

        # 2. InfoNCE 对比学习损失 (Cross-view: UI vs KG)
        # 强制模型对齐 Item 在 UI 行为空间和 KG 语义空间的表征
        ssl_loss = self.calc_ssl_loss(ui_item_emb, kg_item_emb, pos_item)

        # 3. 正则化损失
        reg_loss = self.reg_loss(u_embeddings, pos_embeddings, neg_embeddings)

        total_loss = bpr_loss + self.ssl_reg * ssl_loss + self.reg_weight * reg_loss
        return total_loss
    
    def drop_kg_edges(self, edge_index, edge_type, drop_rate):
        """对知识图谱进行随机边丢弃，生成强扰动视图"""
        n_edges = edge_index.shape[1]
        # 生成随机概率
        random_tensor = torch.rand(n_edges, device=self.device)
        # 保留权重大于 drop_rate 的边
        keep_mask = random_tensor > drop_rate
        
        return edge_index[:, keep_mask], edge_type[keep_mask]

    def predict(self, interaction):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]
        
        # 拿到三种特征
        ui_user_emb, ui_item_emb, kg_item_emb = self.forward()
        
        # 显式特征融合
        # 将协同过滤学到的物品特征，和知识图谱学到的物品特征相加
        final_item_emb = ui_item_emb[item] + kg_item_emb[item]
        
        scores = torch.mul(ui_user_emb[user], final_item_emb).sum(dim=1)
        return scores

    def full_sort_predict(self, interaction):
        user = interaction[self.USER_ID]
        
        if self.restore_user_e is None:
            ui_user_emb, ui_item_emb, kg_item_emb = self.forward()
            self.restore_user_e = ui_user_emb
            
            # 全局特征融合
            self.restore_entity_e = ui_item_emb + kg_item_emb
            
        u_embeddings = self.restore_user_e[user]
        i_embeddings = self.restore_entity_e
        
        scores = torch.matmul(u_embeddings, i_embeddings.transpose(0, 1))
        return scores.view(-1)