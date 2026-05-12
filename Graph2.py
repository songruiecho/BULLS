import Config
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import networkx as nx
import numpy as np
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from torch_geometric.nn import GATConv
import random
from nltk.tokenize import word_tokenize
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util


class GCNModel(nn.Module):
    def __init__(self, in_features, num_classes=None, hidden_dim=128, out_dim=128):
        """
        in_features: 输入节点特征维度
        hidden_dim: GCN隐藏层维度
        out_dim: GCN输出节点embedding维度
        num_classes: 分类类别数，如果为None则不做分类
        """
        super(GCNModel, self).__init__()
        self.conv1 = GCNConv(in_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, out_dim)
        # 分类头
        self.num_classes = num_classes
        if num_classes is not None:
            self.classifier = nn.Linear(out_dim, num_classes)

    def forward(self, x, edge_index, return_logits=True):
        """
        x: [num_nodes, in_features]
        edge_index: 图边索引
        return_logits: 如果为True，返回分类预测结果，否则返回节点embedding
        """
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv2(x, edge_index)
        if return_logits and self.num_classes is not None:
            logits = self.classifier(x)  # [num_nodes, num_classes]
            return x, logits

        return x  # 节点embedding



class GATModel(nn.Module):
    def __init__(self, in_features, num_classes=None, hidden_dim=128, out_dim=64, heads=4):
        super(GATModel, self).__init__()
        # 第一层多头注意力
        self.gat1 = GATConv(in_features, hidden_dim, heads=heads, dropout=0.2)
        # 第二层合并 heads（concat=False 即取平均）
        self.gat2 = GATConv(hidden_dim * heads, out_dim, heads=1, concat=False, dropout=0.2)

        self.num_classes = num_classes
        if num_classes is not None:
            self.classifier = nn.Linear(out_dim, num_classes)

    def forward(self, x, edge_index, return_logits=False):
        x = F.elu(self.gat1(x, edge_index))
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.gat2(x, edge_index)

        if return_logits and self.num_classes is not None:
            return self.classifier(x)
        return x



def build_heterogeneous_graph(cfg):
    """
    构建一个包含文本、单词和标签的异构图（使用 NetworkX Graph）。

    节点顺序：
        1. 所有文本节点（train + test）
        2. 所有单词节点
        3. 所有标签节点
    """
    from dataloader import load_train_test
    train_text, test_text, train_labels, test_labels = load_train_test(cfg.test)

    G = nx.Graph()

    # --- 1. 添加文本节点 ---
    text_nodes = []
    for i, text in enumerate(train_text):
        node_name = f"traintext_{i}"
        G.add_node(node_name, type="train_text", text=text)
        text_nodes.append(node_name)
    for i, text in enumerate(test_text):
        node_name = f"testtext_{i}"
        G.add_node(node_name, type="test_text", text=text)
        text_nodes.append(node_name)

    # --- 2. 添加单词节点 ---
    all_texts = np.concatenate([train_text, test_text], axis=0)
    word_nodes_set = set()
    for text in all_texts:
        words = word_tokenize(text)
        for w in words:
            word_nodes_set.add(w.lower())  # 全部小写统一
    for w in word_nodes_set:
        G.add_node(f"word_{w}", type="word")

    # --- 3. 添加标签节点 ---
    label_set = set([lbl.lower() for lbl in train_labels])
    for lbl in label_set:
        G.add_node(f"label_{lbl}", type="label")

    # --- 4. 添加文本-单词边 ---
    for i, text in enumerate(all_texts):
        text_node = text_nodes[i]
        words = word_tokenize(text)
        for w in words:
            word_node = f"word_{w.lower()}"
            G.add_edge(word_node, text_node, type="word-text")

    # --- 5. 添加文本-标签边（只训练集有标签） ---
    for i, lbl in enumerate(train_labels):
        text_node = f"traintext_{i}"
        label_node = f"label_{lbl.lower()}"
        G.add_edge(text_node, label_node, type="text-label")

    # --- 6. 添加标签-单词边 ---
    for lbl in label_set:
        label_node = f"label_{lbl}"
        words = word_tokenize(lbl)
        for w in words:
            word_node = f"word_{w.lower()}"
            if word_node not in G:
                G.add_node(word_node, type="word")  # 避免孤立单词节点
            G.add_edge(word_node, label_node, type="word-label")

    return G

def build_node_features(cfg, G, batch_size=128):
    """
    为每个节点生成特征，使用sentence-transformers来获取节点的嵌入。
    """
    from dataloader import load_train_test
    train_text, test_text, train_labels, test_labels = load_train_test(cfg.test)
    model = SentenceTransformer(cfg.model_source + '/all-MiniLM-L6-v2').cuda()
    device = model.device
    node_features = []
    # 分批处理文本节点
    text_nodes = [node for node in G.nodes if "text" in node.split('_')[0]]
    text_texts = []
    for node in text_nodes:
        if 'train' in node:
            text = train_text[int(node.split('_')[1])]
        else:
            text = test_text[int(node.split('_')[1])]
        text_texts.append(text)
    # 使用 tqdm 包装文本批处理，显示进度条
    text_embeddings = []
    for i in tqdm(range(0, len(text_texts), batch_size), desc="Processing Texts", unit="batch"):
        batch_texts = text_texts[i:i + batch_size]
        batch_embeddings = model.encode(batch_texts, device=device)  # 批量处理，指定设备
        text_embeddings.append(batch_embeddings)  # 将嵌入结果添加到列表
    text_embeddings = np.concatenate(text_embeddings, axis=0)
    # 将文本节点的嵌入添加到特征列表
    for embedding in text_embeddings:
        node_features.append(torch.tensor(embedding).cpu())  # 转移到CPU
    # 分批处理标签节点
    label_nodes = [node for node in G.nodes if node.split('_')[0] == "label"]
    label_labels = [node.split('_')[1] for node in label_nodes]
    # # 使用 tqdm 包装标签批处理，显示进度条
    label_embeddings = []
    for i in tqdm(range(0, len(label_labels), batch_size), desc="Processing Labels", unit="batch"):
        batch_labels = label_labels[i:i + batch_size]
        batch_embeddings = model.encode(batch_labels, device=device)  # 批量处理，指定设备
        label_embeddings.append(batch_embeddings)  # 将嵌入结果添加到列表
    label_embeddings = np.concatenate(label_embeddings, axis=0)
    # 将标签节点的嵌入添加到特征列表
    for embedding in label_embeddings:
        node_features.append(torch.tensor(embedding).cpu())  # 转移到CPU
    # 分批处理单词节点
    word_nodes = [node for node in G.nodes if node.split('_')[0] == "word"]
    word_words = [node.split('_')[1] for node in word_nodes]
    # 使用 tqdm 包装单词批处理，显示进度条
    word_embeddings = []
    for i in tqdm(range(0, len(word_words), batch_size), desc="Processing Words", unit="batch"):
        batch_words = word_words[i:i + batch_size]
        batch_embeddings = model.encode(batch_words, device=device)  # 批量处理，指定设备
        word_embeddings.append(batch_embeddings)  # 将嵌入结果添加到列表
    word_embeddings = np.concatenate(word_embeddings, axis=0)
    # 将单词节点的嵌入添加到特征列表
    for embedding in word_embeddings:
        node_features.append(torch.tensor(embedding).cpu())  # 转移到CPU
    # 将所有节点的特征转化为 tensor
    node_features = torch.stack(node_features)
    return node_features

def contrastive_loss(label_emb, sample_emb, label_indices, sample_indices, tau=0.1):
    z_l = label_emb[label_indices]  # [num_pairs, d]
    z_s = sample_emb[sample_indices]  # [num_pairs, d]
    sim_matrix = torch.matmul(z_l, z_s.T) / tau
    labels = torch.arange(sim_matrix.size(0), device=sim_matrix.device)
    loss = F.cross_entropy(sim_matrix, labels)
    return loss

def generate_masks(G):
    """
    G: NetworkX 图，每个节点有 'type' 属性，取值 'sample', 'label', 'word'
    返回:
        sample_mask, label_mask, word_mask: Boolean Tensor, 长度 = 节点数
        node2idx: 节点 -> 索引映射
    """
    node2idx = {node: idx for idx, node in enumerate(G.nodes())}
    num_nodes = len(G.nodes())
    sample_mask = torch.zeros(num_nodes, dtype=torch.bool)
    label_mask = torch.zeros(num_nodes, dtype=torch.bool)
    word_mask = torch.zeros(num_nodes, dtype=torch.bool)
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    for node in G.nodes():
        idx = node2idx[node]
        ntype = node.split('_')[0]
        if 'traintext' in ntype:
            train_mask[idx] = True
        if 'testtext' in ntype:
            test_mask[idx] = True
        if 'text' in ntype:
            sample_mask[idx] = True
        elif ntype == 'label':
            label_mask[idx] = True
        elif ntype == 'word':
            word_mask[idx] = True
    return sample_mask, train_mask, test_mask, word_mask, label_mask, node2idx

def generate_masks2(G, sample_ratio=1.0, seed=42):
    """
    G: NetworkX 图，每个节点名称包含类型信息
    sample_ratio: 训练样本采样比例，例如:
        1.0   -> 使用全部训练数据
        0.5   -> 使用 1/2 训练数据
        0.25  -> 使用 1/4 训练数据
        0.125 -> 使用 1/8 训练数据
    seed: 随机种子
    返回:
        sample_mask
        train_mask
        test_mask
        word_mask
        label_mask
        node2idx
    """
    random.seed(seed)
    node2idx = {node: idx for idx, node in enumerate(G.nodes())}
    num_nodes = len(G.nodes())
    sample_mask = torch.zeros(num_nodes, dtype=torch.bool)
    label_mask = torch.zeros(num_nodes, dtype=torch.bool)
    word_mask = torch.zeros(num_nodes, dtype=torch.bool)
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    # 收集所有训练节点
    train_nodes = []
    for node in G.nodes():
        ntype = node.split('_')[0]
        if 'traintext' in ntype:
            train_nodes.append(node)
    # 采样训练节点
    sampled_num = max(1, int(len(train_nodes) * sample_ratio))
    sampled_train_nodes = set(random.sample(train_nodes, sampled_num))
    # 构建 mask
    for node in G.nodes():
        idx = node2idx[node]
        ntype = node.split('_')[0]
        # train mask（采样后）
        if node in sampled_train_nodes:
            train_mask[idx] = True
        # test mask
        if 'testtext' in ntype:
            test_mask[idx] = True
        # sample mask
        if 'text' in ntype:
            sample_mask[idx] = True
        # label mask
        elif ntype == 'label':
            label_mask[idx] = True
        # word mask
        elif ntype == 'word':
            word_mask[idx] = True
    return (
        sample_mask,
        train_mask,
        test_mask,
        word_mask,
        label_mask,
        node2idx
    )


def generate_pos_neg_edges(G, node2idx, label_mask, num_neg_per_pos=1):
    """
    从 NetworkX 图生成对比学习的正负边

    Args:
        G: NetworkX 图，text-label 边 type='text-label'
        node2idx: dict, 节点 -> 索引
        label_mask: Boolean Tensor, 标记哪些节点是标签
        num_neg_per_pos: 每条正边对应生成多少条负边

    Returns:
        pos_edges: [2, num_pos_edges] Tensor
        neg_edges: [2, num_neg_edges] Tensor
    """
    # ------------------------
    # 生成正样本边
    # ------------------------
    pos_edges_list = []
    for u, v, attr in G.edges(data=True):
        if attr.get('type') == 'text-label':
            pos_edges_list.append([node2idx[u], node2idx[v]])
    if len(pos_edges_list) == 0:
        raise ValueError("图中没有 text-label 边！")
    pos_edges = torch.tensor(pos_edges_list, dtype=torch.long).t()  # [2, num_pos_edges]
    # ------------------------
    # 生成负样本边
    # ------------------------
    all_label_indices = torch.where(label_mask)[0].tolist()
    neg_samples = []
    neg_labels = []
    for s_idx, l_idx in zip(pos_edges[0].tolist(), pos_edges[1].tolist()):
        for _ in range(num_neg_per_pos):
            neg_l = random.choice(all_label_indices)
            while neg_l == l_idx:  # 避免选到正边标签
                neg_l = random.choice(all_label_indices)
            neg_samples.append(s_idx)
            neg_labels.append(neg_l)
    neg_edges = torch.tensor([neg_samples, neg_labels], dtype=torch.long)
    return pos_edges, neg_edges


def compute_similarity(h, edges, sim="cosine"):
    """
    h: 节点表征 [num_nodes, dim]
    edges: [num_edges, 2] 的 tensor，存放边的两个节点索引
    sim: 相似度计算方式 ("cosine" or "dot")
    """
    src, dst = edges[:, 0], edges[:, 1]
    h_src, h_dst = h[src], h[dst]
    if sim == "cosine":
        sim_val = F.cosine_similarity(h_src, h_dst)  # [num_edges]
    else:
        sim_val = torch.sum(h_src * h_dst, dim=-1)   # 点积
    return sim_val

def info_nce_loss(h, pos_edges, neg_edges, temperature=0.5):
    # 正例相似度
    pos_sim = compute_similarity(h, pos_edges)  # [num_pos]
    # 负例相似度
    neg_sim = compute_similarity(h, neg_edges)  # [num_pos * num_neg]
    num_pos = pos_sim.size(0)
    num_neg = neg_sim.size(0) // num_pos
    neg_sim = neg_sim.view(num_pos, num_neg)  # [num_pos, num_neg]
    # 拼接 logit
    logits = torch.cat([pos_sim.unsqueeze(1), neg_sim], dim=1)  # [num_pos, 1+num_neg]
    logits = logits / temperature
    labels = torch.zeros(num_pos, dtype=torch.long).cuda()  # 正例在第 0 列
    loss = F.cross_entropy(logits, labels)
    return loss

# --------------------------
# 训练循环
# --------------------------
def train_gcn_contrastive(cfg, model, features, edge_index, pos_edges, neg_edges, lr=1e-3, epochs=20):
    """
    model: GCNModel
    features: [num_nodes, dim] 节点特征
    edge_index: [2, num_edges] 图边索引
    pos_edges: 正边
    neg_edges: 负边
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for epoch in tqdm(range(epochs)):
        model.train()
        optimizer.zero_grad()
        h = model(features.cuda(), edge_index.cuda())  # 前向
        loss = info_nce_loss(h, pos_edges.cuda(), neg_edges.cuda())
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
    # 先将模型参数移动到 CPU
    model_cpu = model.to('cpu')
    torch.save(model_cpu.state_dict(), "gcns/{}_gcn_model.pth".format(cfg.test))
    # --------------------------
    # 节点嵌入保存
    node_emb_cpu = h.cpu()
    torch.save(node_emb_cpu, "gcns/{}_node_embeddings.pth".format(cfg.test))
    return model, h  # 返回训练好的模型和节点嵌入


def train_gcn_joint(cfg, lr=1e-3, epochs=20, alpha=0.5):
    """
    model: GCNClassifier（包含分类头）
    features: [num_nodes, dim] 节点特征
    edge_index: [2, num_edges] 图边索引
    pos_edges: 正边索引 [2, num_pos]
    neg_edges: 负边索引 [2, num_neg]
    node_labels: [num_nodes] 节点类别标签（可选，用于分类损失）
    train_mask: boolean mask，训练节点索引
    alpha: 分类损失权重，(1-alpha)用于contrastive loss
    """
    from dataloader import load_train_test
    train_text, test_text, train_labels, test_labels = load_train_test(cfg.test)
    label_list = []
    for l in train_labels:
        if l not in label_list:
            label_list.append(l)
    print('标签空间大小：',format(len(label_list)), len(set(test_labels)))
    node_labels = torch.Tensor([label_list.index(l) for l in train_labels]).cuda()
    test_labels = torch.Tensor([label_list.index(l) for l in test_labels]).cuda()
    G = build_heterogeneous_graph(cfg)
    sample_mask, train_mask, test_mask, word_mask, label_mask, node2idx = generate_masks(G)
    pos_edges, neg_edges = generate_pos_neg_edges(G, node2idx, label_mask)
    # 构建 edge_index
    edge_index_list = []
    for u, v in G.edges():
        edge_index_list.append([node2idx[u], node2idx[v]])
    edge_index = torch.tensor(edge_index_list, dtype=torch.long).t()
    edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1).cuda()  # 双向边
    features = build_node_features(cfg, G).cuda()
    model = GCNModel(features.shape[1], len(label_list)).cuda()
    # model = GATModel(features.shape[1], len(label_list)).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_acc = 0.0  # 记录最佳准确率
    for epoch in tqdm(range(epochs)):
        model.train()
        optimizer.zero_grad()
        # 前向
        h = model(features, edge_index, return_logits=False)  # 节点embedding
        loss = 0.0
        # 对比损失
        if pos_edges is not None and neg_edges is not None:
            pos_h = h[pos_edges[0]] * h[pos_edges[1]]
            neg_h = h[neg_edges[0]] * h[neg_edges[1]]
            # 简单 info_nce / margin loss 示例
            logits = torch.cat([pos_h.sum(dim=-1, keepdim=True), neg_h.sum(dim=-1, keepdim=True)], dim=1)
            labels = torch.zeros(logits.size(0), dtype=torch.long).cuda()  # 正边为0
            contrastive_loss = F.cross_entropy(logits, labels)
            loss += (1 - alpha) * contrastive_loss
        # 分类损失
        if node_labels is not None and train_mask is not None and model.num_classes is not None:
            h, logits_cls = model(features, edge_index, return_logits=True)
            # 采样 先取 train logits
            train_logits = logits_cls[train_mask]
            # train sample 数量
            N = train_logits.size(0)
            # 随机采样 50%
            perm = torch.randperm(N).cuda()
            keep_num = int(N * 1/16)
            selected = perm[:keep_num]
            cls_loss = F.cross_entropy(
                train_logits[selected],
                node_labels[selected].long()
            )
            loss += alpha * cls_loss
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

        # ------------------- 测试阶段 -------------------
        if test_labels is not None and test_mask is not None:
            model.eval()
            with torch.no_grad():
                h, logits_test = model(features, edge_index, return_logits=True)
                preds = logits_test[test_mask].argmax(dim=-1)
                correct = (preds == test_labels).sum().item()
                total = test_mask.sum().item() if test_mask.dtype == torch.bool else len(test_mask)
                acc = correct / total
        # else:
        #     print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
        # 保存模型
        if acc > best_acc:
            best_acc = acc
            best_model_state = model.state_dict()
            best_node_emb = h.clone().detach().cpu()
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}, Test Acc: {acc:.4f}")
            # torch.save(best_model_state, f"gcns/{cfg.test}_gcn_best_model.pth")
            # torch.save(best_node_emb, f"gcns/{cfg.test}_node_embeddings_best.pth")


if __name__ == '__main__':
    cfg = Config.Config()
    for dataset in ['HWU64', 'GoEmotions', 'HuffPost15', 'TacRED', 'Banking77', 'Clinc150']:
        cfg.test = dataset
        # G = build_heterogeneous_graph(cfg)
        # sample_mask, word_mask, label_mask, node2idx = generate_masks(G)
        # pos_edges, neg_edges = generate_pos_neg_edges(G, node2idx, label_mask)
        # # 构建 edge_index
        # edge_index_list = []
        # for u, v in G.edges():
        #     edge_index_list.append([node2idx[u], node2idx[v]])
        # edge_index = torch.tensor(edge_index_list, dtype=torch.long).t()
        # edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)  # 双向边
        # node_features = build_node_features(G)
        # model = GCNModel(in_features=node_features.size(1), hidden_dim=128, out_dim=64).cuda()
        # trained_model, node_emb = train_gcn_contrastive(
        #     cfg, model, node_features, edge_index, pos_edges, neg_edges,
        #     lr=1e-3, epochs=100
        # )
        # print(node_emb.shape)
        train_gcn_joint(cfg, lr=1e-2, epochs=600, alpha=0.9)
