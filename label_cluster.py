# 标签层次聚类模型
from Graph2 import build_heterogeneous_graph, generate_masks
import Config
import torch
from scipy.cluster.hierarchy import linkage, to_tree, dendrogram
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import json
import numpy as np
from tqdm import tqdm
from sklearn.decomposition import PCA
from Graph2 import build_node_features

def get_label_embeddings(cfg):
    """
    提取标签节点对应的嵌入
    """
    G = build_heterogeneous_graph(cfg)
    sample_mask, train_mask, test_mask, word_mask, label_mask, node2idx = generate_masks(G)
    node_emb = torch.load("gcns/{}_node_embeddings_best.pth".format(cfg.test), map_location="cpu")
    # node_emb = build_node_features(cfg, G)
    label_indices = torch.where(label_mask)[0]
    label_emb = node_emb[label_indices]  # [num_labels, dim]
    labels = [node for node, idx in node2idx.items() if idx in label_indices.tolist()]
    labels = [each.split("_")[1] for each in labels]
    return label_emb, labels


def label_cluster(cfg):
    label_emb, labels = get_label_embeddings(cfg)
    # Z = linkage(label_emb.detach().numpy(), method="ward", metric="euclidean")
    Z = linkage(label_emb.detach().numpy(), method="average", metric="cosine")
    plt.figure(figsize=(10, 7), dpi=600)
    dendrogram(
        Z,
        labels=labels,
        leaf_rotation=90,
        leaf_font_size=14  # 叶子标签字体大小
    )
    plt.title("Label Hierarchical Clustering", fontsize=16)  # 标题字体大小
    plt.xlabel("Labels", fontsize=14)  # x轴标签字体大小
    plt.ylabel("Distance", fontsize=14)  # y轴标签字体大小
    plt.xticks(fontsize=14)  # x轴刻度字体大小
    plt.yticks(fontsize=14)  # y轴刻度字体大小
    plt.subplots_adjust(bottom=0.6)  # 调整图像底部以防叶子标签被遮挡
    plt.savefig("{}-clustering.pdf".format(cfg.test))
    plt.show()
    # 转换为树
    root_node, _ = to_tree(Z, rd=True)
    tree = build_tree(root_node, labels)
    # 保存为 JSON
    save_path = "trees/{}_label_tree.json".format(cfg.test)
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(tree, f, ensure_ascii=False, indent=2)
    print(f"树结构已保存到 {save_path}")
    return tree


# def label_cluster_for_each(cfg, current_labels_list, save_path=None):
#     """
#     对 current_labels_list 中的每个子集生成等高二叉树，
#     最后保存到一个 JSON 文件。
#
#     :param cfg: 配置对象，至少包含 cfg.test
#     :param current_labels_list: list of list，每个子list是一组子集标签
#     :param save_path: 保存路径（默认 "trees/{cfg.test}_label_trees.json"）
#     :return: dict，包含所有子集的树结构
#     """
#     # 获取全量标签及其表征
#     label_emb, labels = get_label_embeddings(cfg)
#     label2emb = {lbl: emb.detach().numpy() for lbl, emb in zip(labels, label_emb)}
#
#     cluster_counter = [0]  # 用列表包裹方便在递归里修改
#
#     def build_balanced_tree(sub_labels, sub_embs):
#         """递归构建等高二叉树，并生成唯一 Cluster 编号"""
#         if len(sub_labels) == 1:
#             return {"name": sub_labels[0], "children": []}
#
#         # 用 PCA 第一主成分排序
#         pca = PCA(n_components=1)
#         coords = pca.fit_transform(sub_embs)
#         idx_sorted = coords[:, 0].argsort()
#         sorted_labels = [sub_labels[i] for i in idx_sorted]
#         sorted_embs = sub_embs[idx_sorted]
#
#         mid = len(sorted_labels) // 2
#         left_tree = build_balanced_tree(sorted_labels[:mid], sorted_embs[:mid])
#         right_tree = build_balanced_tree(sorted_labels[mid:], sorted_embs[mid:])
#
#         cluster_id = f"Cluster_{cluster_counter[0]}"
#         cluster_counter[0] += 1
#         return {"name": cluster_id, "children": [left_tree, right_tree]}
#
#     all_trees = {}
#     for idx, subset in tqdm(enumerate(current_labels_list),
#                             desc="Clustering label subsets", unit="subset"):
#         filtered_labels = []
#         filtered_embs = []
#         subset = subset[:cfg.candidates]
#         for lbl in subset:
#             if lbl.lower() in label2emb:
#                 filtered_labels.append(lbl.lower())
#                 filtered_embs.append(label2emb[lbl.lower()])
#             else:
#                 print(f"[Warning] !!!{lbl} 不在全量标签中，已跳过")
#         if len(filtered_labels) < 2:
#             print(f"[Skip] 子集 {idx} 标签数不足 2，无法聚类")
#             continue
#
#         filtered_embs = np.stack(filtered_embs, axis=0)
#         tree = build_balanced_tree(filtered_labels, filtered_embs)
#         all_trees[f"test_{idx}"] = tree
#
#     # 保存到文件
#     if save_path is None:
#         save_path = f"trees/{cfg.test}_label_trees_{cfg.candidates}.json"
#     with open(save_path, "w", encoding="utf-8") as f:
#         json.dump(all_trees, f, ensure_ascii=False, indent=2)
#
#     print(f"所有子集树结构已保存到 {save_path}")
#     return all_trees

def label_cluster_for_each(cfg, current_labels_list, n_children=2, save_path=None):
    """
    对 current_labels_list 中的每个子集生成等高 n 叉树，最后保存到一个 JSON 文件。
    :param cfg: 配置对象，至少包含 cfg.test
    :param current_labels_list: list of list，每个子list是一组子集标签
    :param save_path: 保存路径（默认 "trees/{cfg.test}_label_trees.json"）
    :param n_children: 每个节点的子节点数量，生成 n 叉树
    :return: dict，包含所有子集的树结构
    """
    # 获取全量标签及其表征
    label_emb, labels = get_label_embeddings(cfg)
    label2emb = {lbl: emb.detach().numpy() for lbl, emb in zip(labels, label_emb)}
    cluster_counter = [0]  # 用列表包裹方便在递归里修改
    def build_n_tree(sub_labels, sub_embs):
        """
        递归构建 n 叉树，每个 Cluster 节点直接挂子 Cluster 或叶子标签
        """
        # 如果标签数量小于等于 n_children，每个标签直接挂在一个 Cluster 下
        if len(sub_labels) <= n_children:
            cluster_id = f"Cluster_{cluster_counter[0]}"
            cluster_counter[0] += 1
            children = [{"name": lbl, "children": []} for lbl in sub_labels]
            return {"name": cluster_id, "children": children}

        # 用 PCA 第一主成分排序
        pca = PCA(n_components=1)
        coords = pca.fit_transform(sub_embs)
        idx_sorted = coords[:, 0].argsort()
        sorted_labels = [sub_labels[i] for i in idx_sorted]
        sorted_embs = sub_embs[idx_sorted]

        # 生成索引数组并切分成 n_children 个子集
        indices = np.arange(len(sorted_labels))
        split_indices = np.array_split(indices, n_children)

        children = []
        for split_idx in split_indices:
            split_labels = [sorted_labels[i] for i in split_idx]
            split_embs = sorted_embs[split_idx]
            # 如果这个子集只有 1 个标签，则直接挂叶子
            if len(split_labels) == 1:
                children.append({"name": split_labels[0], "children": []})
            else:
                # 递归生成 Cluster
                children.append(build_n_tree(split_labels, split_embs))

        cluster_id = f"Cluster_{cluster_counter[0]}"
        cluster_counter[0] += 1
        return {"name": cluster_id, "children": children}

    all_trees = {}
    for idx, subset in tqdm(enumerate(current_labels_list),
                            desc="Clustering label subsets", unit="subset"):
        filtered_labels = []
        filtered_embs = []
        subset = subset[:cfg.candidates]
        for lbl in subset:
            if lbl.lower() in label2emb:
                filtered_labels.append(lbl.lower())
                filtered_embs.append(label2emb[lbl.lower()])
            else:
                print(f"[Warning] !!!{lbl} 不在全量标签中，已跳过")
        if len(filtered_labels) < 2:
            print(f"[Skip] 子集 {idx} 标签数不足 2，无法聚类")
            continue

        filtered_embs = np.stack(filtered_embs, axis=0)
        tree = build_n_tree(filtered_labels, filtered_embs)
        all_trees[f"test_{idx}"] = tree

    # 保存到文件
    if save_path is None:
        save_path = f"trees/{cfg.test}_label_trees_{cfg.candidates}.json"
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(all_trees, f, ensure_ascii=False, indent=2)

    print(f"所有子集树结构已保存到 {save_path}")
    return all_trees


def build_tree(node, labels):
    """递归把 scipy.cluster.hierarchy 的树节点转成 dict"""
    if node.is_leaf():
        return {"name": labels[node.id], "children": []}
    else:
        left = build_tree(node.get_left(), labels)
        right = build_tree(node.get_right(), labels)
        return {"name": f"Cluster_{node.id}", "children": [left, right]}

def recursive_kmeans(embeddings, labels, depth=0, max_depth=4, cid=0):
    """
    固定层数递归 KMeans 多叉树，保证每个节点唯一命名
    """
    N = len(labels)
    node_name = f"Cluster_{depth}_{cid}" if depth < max_depth else f"Leaf_{depth}_{cid}"
    if depth >= max_depth or N <= 1:
        return {"name": node_name, "labels": labels, "children": []}
    # 动态计算当前层的 k
    k = int(np.ceil(N ** (1 / (max_depth - depth))))
    k = min(k, N)
    kmeans = KMeans(n_clusters=k, random_state=42)
    cluster_ids = kmeans.fit_predict(embeddings)
    children = []
    for sub_cid in range(k):
        idxs = np.where(cluster_ids == sub_cid)[0]
        sub_emb = embeddings[idxs]
        sub_labels = [labels[i] for i in idxs]
        # 递归生成子节点，传入子编号
        child_node = recursive_kmeans(sub_emb, sub_labels, depth=depth+1, max_depth=max_depth, cid=sub_cid)
        children.append(child_node)

    return {"name": node_name, "children": children}


def label_cluster_fixed_depth(cfg, max_depth=3):
    label_emb, labels = get_label_embeddings(cfg)
    label_emb_np = label_emb.detach().cpu().numpy()
    tree = recursive_kmeans(label_emb_np, labels, max_depth=max_depth)
    with open("trees/{}_label_tree.json".format(cfg.test), "w", encoding="utf-8") as f:
        json.dump(tree, f, ensure_ascii=False, indent=2)
    return tree


if __name__ == '__main__':
    cfg = Config.Config()
    for data in ['GoEmotions']:
        cfg.test = data
        label_cluster(cfg)
        # label_cluster_for_each(cfg)
    # tree = label_cluster_fixed_depth(cfg)
    # visualize_tree_multipartite(tree)