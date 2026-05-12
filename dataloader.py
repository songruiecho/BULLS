import pandas as pd
from sentence_transformers import SentenceTransformer, util
from sentence_transformers.util import cos_sim
from transformers import AutoModelForCausalLM, AutoTokenizer
import Config
import torch
from tqdm import tqdm
import json
from collections import defaultdict
from tree_parser import get_sibling_groups_with_parent
import random
import os
from label_cluster import build_heterogeneous_graph, generate_masks, label_cluster_for_each
import numpy as np
from Twin import TwinBERT, TextLabelDataset, DataLoader
from Graph2 import build_node_features, GCNModel, GATModel
import re
from Influence import calc_influence_set, MLPClassifier

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

def load_train_test(dataname='Banking77'):
    if dataname == 'Clinc150':
        train_file = 'dataset/{}/train.tsv'.format(dataname)
        test_file = 'dataset/{}/test.tsv'.format(dataname)
    else:
        train_file = 'dataset/{}/train.csv'.format(dataname)
        test_file = 'dataset/{}/test.csv'.format(dataname)
    try:
        train_df = pd.read_csv(train_file, sep='\t').dropna()
        testdf = pd.read_csv(test_file, sep='\t').dropna()
        train_df = train_df.drop_duplicates(subset='text', keep='first')
        testdf = testdf.drop_duplicates(subset='text', keep='first')
        train_text = train_df['text'].values
        test_text = testdf['text'].values
        train_labels = train_df['label'].values
        test_labels = testdf['label'].values
    except:
        train_df = pd.read_csv(train_file, sep=',').dropna()
        testdf = pd.read_csv(test_file, sep=',').dropna()
        train_df = train_df.drop_duplicates(subset='text', keep='first')
        testdf = testdf.drop_duplicates(subset='text', keep='first')
        train_text = train_df['text'].values
        test_text = testdf['text'].values
        train_labels = train_df['label'].values
        test_labels = testdf['label'].values

    train_labels = [re.sub('[_/]', ' ', each.lower()) for each in train_labels]
    test_labels = [re.sub('[_/]', ' ', each.lower()) for each in test_labels]
    print(len(train_text), len(test_text))
    return train_text, test_text, train_labels, test_labels


def retriever(cfg, dataname='Banking77', topk=100, batch_size=64):
    train_text, test_text, train_labels, test_labels = load_train_test(dataname)
    # 加载 sentence-bert 模型
    model = SentenceTransformer(cfg.model_source + '/all-MiniLM-L6-v2').cuda()
    # 编码 train / test 文本（分 batch）
    train_emb = model.encode(
        train_text,
        batch_size=batch_size,
        convert_to_tensor=True,
        show_progress_bar=True
    )
    test_emb = model.encode(
        test_text,
        batch_size=batch_size,
        convert_to_tensor=True,
        show_progress_bar=True
    )
    # 一次性计算相似度矩阵
    cos_scores = util.cos_sim(test_emb, train_emb)

    # 构建类别到训练样本索引映射
    class_to_train_ids = defaultdict(list)
    for idx, label in enumerate(train_labels):
        class_to_train_ids[label].append(idx)

    results = []
    for i in tqdm(range(len(test_text)), desc="Constructing results"):
        # 总体 top-k
        topk_values, topk_indices = torch.topk(cos_scores[i], k=topk)
        overall_topk = [
            {"train_id": tid, "score": float(score)}
            for tid, score in zip(topk_indices.tolist(), topk_values.tolist())
        ]
        # 按类别 top-k
        per_class_topk = {}
        for lbl, ids in class_to_train_ids.items():
            # 对每个类别取该类别训练样本的相似度
            class_scores = cos_scores[i, ids]
            class_topk_values, class_topk_indices = torch.topk(class_scores, k=min(10, len(ids)))
            per_class_topk[lbl] = [
                {"train_id": ids[idx], "score": float(score)}
                for idx, score in zip(class_topk_indices.tolist(), class_topk_values.tolist())
            ]

        results.append({
            "query_id": i,
            "query_label": test_labels[i],
            "overall_topk": overall_topk,
            "per_class_topk": per_class_topk
        })

    save_path = f'dataset/retriever_results/knn_{dataname}.json'
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"Saved retrieval results to {save_path}")
    return results

def retriever_mmr(cfg, dataname='Banking77', topk=10, batch_size=64, lambda_mmr=0.7):
    """
    基于 MMR (Maximal Marginal Relevance) 的多样性检索
    """
    train_text, test_text, train_labels, test_labels = load_train_test(dataname)
    # 加载 Sentence-BERT 模型
    model = SentenceTransformer(cfg.model_source + '/all-MiniLM-L6-v2').cuda()
    # 编码 train / test 文本
    train_emb = model.encode(train_text, batch_size=batch_size, convert_to_tensor=True, show_progress_bar=True)
    test_emb = model.encode(test_text, batch_size=batch_size, convert_to_tensor=True, show_progress_bar=True)
    # 构建类别到训练样本索引映射
    class_to_train_ids = defaultdict(list)
    for idx, label in enumerate(train_labels):
        class_to_train_ids[label].append(idx)
    results = []
    prefilter_k = 20
    for i in tqdm(range(len(test_text)), desc="MMR fast retrieval"):
        query_emb = test_emb[i].unsqueeze(0)  # [1, d]
        sim_qt = util.cos_sim(query_emb, train_emb).squeeze(0)  # [n_train]
        # Step 1: 先取最相关的 prefilter_k 个候选（减少后续计算量）
        prefilter_k = min(prefilter_k, len(train_text))
        sim_topk, idx_topk = torch.topk(sim_qt, k=prefilter_k)
        candidate_emb = train_emb[idx_topk]  # [prefilter_k, d]
        # Step 2: 计算候选间相似度矩阵（prefilter_k × prefilter_k）
        sim_tt = util.cos_sim(candidate_emb, candidate_emb).to(sim_qt.device)
        # Step 3: 初始化
        selected = []
        remaining = list(range(prefilter_k))
        # mmr_scores = sim_topk.clone()  # 初始化为 query–train 相似度
        max_div = torch.zeros(prefilter_k, device=sim_qt.device)
        for _ in range(topk):
            if not remaining:
                break
            # 综合打分
            combined = lambda_mmr * sim_topk - (1 - lambda_mmr) * max_div
            best_idx_local = torch.argmax(combined[remaining]).item()
            best_global = remaining[best_idx_local]
            selected.append(best_global)
            remaining.remove(best_global)
            # 更新 diversity 分数
            new_sim = sim_tt[:, best_global]
            max_div = torch.maximum(max_div, new_sim)
        # Step 4: 返回 topk 结果（映射回原始索引）
        selected_global = idx_topk[selected]
        topk_scores = sim_qt[selected_global]
        overall_topk = [
            {"train_id": int(idx), "score": float(score)}
            for idx, score in zip(selected_global.tolist(), topk_scores.tolist())
        ]
        # # 按类别 Top-K
        # per_class_topk = {}
        # for lbl, ids in class_to_train_ids.items():
        #     class_scores = sim_query_train[ids]
        #     class_topk_values, class_topk_indices = torch.topk(class_scores, k=min(10, len(ids)))
        #     per_class_topk[lbl] = [
        #         {"train_id": ids[idx], "score": float(score)}
        #         for idx, score in zip(class_topk_indices.tolist(), class_topk_values.tolist())
        #     ]
        results.append({
            "query_id": i,
            "query_label": test_labels[i],
            "overall_topk": overall_topk,
            # "per_class_topk": per_class_topk
        })
    save_path = f'dataset/retriever_results/mmr_{dataname}.json'
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Saved MMR retrieval results to {save_path}")
    return results

def retriever_gcn(cfg, dataname='Banking77', topk=10, batch_size=64):
    '''  根据GCN按照标签表征与测试样本表征检索最相关的标签值
    :param cfg:
    :param dataname:
    :param topk:
    :param batch_size:
    :return:
    '''
    train_text, test_text, train_labels, test_labels = load_train_test(dataname)
    label_set = []
    for l in train_labels:
        if l not in label_set:
            label_set.append(l)
    G = build_heterogeneous_graph(cfg)
    sample_mask, train_mask, test_mask, word_mask, label_mask, node2idx = generate_masks(G)
    edge_index_list = []
    for u, v in G.edges():
        edge_index_list.append([node2idx[u], node2idx[v]])
    edge_index = torch.tensor(edge_index_list, dtype=torch.long).t()
    edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1).cuda()  # 双向边
    node_emb = torch.load("gcns/{}_node_embeddings_best.pth".format(cfg.test), map_location="cpu")
    features = build_node_features(cfg, G).cuda()
    model = GCNModel(features.shape[1], len(label_set)).cuda()
    state_dict = torch.load("gcns/{}_gcn_best_model.pth".format(cfg.test), map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    with torch.no_grad():
        x, logits_test = model(features, edge_index, return_logits=True)
        logits_test = logits_test[test_mask].cpu()

    sample_indices = torch.where(sample_mask)[0]
    assert sample_indices.shape[0] == len(train_labels) + len(test_labels)
    train_indices, test_indices = sample_indices[:len(train_text)], sample_indices[len(train_text):]
    # 测试文本 embedding
    train_emb, test_emb = features[train_indices], features[test_indices]  # shape: [num_test, dim]
    # train_emb, test_emb = node_emb[train_indices], node_emb[test_indices]  # shape: [num_test, dim]
    # train_emb = torch.cat([train_emb.cpu(), train_emb2], dim=1)    # 融合图和文本的双重表征
    # test_emb = torch.cat([test_emb.cpu(), test_emb2], dim=1)
    # # 标签 embedding
    # label_emb = node_emb[label_indices]  # shape: [num_labels, dim]
    text_sim_matrix = util.cos_sim(train_emb, test_emb)  # [num_train, num_test]
    # 10. top-k
    results = []
    for i in tqdm(range(len(test_text)), desc="Constructing results"):
        # topk 标签
        sim_row = logits_test[i]  # 当前测试样本与所有标签的相似度
        topk_vals, topk_indices = torch.topk(sim_row, k=topk)
        topk_labels = [label_set[idx] for idx in topk_indices]
        # overall_topk: topk训练样本id
        sims = text_sim_matrix[:, i]
        topk_train_vals, topk_train_indices = torch.topk(sims, k=topk)
        overall_topk = topk_train_indices.tolist()
        # per_class_topk: 每个类别下与测试样本最相似的训练样本id
        per_class_topk = {}
        for lbl in label_set:
            train_ids_of_class = [j for j, l in enumerate(train_labels) if l == lbl]
            if not train_ids_of_class:
                continue
            class_sims = sims[train_ids_of_class]
            class_topk_values, class_topk_indices = torch.topk(class_sims, k=min(topk, len(class_sims)))
            per_class_topk[lbl] = [
                {"train_id": train_ids_of_class[idx], "score": float(score)}
                for idx, score in zip(class_topk_indices.tolist(), class_topk_values.tolist())
            ]
        results.append({
            "query_id": i,
            "query_label": test_labels[i],
            "topk": topk_labels,
            "overall_topk": overall_topk,
            "per_class_topk": per_class_topk
        })
    save_path = f'dataset/retriever_results/gcn_{dataname}.json'
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    return results


# def retriever_bert(cfg, dataname='Banking77', topk=10, batch_size=64):
#     train_text, test_text, train_labels, test_labels = load_train_test(cfg.test)
#     label_set = list(set(train_labels))
#     model = TwinBERT(model_name='/home/tianmingjie/songrui/models/SimCSE/').cuda()
#     tokenizer = AutoTokenizer.from_pretrained('/home/tianmingjie/songrui/models/SimCSE/')
#     save_path = 'twins/{}'.format(cfg.test)
#     model.load_state_dict(torch.load(save_path, map_location="cuda"))
#     model.eval()
#     with torch.no_grad():
#         enc = tokenizer(label_set, padding=True, truncation=True, return_tensors="pt")
#         label_embs = model.encode_label(enc["input_ids"].cuda(), enc["attention_mask"].cuda()).cpu()
#     test_dataset = TextLabelDataset(test_text, test_labels, tokenizer, 128)
#     test_loader = DataLoader(test_dataset, batch_size=128, shuffle=True)
#     all_test_embds = []
#     with torch.no_grad():
#         for batch in tqdm(test_loader):
#             input_ids = batch["text_input_ids"].cuda()
#             attention_mask = batch["text_attention_mask"].cuda()
#             text_embs = model.encode_text(input_ids, attention_mask)  # [B, D]
#             all_test_embds.append(text_embs.cpu())
#         all_test_embds = torch.cat(all_test_embds, dim=0)
#     train_dataset = TextLabelDataset(train_text, train_labels, tokenizer, 128)
#     train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
#     all_train_embds = []
#     with torch.no_grad():
#         for batch in tqdm(train_loader):
#             input_ids = batch["text_input_ids"].cuda()
#             attention_mask = batch["text_attention_mask"].cuda()
#             text_embs = model.encode_text(input_ids, attention_mask)  # [B, D]
#             all_train_embds.append(text_embs.cpu())
#         all_train_embds = torch.cat(all_train_embds, dim=0)
#     sim_matrix = util.cos_sim(all_test_embds, label_embs)  # [num_test, num_labels]
#     text_sim_matrix = util.cos_sim(all_train_embds, all_test_embds)  # [num_train, num_test]
#     # 10. top-k
#     results = []
#     for i in tqdm(range(len(test_text)), desc="Constructing results"):
#         # topk 标签
#         sim_row = sim_matrix[i]  # 当前测试样本与所有标签的相似度
#         topk_vals, topk_indices = torch.topk(sim_row, k=topk)
#         topk_labels = [label_set[idx] for idx in topk_indices]
#         # overall_topk: topk训练样本id
#         sims = text_sim_matrix[:, i]
#         topk_train_vals, topk_train_indices = torch.topk(sims, k=topk)
#         overall_topk = topk_train_indices.tolist()
#         # per_class_topk: 每个类别下与测试样本最相似的训练样本id
#         per_class_topk = {}
#         for lbl in label_set:
#             train_ids_of_class = [j for j, l in enumerate(train_labels) if l == lbl]
#             if not train_ids_of_class:
#                 continue
#             class_sims = sims[train_ids_of_class]
#             class_topk_values, class_topk_indices = torch.topk(class_sims, k=min(topk, len(class_sims)))
#             per_class_topk[lbl] = [
#                 {"train_id": train_ids_of_class[idx], "score": float(score)}
#                 for idx, score in zip(class_topk_indices.tolist(), class_topk_values.tolist())
#             ]
#         results.append({
#             "query_id": i,
#             "query_label": test_labels[i],
#             "topk": topk_labels,
#             "overall_topk": overall_topk,
#             "per_class_topk": per_class_topk
#         })
#     save_path = f'dataset/retriever_results/bert_{dataname}.json'
#     with open(save_path, 'w', encoding='utf-8') as f:
#         json.dump(results, f, ensure_ascii=False, indent=2)
#     return results

def retriever_tree(cfg, nchild=3):
    ''' 根据测试样本retriever_label的结果自动聚类并返回树结构，树被用于后续的标签自底向上的层次聚类
    :param cfg:
    :return:
    '''
    save_path = 'dataset/retriever_results/gcn_{}.json'.format(cfg.test)
    with open(save_path, "r", encoding="utf-8") as f:
        gcn_topks = json.load(f)
    gcn_topks = [each['topk'] for each in gcn_topks]
    all_trees = label_cluster_for_each(cfg, gcn_topks, nchild)
    return all_trees


def get_prompt(cfg, dataname='Banking77'):
    train_text, test_text, train_labels, test_labels = load_train_test(dataname)
    instruct = 'Solve the classification task, options for answers: '
    # save_path = 'dataset/retriever_results/mmr_{}.json'.format(dataname)
    save_path = 'dataset/retriever_results/gcn_{}.json'.format(dataname)
    with open(save_path, "r", encoding="utf-8") as f:
        results = json.load(f)
    label_set = list(set(train_labels + test_labels))
    datas = []
    for each in tqdm(results, desc="Constructing prompts"):
        query = test_text[each["query_id"]]
        # ================================ FOR Both ==========================================
        # demo_labels = [each["per_class_topk"][l][0] for l in label_set]
        # demo_labels = [label for _, label in
        #                  sorted(zip(demo_labels, label_set), key=lambda x: x[0]['score'], reverse=True)]
        # demos = []
        # for l in demo_labels:
        #     demos.extend(each["per_class_topk"][l])
        # ================================ FOR DG ==============================================
        # demo_labels = [each["per_class_topk"][l][0] for l in label_set]
        # demo_labels = [label for _, label in
        #                sorted(zip(demo_labels, label_set), key=lambda x: x[0]['score'], reverse=True)][:5]
        # demos = each["overall_topk"][:cfg.shots]
        # ================================ FOR LG ================================================
        # demo_labels = [each["per_class_topk"][l][0] for l in label_set]
        # demo_labels = [label for _, label in
        #                sorted(zip(demo_labels, label_set), key=lambda x: x[0]['score'], reverse=True)][:cfg.shots]
        # demos = each["overall_topk"][:5]
        # ================================ FOR Normal ICL =========================================
        demos = each["overall_topk"][:cfg.shots]
        # print(each["overall_topk"])
        demo_labels = [train_labels[top] for top in each["overall_topk"]][:cfg.shots]
        unique_labels = list(set(demo_labels))
        prompt = instruct + ', '.join(unique_labels) + '.\n'
        for demo, label in zip(demos, demo_labels):
            prompt = prompt + 'Text: {} Answer: {}\n'.format(train_text[demo], label)
        prompt = prompt + 'Text: {} Answer:'.format(query)
        datas.append([prompt, test_labels[each["query_id"]]])
    return datas


def get_rank_label(cfg):
    '''  通过标签表征与query表征的相似性作为得分，执行Top-K based on cumulative probability，自动选择top标签
    :param cfg:
    :return:
    '''
    train_text, test_text, train_labels, test_labels = load_train_test(cfg.test)
    label_set = []
    for l in train_labels:
        if l not in label_set:
            label_set.append(l)
    G = build_heterogeneous_graph(cfg)
    sample_mask, train_mask, test_mask, word_mask, label_mask, node2idx = generate_masks(G)
    node_emb = torch.load("gcns/{}_node_embeddings_best.pth".format(cfg.test), map_location="cpu")
    sample_indices = torch.where(sample_mask)[0]
    assert sample_indices.shape[0] == len(train_labels) + len(test_labels)
    train_indices, test_indices = sample_indices[:len(train_text)], sample_indices[len(train_text):]
    # 测试文本 embedding
    train_emb, test_emb = node_emb[train_indices], node_emb[test_indices]  # shape: [num_test, dim]
    # 标签 embedding
    label_indices = torch.where(label_mask)[0]
    label_emb = node_emb[label_indices]  # shape: [num_labels, dim]
    save_path = 'dataset/retriever_results/gcn_{}.json'.format(cfg.test)
    with open(save_path, "r", encoding="utf-8") as f:
        results = json.load(f)
    cand_labels = [each['topk'] for each in results]  # 从候选标签里进一步提取
    test_label_sim = cos_sim(test_emb, label_emb)
    ks = []
    for i, cand in enumerate(cand_labels):
        cand_idx = [label_set.index(c) for c in cand]
        sims = test_label_sim[i, cand_idx]  # 保留原顺序
        probs = torch.softmax(sims, dim=0)
        cumulative = torch.cumsum(probs, dim=0)
        k = torch.searchsorted(cumulative, 0.4).item() + 1
        ks.append(cand[:k])
    return ks


def gap_selection_smooth(sim_scores, window=3, min_k=1, max_k=None):
    """
    平滑 gap 方法选择 top-k
    sim_scores: list 或 tensor, 已降序
    window: 平滑窗口大小
    """
    # 转 tensor
    if not isinstance(sim_scores, torch.Tensor):
        sim_scores = torch.tensor(sim_scores, dtype=torch.float32)
    # 原始差值
    diffs = sim_scores[:-1] - sim_scores[1:]
    # 平滑卷积
    kernel = torch.ones(window) / window
    diffs_padded = torch.nn.functional.pad(diffs, (0, window - 1))  # 补齐
    diffs_smooth = torch.nn.functional.conv1d(diffs_padded.view(1, 1, -1), kernel.view(1, 1, -1)).view(-1)
    diffs_smooth = diffs_smooth[:len(diffs)]  # 恢复长度
    # 找最大平滑 gap
    k = torch.argmax(diffs_smooth).item() + 1
    # 约束
    if max_k is not None:
        k = min(k, max_k)
    k = max(k, min_k)
    return k

def get_inf_prompt(cfg):
    train_text, test_text, train_labels, test_labels = load_train_test(cfg.test)
    labels = []
    for each in train_labels:
        if each not in labels:
            labels.append(each)
    encoder = SentenceTransformer(cfg.model_source + '/all-MiniLM-L6-v2').cuda()
    instruct = 'Solve the classification task, options for answers: '
    save_path = 'dataset/retriever_results/knn_{}.json'.format(cfg.test)
    model = MLPClassifier(384, 256, len(labels), num_layers=2, dropout=0.1).cuda()
    model.load_state_dict(torch.load("gcns/{}_gcn_best_model.pth".format(cfg.test)))
    with open(save_path, "r", encoding="utf-8") as f:
        topks = json.load(f)
    all_prompts = []
    for i in tqdm(range(len(test_text)), desc="Constructing prompts"):
        candidates = []
        top_label = []
        for each in topks[i]["overall_topk"]:
            top_label.append(each["train_label"])
        top_label = set(top_label)
        for k in top_label:
            candidates.extend(topks[i]["per_class_topk"][k])
        candidates = sorted(candidates, key=lambda d: d["score"], reverse=True)[:8]
        cand_texts = [train_text[each["train_id"]] for each in candidates]
        cand_labels = [labels.index(train_labels[each["train_id"]]) for each in candidates]
        x_t, x_t_lab = test_text[i], test_labels[i]
        inf_values = []
        for j in range(0, len(cand_texts)):   # 贪心算法计算影响函数，得到的是有益样本的组合
            inf_value = calc_influence_set(cfg, encoder, model, cand_texts[j],  cand_labels[j], x_t, labels.index(x_t_lab))
            inf_values.append(inf_value)
        inf_values = np.array(inf_values)  # 确保是 np.array
        # 得到排序索引（从小到大）
        sorted_idx = np.argsort(inf_values)[:cfg.shots]
        # 随机打乱
        sorted_idx = np.random.permutation(sorted_idx)
        cand_texts = np.array(cand_texts)[sorted_idx].tolist()
        cand_labels = np.array(cand_labels)[sorted_idx].tolist()
        cand_labels = [labels[l] for l in cand_labels]
        unique_labels = set(top_label)
        prompt = instruct + ', '.join(unique_labels) + '.\n'
        for text, label in zip(cand_texts, cand_labels):
            prompt = prompt + 'Text: {} Answer: {}\n'.format(text, label)
        prompt = prompt + 'Text: {} Answer:'.format(x_t)
        # print(prompt)
        all_prompts.append([prompt, x_t_lab])
    return all_prompts


def get_rank_prompt(cfg):
    train_text, test_text, train_labels, test_labels = load_train_test(cfg.test)
    instruct = 'Solve the classification task, options for answers: '
    save_path = 'dataset/retriever_results/knn_{}.json'.format(cfg.test)
    with open(save_path, "r", encoding="utf-8") as f:
        topks = json.load(f)
    ks = get_rank_label(cfg)
    all_prompts = []
    for i in tqdm(range(len(test_text)), desc="Constructing prompts"):
        candidates = []
        for k in ks[i]:
            candidates.extend(topks[i]["per_class_topk"][k])
        candidates = sorted(candidates, key=lambda d: d["score"], reverse=True)
        candidate_ids = [c['train_id'] for c in candidates]
        scores = [c['score'] for c in candidates]
        selected = gap_selection_smooth(scores, min_k=1, max_k=len(candidates))
        selected = min(selected, cfg.shots)
        candidate_ids = candidate_ids[:selected]
        unique_labels = set([train_labels[each] for each in candidate_ids])
        prompt = instruct + ', '.join(unique_labels) + '.\n'
        for id in candidate_ids:
            prompt = prompt + 'Text: {} Answer: {}\n'.format(train_text[id], train_labels[id])
        prompt = prompt + 'Text: {} Answer:'.format(test_text[topks[i]["query_id"]])
        all_prompts.append([prompt, test_labels[i]])
    return all_prompts

def get_pair_prompt(cfg, all_pairs):
    ''' 根据输入的pair构建ICL的全部prompt
    :param cfg:
    :return:
    '''
    instruct = 'Solve the classification task, options for answers: '
    train_text, test_text, train_labels, test_labels = load_train_test(cfg.test)
    save_path = 'dataset/retriever_results/knn_{}.json'.format(cfg.test)
    # save_path = 'dataset/retriever_results/gcn_{}.json'.format(cfg.test)
    with open(save_path, "r", encoding="utf-8") as f:
        topks = json.load(f)
    # assert len(all_pairs) == len(topks) == len(test_labels)
    all_prompts = []
    for pairs, topk in zip(all_pairs, topks):
        prompts = []
        for pair in pairs:
            candidate_labels = [pair[i][0] for i in range(len(pair))]    # 根据canditate labels构建样本
            demos = []
            for candidate_label in candidate_labels:
                demos.extend(topk["per_class_topk"][candidate_label])
            demos = sorted(demos, key=lambda d: d["score"], reverse=True)[:cfg.shots]
            unique_labels = [train_labels[each["train_id"]] for each in demos]
            prompt = instruct + ', '.join(set(unique_labels)) + '.\n'
            for demo in demos:
                id = demo['train_id']
                prompt = prompt + 'Text: {} Answer: {}\n'.format(train_text[id], train_labels[id])
            prompt = prompt + 'Text: {} Answer:'.format(test_text[topk["query_id"]])
            prompts.append(prompt)
        all_prompts.append(prompts)
    return all_prompts


def get_competition(cfg, train_text, test_text, train_labels, test_labels, last_raw_pairs=None):
    ''' 根据测试样本解析tree结构并获取竞争关系
    :param cfg:
    :return:
    '''
    # train_text, test_text, train_labels, test_labels = load_train_test(cfg.test)
    label_set = set(np.concatenate([train_labels, test_labels]))
    all_pairs, all_raw_pairs = [], []
    if not last_raw_pairs:
        save_path = f"trees/{cfg.test}_label_trees_{cfg.candidates}.json"
        with open(save_path, "r", encoding="utf-8") as f:
            trees = json.load(f)
        for id in tqdm(trees.keys(), desc="Constructing pairs"):
            tree = trees[id]
            groups = get_sibling_groups_with_parent(tree)
            real_pairs = [each for each in groups if all(item[0] in label_set for item in each)]
            all_pairs.append(real_pairs)
            all_raw_pairs.append(groups)
    else:
        for pairs in last_raw_pairs:
            real_pairs = [each for each in pairs if all(item[0] in label_set for item in each)]
            all_pairs.append(real_pairs)
            all_raw_pairs.append(pairs)

    return all_pairs, all_raw_pairs


def batchify(datas, batch_size):
    batches = []
    for i in range(0, len(datas), batch_size):
        batches.append(datas[i:i + batch_size])
    return batches


def load_LLMs(cfg):
    """
    加载大语言模型并转换为半精度（FP16）并支持指定的多卡加载。
    :param cfg: 配置对象，包含与模型相关的配置。
    :param device_ids: 用于指定多卡的设备ID列表。默认值为 [1, 2, 3]，即使用第1,2,3号GPU。
    :return: 返回加载的模型和tokenizer
    """
    # 加载预训练模型和 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_source + '/' + cfg.model_name, clean_up_tokenization_spaces=True)
    # 设置 padding token 为 eos token，并确保 padding 方向为 left
    tokenizer.pad_token = tokenizer.eos_token  # 设置 padding_token 为 eos_token
    tokenizer.padding_side = 'left'  # 设置 padding 方向为左侧
    # 加载模型
    model = AutoModelForCausalLM.from_pretrained(cfg.model_source + '/' + cfg.model_name).cuda().half()
    print(f"Model loaded")
    return model, tokenizer


if __name__ == '__main__':
    cfg = Config.Config()
    for data in ['TacRED']:
        cfg.test = data
        retriever_gcn(cfg, dataname=data)
        for ccc in [2, 4, 8]:
            cfg.candidates = ccc
            retriever_tree(cfg, nchild=2)
        # all_pairs = get_competition(cfg)
        # get_pair_prompt(cfg, all_pairs)
        # cfg.test = data
        # get_tree_prompt(cfg, dataname=data)
