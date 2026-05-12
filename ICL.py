import time

import Config
from dataloader import *
from sklearn.metrics import accuracy_score
from tqdm import tqdm

def ICL(cfg, datas, tree_search=True):
    start = time.time()
    # datas = get_rank_prompt(cfg)
    batches = batchify(datas, cfg.batch_size)
    batches = batches
    LLM, tokenizer = load_LLMs(cfg)
    correct_predictions, total_predictions = 0, 0
    # 上下文学习过程
    for batch in tqdm(batches, desc="Evaluating ICL"):
        inputs = []
        labels = []
        # 准备输入和标签
        for prompt, label in batch:
            inputs.append(prompt)
            labels.append(label)
            # 通过模型生成回答
        device = LLM.device  # 单卡情况下直接使用 model.device
        encodings = tokenizer(inputs, return_tensors="pt", padding=True, truncation=True).to(device)
        outputs = LLM.generate(
            input_ids=encodings['input_ids'],
            attention_mask=encodings['attention_mask'],
            max_new_tokens = 8,
            num_return_sequences=1,
            pad_token_id=tokenizer.pad_token_id
        )
        # 解析生成的结果
        generated_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True, pad_token_id=tokenizer.pad_token_id)
        for i in range(len(generated_texts)):
            predicted_label = generated_texts[i].strip().split('Answer: ')[-1]
            true_label = labels[i]
            # print('--------------------------------------')
            # print(inputs[i], predicted_label, true_label)
            if true_label in predicted_label:
                correct_predictions += 1
            total_predictions += 1
    accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
    end = time.time()
    print(f"ICL Accuracy: {accuracy*100:.2f} with time: {(end-start)}" )

def update_pairs_with_winners(all_pairs, all_winners):
    """
    根据每轮竞争胜利者更新 all_pairs：
      - 父节点标签替换为胜者标签
      - 原有子节点移除

    :param all_pairs: 原始 pairs，形状 [num_batch, batch_size, 2] 或类似结构
    :param all_winners: 每轮对应胜者，形状与 batch 一致
    :return: 更新后的 all_pairs
    """
    updated_all_pairs = []
    for pairs, winners in zip(all_pairs, all_winners):
        replace = {}
        updated_pairs = []
        # Step 1: 找出已经有胜出者的父节点
        for group in pairs:
            # group 形如: [('Cluster_0','Cluster_3'), ('Cluster_1','Cluster_3'), ...]
            for label, parent in group:
                if label in winners:
                    replace[parent] = label
                    break  # 一个父节点只需要一个 winner
        # Step 2: 生成新的 pairs
        for group in pairs:
            new_group = []
            if any(label in winners for label, _ in group):   # 竞争组跳过
                continue
                # 替换为胜出的 label（如果父节点已映射）
            for label, parent in group:
                # 替换为胜出的 label（如果父节点已映射）
                if label in replace:
                    label = replace[label]
                new_group.append((label, parent))
            updated_pairs.append(new_group)
        updated_all_pairs.append(updated_pairs)
    return updated_all_pairs


def competite(cfg, all_pairs=None, all_raw_pairs=None):
    """
    对给出的 pairs 进行竞争以获取预测结果，使用批量化 LLM 生成优化速度。
    :param cfg: 配置对象
    :param all_pairs: list of pairs，每个 pair 是 [(label1, parent), (label2, parent)]
    :param all_raw_pairs: 原始 pairs，可选
    :return: all_raw_pairs, all_winners
    """
    datas = get_pair_prompt(cfg, all_pairs)
    batches = batchify(datas, cfg.batch_size)
    all_pairs_batch = batchify(all_pairs, cfg.batch_size)
    LLM, tokenizer = load_LLMs(cfg)
    device = LLM.device
    # 可选：fp16 加速
    all_winners = []
    for batch, pairs in tqdm(zip(batches, all_pairs_batch), total=len(batches), desc="Evaluating ICL"):
        # 每列是一个 pair 的文本
        cols = list(zip(*batch))  # batch_size x pair_num -> pair_num x batch_size
        pair_cols = list(zip(*pairs))  # 对应标签对
        # 扁平化文本列用于一次性生成
        flat_cols = [text for col in cols for text in col]  # len = batch_size * pair_num
        encodings = tokenizer(flat_cols, return_tensors="pt", padding=True, truncation=True).to(device)
        # 一次性生成所有文本
        outputs = LLM.generate(
            input_ids=encodings['input_ids'],
            attention_mask=encodings['attention_mask'],
            max_new_tokens=8,
            num_return_sequences=1,
            pad_token_id=tokenizer.pad_token_id
        )
        generated_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True, pad_token_id=tokenizer.pad_token_id)
        # 重新 reshape 回 batch_size x pair_num
        batch_size = len(batch)
        pair_num = len(pair_cols)
        generated_texts_reshaped = [generated_texts[i * batch_size:(i + 1) * batch_size] for i in range(pair_num)]
        # 解析胜者
        col_winners = []
        for col_idx, (col_texts, pair) in enumerate(zip(generated_texts_reshaped, pair_cols)):
            winners = []
            for row_idx in range(len(col_texts)):
                original_text = cols[col_idx][row_idx]  # 对应原始文本
                predicted_label = col_texts[row_idx].split('Answer:')[-1].split('\n')[0].strip()
                # print(original_text)
                # print('====================')
                # print(predicted_label)
                candidates = [pair[row_idx][ppp][0] for ppp in range(len(pair[row_idx]))]
                # 优先选择 predicted_label 中出现的标签
                win_label = None
                for lbl in candidates:
                    if lbl in predicted_label:
                        win_label = lbl
                        break
                # 如果没有匹配，则随机选择一个
                if win_label is None:
                    win_label = random.choice(candidates)
                winners.append(win_label)
            col_winners.append(winners)
        # 转置回原 batch 的 shape
        batch_winners = list(map(list, zip(*col_winners)))  # 每行对应原 batch 的一行
        # print(batch_winners)
        all_winners.extend(batch_winners)

    return all_raw_pairs, all_winners

def ICL_compete(cfg):
    train_text, test_text, train_labels, test_labels = load_train_test(cfg.test)
    all_pairs, all_raw_pairs = get_competition(cfg, train_text, test_text, train_labels, test_labels)
    while len(all_pairs[0]) >= 1:    # 说明仍有可比较的pair
        all_raw_pairs, all_winners = competite(cfg, all_pairs, all_raw_pairs)
        all_raw_pairs = update_pairs_with_winners(all_raw_pairs, all_winners)
        all_pairs, all_raw_pairs = get_competition(cfg, train_text, test_text, train_labels, test_labels, all_raw_pairs)
    all_winners = [w[0] for w in all_winners]
    acc = accuracy_score(test_labels[:len(all_winners)], all_winners)
    print(f"Accuracy: {acc*100:.2f}% ")



if __name__ == '__main__':
    # for data in ['GoEmotions']:
    for data in ['HWU64', 'GoEmotions', 'HuffPost15', 'TacRED', 'Banking77', 'Clinc150']:
        for model_name in ['llama3.2-1b', 'qwen3-1.7b']:
            cfg = Config.Config()
            cfg.test = data
            datas = get_inf_prompt(cfg)
            datas = get_prompt(cfg, cfg.test)
            for shot in [4]:
                cfg.model_name = model_name
                cfg.shots = shot
                ICL(cfg, datas, tree_search=False)
                cfg.candidates = 8
                ICL_compete(cfg)
            all_pairs, all_winners = competite(cfg)
            print(all_pairs[0], all_winners[0])
            # 计算winners中包含真实标签的概率
            # update_pairs_with_winners(all_pairs, all_winners)