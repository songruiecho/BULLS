import pandas as pd
from sklearn.model_selection import train_test_split
import json
from collections import Counter
import os
import ast  # 用于安全解析字符串为 Python 列表

# 第一步拆分测试训练集
def split_HWU64():
    frame = pd.read_csv(
        'HWU64/NLU-Data-Home-Domain-Annotated-All.csv',
        quotechar='"',
        on_bad_lines='skip',
        sep=';'
    )

    # 去掉列名首尾空格
    frame.columns = frame.columns.str.strip()
    # 选择需要的列
    frame = frame[['scenario', 'intent', 'question', 'answer']].fillna('')

    # 合并列
    frame['label'] = frame['scenario'].astype(str) + ' ' + frame['intent'].astype(str)
    frame['text'] = frame['question'].astype(str) + ' ' + frame['answer'].astype(str)

    # 保留最终两列
    frame = frame[['label', 'text']]

    train_list = []
    test_list = []

    # 按 label 分组
    for lbl, group in frame.groupby('label'):
        # 如果某个 label 样本数太少，可能需要特殊处理
        train, test = train_test_split(
            group,
            test_size=0.1,  # 10% 作为测试集
            random_state=42,  # 固定随机种子保证可复现
            shuffle=True
        )
        train_list.append(train)
        test_list.append(test)

    # 合并所有 label 的训练集和测试集
    train_df = pd.concat(train_list).reset_index(drop=True)
    test_df = pd.concat(test_list).reset_index(drop=True)

    # 查看样本数
    print("训练集样本数:", len(train_df))
    print("测试集样本数:", len(test_df))
    print("训练集每个 label 样本数:\n", train_df['label'].value_counts())
    print("测试集每个 label 样本数:\n", test_df['label'].value_counts())

    # 保存训练集
    train_df.to_csv('HWU64/train.csv', index=False, encoding='utf-8')
    # 保存测试集
    test_df.to_csv('HWU64/test.csv', index=False, encoding='utf-8')



def split_Clink150():
    # 假设文件名是 data.json
    with open("Clinc150/data_full.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    train = data['train']
    test = data['test']

    print(len(train), len(test))

    # 转换为 DataFrame
    df_train = pd.DataFrame(train, columns=["text", "label"])
    df_test = pd.DataFrame(test, columns=["text", "label"])

    print(df_train.shape, df_test.shape)

    # 保存为 TSV 文件
    df_train.to_csv("Clinc150/train.tsv", sep="\t", index=False)
    df_test.to_csv("Clinc150/test.tsv", sep="\t", index=False)


def split_GoEmotions():

    with open("GoEmotions/emotions.txt", "r", encoding="utf-8") as f:
        emotions = [each.strip() for each in f.readlines()]

    train_df = pd.read_csv("GoEmotions/train.tsv", sep="\t", names=["text", "label", "C3"])
    test_df = pd.read_csv("GoEmotions/test.tsv", sep="\t", names=["text", "label", "C3"])

    # 定义过滤函数
    def filter_single_label(df):
        df = df.copy()
        df["label"] = df["label"].astype(str).str.split(",")
        # 将 label 的数值映射为 emotions 中的对应值
        return df[df["label"].str.len() == 1]

    # 过滤
    train_single = filter_single_label(train_df)
    train_single["label"] = train_single["label"].apply(lambda x: emotions[int(x[0])])
    test_single = filter_single_label(test_df)
    test_single["label"] = test_single["label"].apply(lambda x: emotions[int(x[0])])
    # 重新保存
    train_single.to_csv("GoEmotions/train.csv", sep="\t", index=False)
    test_single.to_csv("GoEmotions/test.csv", sep="\t", index=False)

# split_GoEmotions()


def split_HuffPost15():
    # 假设文件名为 data.jsonl
    input_file = "HuffPost15/News_Category_Dataset_v3.json"
    # 1. 读取数据
    records = []
    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line.strip())
            text = obj["headline"].strip().lower()  # headline -> text, 转小写
            label = obj["category"].strip().lower()  # category -> label, 转小写
            records.append({"text": text, "label": label})
    df = pd.DataFrame(records).dropna()
    # 2. 统计各类别数量，确定最小类别的样本数
    counts = Counter(df["label"])
    min_count = min(counts.values())
    # 3. 按类别均等采样
    balanced_dfs = []
    for label, count in counts.items():
        subset = df[df["label"] == label].sample(n=min_count, random_state=42)
        balanced_dfs.append(subset)
    balanced_df = pd.concat(balanced_dfs).reset_index(drop=True)
    # 4. 划分训练集和测试集（9:1）
    train_df, test_df = train_test_split(balanced_df, test_size=0.1, stratify=balanced_df["label"], random_state=42)
    print(train_df.shape, test_df.shape)
    # 5. 保存为 CSV
    train_df.dropna().to_csv("HuffPost15/train.csv", index=False)
    test_df.dropna().to_csv("HuffPost15/test.csv", index=False)


# split_HuffPost15()

def split_Reuters():
    input_dir = 'Reuters21578/'
    # 找到目录下所有的 csv 文件
    csv_files = [f for f in os.listdir(input_dir) if f.endswith(".csv")]
    # 逐个读取并存入列表
    dfs = []
    for f in csv_files:
        file_path = os.path.join(input_dir, f)
        df = pd.read_csv(file_path)
        dfs.append(df)
    # 合并为一个大的 DataFrame
    df = pd.concat(dfs, ignore_index=True).dropna()
    # 如果 topics 是字符串（例如 "['cocoa']"），需要转成真正的 list
    df["topics"] = df["topics"].apply(ast.literal_eval)
    # 筛选：topics 列长度等于 1
    filtered_df = df[df["topics"].apply(lambda x: isinstance(x, list) and len(x) == 1)]
    # 展开所有单标签 topic
    single_topics = [topics[0] for topics in filtered_df["topics"]]
    # 统计出现频次
    topic_counts = Counter(single_topics)
    # 保留出现次数 >= 15 的 topic
    valid_topics = {topic for topic, count in topic_counts.items() if count >= 15}
    # 在 filtered_df 中只保留这些 topic 的行
    final_df = filtered_df[filtered_df["topics"].apply(lambda x: x[0] in valid_topics)]
    data = final_df[["topics", "title"]].copy()
    # 由于 topics 列是列表，取第一个元素作为标签
    data["topics"] = data["topics"].apply(lambda x: ' '.join(x[0].split('-')))
    data["title"] = data["title"].apply(lambda x: x.lower())
    data = data.dropna(subset=["topics", "title"])
    # 分层拆分：按 topic_label 保证每个标签在 train/test 中比例相同
    train_df, test_df = train_test_split(
        data,
        test_size=0.1,
        stratify=data["topics"],
        random_state=42
    )
    print(train_df.shape, test_df.shape)
    # 保存训练集
    train_df.rename(columns={"title": "text", "topics": "label"})[["text", "label"]].to_csv(
        "Reuters21578/train.csv", index=False, encoding="utf-8"
    )
    # 保存测试集
    test_df.rename(columns={"title": "text", "topics": "label"})[["text", "label"]].to_csv(
        "Reuters21578/test.csv", index=False, encoding="utf-8"
    )


def split_Discover():
    labels = ["no_conn", "absolutely", "accordingly", "actually", "additionally", "admittedly", "afterward", "again",
              "already", "also", "alternately", "alternatively", "although", "altogether", "amazingly", "and", "anyway",
              "apparently", "arguably", "as_a_result", "basically", "because_of_that", "because_of_this", "besides",
              "but", "by_comparison", "by_contrast", "by_doing_this", "by_then", "certainly", "clearly",
              "coincidentally", "collectively", "consequently", "conversely", "curiously", "currently", "elsewhere",
              "especially", "essentially", "eventually", "evidently", "finally", "first", "firstly", "for_example",
              "for_instance", "fortunately", "frankly", "frequently", "further", "furthermore", "generally",
              "gradually", "happily", "hence", "here", "historically", "honestly", "hopefully", "however", "ideally",
              "immediately", "importantly", "in_contrast", "in_fact", "in_other_words", "in_particular", "in_short",
              "in_sum", "in_the_end", "in_the_meantime", "in_turn", "incidentally", "increasingly", "indeed",
              "inevitably", "initially", "instead", "interestingly", "ironically", "lastly", "lately", "later",
              "likewise", "locally", "luckily", "maybe", "meaning", "meantime", "meanwhile", "moreover", "mostly",
              "namely", "nationally", "naturally", "nevertheless", "next", "nonetheless", "normally", "notably", "now",
              "obviously", "occasionally", "oddly", "often", "on_the_contrary", "on_the_other_hand", "once", "only",
              "optionally", "or", "originally", "otherwise", "overall", "particularly", "perhaps", "personally", "plus",
              "preferably", "presently", "presumably", "previously", "probably", "rather", "realistically", "really",
              "recently", "regardless", "remarkably", "sadly", "second", "secondly", "separately", "seriously",
              "significantly", "similarly", "simultaneously", "slowly", "so", "sometimes", "soon", "specifically",
              "still", "strangely", "subsequently", "suddenly", "supposedly", "surely", "surprisingly", "technically",
              "thankfully", "then", "theoretically", "thereafter", "thereby", "therefore", "third", "thirdly", "this",
              "though", "thus", "together", "traditionally", "truly", "truthfully", "typically", "ultimately",
              "undoubtedly", "unfortunately", "unsurprisingly", "usually", "well", "yet"]
    df_train = pd.read_parquet("Discover/train-00000-of-00001-159135b13a3ccd61.parquet")
    df_test = pd.read_parquet("Discover/test-00000-of-00001-cbca28caeaea3900.parquet")
    # Index(['sentence1', 'sentence2', 'label', 'idx'], dtype='object') (87000, 4)
    train_counts = df_train['label'].value_counts()
    test_counts = df_test['label'].value_counts()
    df_train = df_test.groupby('label', group_keys=False).apply(lambda x: x.sample(n=min(len(x), 20), random_state=42))
    df_test = df_test.groupby('label', group_keys=False).apply(lambda x: x.sample(n=min(len(x), 100), random_state=42))
    # 将sentence1 和 sentence2用 \t拼接然后把label换成labels对应的字符串，并写入train.csv 和 test.csv，包含text 和 label两列
    # 将 label id 转换为字符串
    df_train['label'] = df_train['label'].apply(lambda x: labels[x])
    df_test['label'] = df_test['label'].apply(lambda x: labels[x])
    # 拼接 sentence1 和 sentence2
    df_train['text'] = df_train['sentence1'] + '\t' + df_train['sentence2']
    df_test['text'] = df_test['sentence1'] + '\t' + df_test['sentence2']
    # 只保留 text 和 label 两列
    df_train_out = df_train[['text', 'label']]
    df_test_out = df_test[['text', 'label']]
    # 写入 CSV
    df_train_out.to_csv("train.csv", index=False, encoding='utf-8')
    df_test_out.to_csv("test.csv", index=False, encoding='utf-8')
    print("train.csv 和 test.csv 已生成")


def split_TacRED():

    def filter_and_create_df(records, min_count=20):
        # 转换为 DataFrame
        df = pd.DataFrame(records)
        # 删除 no_relation
        df = df[df['label'] != "no_relation"]
        # 保留数量 >= min_count 的标签
        label_counts = df['label'].value_counts()
        valid_labels = label_counts[label_counts >= min_count].index
        df = df[df['label'].isin(valid_labels)]
        return df

    with open("TacRED/train.json", 'r') as train:
        train_data = json.load(train)
    with open("TacRED/test.json", 'r') as test:
        test_data = json.load(test)
    train_records, test_records = [], []
    for data in train_data:
        tokens = data['token']
        label = data['relation'].split(':')[-1]
        subj_tokens = tokens[data['subj_start']:data['subj_end']+1]
        obj_tokens = tokens[data['obj_start']:data['obj_end']+1]
        subj = ' '.join(subj_tokens)
        obj = ' '.join(obj_tokens)
        text_tokens = ' '.join(tokens)
        # text 字段由 subj, obj, text 用 \t 拼接
        text_field = f"{subj}\t{obj}\t{text_tokens}"
        train_records.append({'text': text_field, 'label': label})
    for data in test_data:
        tokens = data['token']
        label = data['relation'].split(':')[-1]
        subj_tokens = tokens[data['subj_start']:data['subj_end']+1]
        obj_tokens = tokens[data['obj_start']:data['obj_end']+1]
        subj = ' '.join(subj_tokens)
        obj = ' '.join(obj_tokens)
        text_tokens = ' '.join(tokens)
        # text 字段由 subj, obj, text 用 \t 拼接
        text_field = f"{subj}\t{obj}\t{text_tokens}"
        test_records.append({'text': text_field, 'label': label})

    df_train = filter_and_create_df(train_records, min_count=20)
    df_test = filter_and_create_df(test_records, min_count=2)
    # 取出两个 DataFrame 的标签集合
    train_labels = set(df_train['label'].unique())
    test_labels = set(df_test['label'].unique())

    # 交集
    common_labels = train_labels & test_labels
    print("重叠的标签数量:", len(common_labels))
    print("重叠的标签:", common_labels)

    # 如果你想只保留交集里的样本
    df_train = df_train[df_train['label'].isin(common_labels)]
    df_test = df_test[df_test['label'].isin(common_labels)]
    # 写入 CSV
    df_train.to_csv("TacRED/train.csv", index=False, encoding='utf-8')
    df_test.to_csv("TacRED/test.csv", index=False, encoding='utf-8')
    print("train.csv 和 test.csv 已生成")
