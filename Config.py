
class Config:
    model_source = '/home/chenhechang/models/'
    shots = 2
    batch_size = 64
    datasets = ['Banking77', 'Clinc150', 'HWU64', 'GoEmotions', 'HuffPost15', 'TacRED']
    test = 'GoEmotions'
    candidates = 9      # 初始的候选标签空间，用于在这个空间内进行树构建
    leaves = 3          # k叉树的节点
    # model_name = 'gpt-j-6b'
    # model_name = 'llama3.2-1b'
    # model_name = 'llama3.2-3b'
    model_name = 'qwen3-1.7b'