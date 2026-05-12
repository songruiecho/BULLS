import copy
import json

def find_path_to_root(tree, target_name, path=None):
    """
    递归查找从根到目标节点的路径
    返回路径列表 [root,...,target]
    """
    if path is None:
        path = []
    path.append(tree)
    if tree["name"] == target_name:
        return path
    if "children" in tree:
        for child in tree["children"]:
            res = find_path_to_root(child, target_name, path.copy())
            if res:
                return res
    return None

def build_subtree_upwards(tree, target_name, k=3):
    """
    以 target_name 节点向上追溯 k 层，返回对应子树
    """
    path = find_path_to_root(tree, target_name)
    if path is None:
        raise ValueError(f"节点 {target_name} 不存在于树中")
    # 找到向上追溯 k 层的祖先
    ancestor_index = max(len(path) - 1 - k, 0)
    ancestor_node = path[ancestor_index]
    # 复制子树，避免修改原树
    return copy.deepcopy(ancestor_node)

def get_label2sub_trees(labels, tree, depth):
    label2sub_trees = {}
    for label in labels:
        label2sub_trees[label] = build_subtree_upwards(tree, label, depth)
    return label2sub_trees


def get_leaf_labels(node):
    """
    递归获取子树下所有叶子节点的名称
    """
    if "children" not in node or len(node["children"]) == 0:
        # 没有子节点，说明是叶子
        return [node["name"]]

    leaves = []
    for child in node["children"]:
        leaves.extend(get_leaf_labels(child))
    return leaves

def get_sibling_groups_with_parent(json_data):
    """
    从 JSON 树结构解析兄弟节点组（竞争组），每个子节点携带父节点信息
    支持 n 叉树

    :param json_data: str | dict
    :return: list of list，每个子 list 是一个兄弟节点组，包含 (child_name, parent_name)
    """
    if isinstance(json_data, str):
        json_data = json.loads(json_data)
    # 如果 JSON 外层有 test_id 包装，取第一个子树
    if isinstance(json_data, dict) and len(json_data) == 1 and isinstance(next(iter(json_data.values())), dict):
        json_data = next(iter(json_data.values()))
    sibling_groups = []
    def dfs(node):
        children = node.get("children", [])
        # 兄弟节点数量 >= 2 才算竞争组
        if len(children) >= 2:
            sibling_groups.append([(child["name"], node["name"]) for child in children])
        for child in children:
            dfs(child)
    dfs(json_data)
    return sibling_groups