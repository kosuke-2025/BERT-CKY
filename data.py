from nltk import Tree
from collections import defaultdict
import random

def read_and_binarize(file_names):
    all_lines = []
    for fn in file_names:
        with open(fn, 'r') as f:
            lines = f.readlines()
            all_lines.extend(lines)

    def right_binarize(tree):
        if isinstance(tree, Tree):
            if len(tree) <= 2:
                return Tree(tree.label(), [right_binarize(child) for child in tree])
            else:
                first = right_binarize(tree[0])
                rest = right_binarize(Tree(tree.label() + '*', tree[1:]))
                return Tree(tree.label(), [first, rest])
        else:
            return tree

    binarized_trees = [right_binarize(Tree.fromstring(line)) for line in all_lines]
    return binarized_trees



def build_label2id(trees):
    labels = set()
    for tree in trees:
        # tree.subtrees() は、木そのものと、その全ての部分木 (ノード) をイテレートします
        for subtree in tree.subtrees():
            # isinstance で nltk.Tree オブジェクトであることを確認 (葉の文字列は除外)
            if isinstance(subtree, Tree):
                labels.add(subtree.label()) # ノードのラベルを取得
        
        for subtree in tree.subtrees():
            # 現在のノードが、子が1つだけで、かつその子もTreeオブジェクト（非終端記号）であるかチェック
            if isinstance(subtree, Tree) and len(subtree) == 1 and isinstance(subtree[0], Tree):
                
                parent_label = subtree.label()
                child_node = subtree[0]
                child_label = child_node.label()
                
                # "子ラベル-親ラベル" の形式で複合ラベルを作成
                # 例: (NP (PRP He)) -> 'PRP-NP'
                composite_label = f"{child_label}-{parent_label}"
                labels.add(composite_label)
    
    # sortedにlist()を挟むのは古いPythonバージョンでのsetの順序不定性対策ですが、
    # sorted自体がリストを返すので、labelsを直接sortedしても問題ありません。
    return {label: idx for idx, label in enumerate(sorted(list(labels)))}



def prepare_input(tree, label2id, tokenizer):
    """
    構成素ツリーから、モデル入力用のエンコーディング、スパン、ラベルを準備します。
    全ての構成素（句および末端の品詞タグ）を対象とします。
    """
    if not tree or not tree.leaves(): # 空または葉のないツリーの場合
        dummy_encoding = tokenizer([], return_tensors="pt", is_split_into_words=True)
        return dummy_encoding, [], []

    tokens = tree.leaves()
    # is_split_into_words=True: トークンが既に単語単位で分割済みであることを示す
    # word_ids(): 各サブワードが元のtokensのどの単語に対応するかを示す
    encoding = tokenizer(tokens, return_tensors="pt", is_split_into_words=True, truncation=True, max_length=512)
    all_word_ids_map = encoding.word_ids()

    spans_data_tuples = []
    _get_constituent_spans_recursive(tree, 0, all_word_ids_map, label2id, spans_data_tuples)

    # 重複排除とソート（主にテスト時の再現性のため）
    unique_spans_data_tuples = sorted(list(set(spans_data_tuples)))
    
    converted_spans = [s_tuple[0] for s_tuple in unique_spans_data_tuples]
   
    span_labels = [s_tuple[1] for s_tuple in unique_spans_data_tuples]
    
    return encoding, converted_spans, span_labels


def _get_constituent_spans_recursive(tree_node, current_word_idx, all_word_ids_map, label2id, spans_data):
    """
    構成素ツリーを再帰的に探索し、各構成素のスパンとラベルIDを収集します。

    Args:
        tree_node (nltk.Tree): 現在処理中の構成素ノード。
        current_word_idx (int): このノードがカバーする最初の単語の、文全体でのインデックス。
        all_word_ids_map (list): サブワードトークンインデックスから単語インデックスへのマッピング。
        label2id (dict): ラベル文字列をIDに変換する辞書。
        spans_data (list): ( (start_subtoken, end_subtoken), label_id ) のタプルを格納するリスト。

    Returns:
        int: このノードがカバーする葉（単語）の数。
    """
    if not isinstance(tree_node, Tree):
        return 0

    node_label_str = tree_node.label()
    num_leaves_in_this_node = 0
    
    child_word_offset = 0
    for child in tree_node:
        if isinstance(child, Tree):
            leaves_in_child = _get_constituent_spans_recursive(
                child,
                current_word_idx + child_word_offset,
                all_word_ids_map,
                label2id,
                spans_data
            )
            num_leaves_in_this_node += leaves_in_child
            child_word_offset += leaves_in_child
        elif isinstance(child, str):
            num_leaves_in_this_node += 1
    
    if num_leaves_in_this_node == 0:
        return 0
    
    start_word_idx_for_node = current_word_idx
    end_word_idx_for_node = current_word_idx + num_leaves_in_this_node - 1

    try:
        node_start_subtoken_idx = -1
        for k, mapped_word_id in enumerate(all_word_ids_map):
            if mapped_word_id == start_word_idx_for_node:
                node_start_subtoken_idx = k
                break
        
        if node_start_subtoken_idx == -1: return num_leaves_in_this_node

        node_end_subtoken_idx_inclusive = -1
        for k_rev, mapped_word_id in reversed(list(enumerate(all_word_ids_map))):
            if mapped_word_id == end_word_idx_for_node:
                node_end_subtoken_idx_inclusive = k_rev
                break
        
        if node_end_subtoken_idx_inclusive == -1: return num_leaves_in_this_node

        # スパンは (inclusive_start_subtoken, exclusive_end_subtoken)
        subword_span = (node_start_subtoken_idx, node_end_subtoken_idx_inclusive + 1)
        
        if node_label_str in label2id:
            label_id = label2id[node_label_str]
            if subword_span[0] < subword_span[1]:
                spans_data.append((subword_span, label_id))
        
    except Exception: # サブトークンスパン計算中の予期せぬエラー
        pass

    return num_leaves_in_this_node



def build_rhs_to_lhs(trees):
    
    rhs_to_lhs = defaultdict(list)

    for tree in trees:
        for prod in tree.productions():
            if not prod.is_lexical():
                lhs = prod.lhs().symbol()
                rhs = tuple(sym.symbol() for sym in prod.rhs())
                if lhs not in rhs_to_lhs[rhs]:
                    rhs_to_lhs[rhs].append(lhs)

    return rhs_to_lhs



def split_data(binarized_trees):
    random.seed(42)
    random.shuffle(binarized_trees)

    n = len(binarized_trees)
    n_train = int(n * 0.8)
    n_val = int(n * 0.1)

    train_trees = binarized_trees[:n_train]
    val_trees = binarized_trees[n_train : n_train + n_val]
    test_trees = binarized_trees[n_train + n_val:]

    return train_trees, val_trees, test_trees



def debinarize(tree):
    """
    right_binarizeによって「右結合」で二項化された木を、
    元の多分木に正しく戻す（全ての入れ子構造に対応した最終確定版）。
    """
    # ベースケース: 葉（文字列）は、それ以上分解できないのでそのまま返す
    if not isinstance(tree, Tree):
        return tree

    # 現在のノードが、子が2つの木で、かつ右の子が「親ラベル+*」で始まる
    # 人工ノードであるかチェックする
    if len(tree) == 2 and isinstance(tree[1], Tree) and \
       tree[1].label().startswith(tree.label() + '*'):
        
        # このノードは二項化によって作られた中間ノード。
        # これを解消（unroll）する。
        
        # 1. 左の子（これは最終的な木に残る本物の要素）を取得
        left_child = tree[0]
        
        # 2. 右の子（人工ノード）を取得し、これをさらにdebinarizeする。
        #    これにより、(S* B (S** C)) のような入れ子が解消され、
        #    中身の [B, C] が展開された (S* B C) のような木が得られる。
        unrolled_right_node = debinarize(tree[1])
        
        # 3. 新しい子のリスト = [左の子] + [展開された右の子の中身]
        #    unrolled_right_nodeはTreeオブジェクトなので、[:]で子要素のリストを取得
        new_children = [left_child] + unrolled_right_node[:]
        
        # 4. 新しくできた子のリストを持つ木を、さらにdebinarizeする。
        #    これにより、(S A B C D) のような多段の入れ子も一度に解消される。
        return debinarize(Tree(tree.label(), new_children))

    else:
        # 上記のパターンに一致しないノード（葉、終端ノード、通常の多分木など）は、
        # その子要素をそれぞれ再帰的にdebinarizeして、木を再構築する
        return Tree(tree.label(), [debinarize(c) for c in tree])