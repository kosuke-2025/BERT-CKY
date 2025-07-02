from nltk import Tree
import torch
from torch.nn.functional import log_softmax

def parsing(tokens, model, id2label, tokenizer, rhs_to_lhs, device):
    """
    与えられたトークン列を構文解析し、最もスコアの高い構文木を返す。
    内部でCKYアルゴリズムとSpanScorerモデルを使用し、ユナリー規則も考慮する。
    """

    def cky_parse(tokens, model, id2label, tokenizer, rhs_to_lhs, device, top_k=5):
        """
        CKYアルゴリズムを実行し、解析チャートを返す。
        """
        n = len(tokens)
        if n == 0:
            return [[[] for _ in range(1)] for _ in range(1)]

        # ステップ1, 2, 3: スコアの事前計算
        encoding = tokenizer(tokens, return_tensors="pt", is_split_into_words=True, truncation=True, max_length=512)
        word_ids_list = encoding.word_ids()
        input_ids = encoding["input_ids"].to(device)
        attention_mask = encoding["attention_mask"].to(device)
        original_token_subword_indices = [(None, None)] * n
        current_word_idx = -1
        for i_subword, word_id in enumerate(word_ids_list):
            if word_id is not None:
                if word_id != current_word_idx:
                    current_word_idx = word_id
                    if current_word_idx < n:
                        original_token_subword_indices[current_word_idx] = (i_subword, i_subword)
                else:
                    if current_word_idx < n:
                        original_token_subword_indices[current_word_idx] = (
                            original_token_subword_indices[current_word_idx][0], i_subword
                        )
        all_chart_spans_info = [] 
        for length in range(1, n + 1):
            for i in range(n - length + 1):
                j = i + length
                start_token_idx = i
                end_token_idx = j - 1
                if original_token_subword_indices[start_token_idx][0] is not None and \
                   original_token_subword_indices[end_token_idx][1] is not None:
                    subword_start_inclusive = original_token_subword_indices[start_token_idx][0]
                    subword_end_inclusive = original_token_subword_indices[end_token_idx][1]
                    model_input_subword_span = (subword_start_inclusive, subword_end_inclusive + 1)
                    if model_input_subword_span[0] < model_input_subword_span[1] and \
                       model_input_subword_span[1] <= input_ids.shape[1]:
                        all_chart_spans_info.append(((i, j), model_input_subword_span))
        span_to_bert_log_score_map = {}
        if all_chart_spans_info:
            model_input_spans_list = [info[1] for info in all_chart_spans_info]
            with torch.no_grad():
                model.eval()
                logits_all_spans = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    spans=[model_input_spans_list] 
                )
                if logits_all_spans.ndim == 2 and logits_all_spans.shape[0] == len(model_input_spans_list):
                    log_probs_all_spans = log_softmax(logits_all_spans, dim=-1)
                    for k_span, (orig_span_ij, _) in enumerate(all_chart_spans_info):
                        span_log_probs = log_probs_all_spans[k_span]
                        scores_for_this_span_by_label_str = {}
                        for label_id, log_prob_tensor in enumerate(span_log_probs):
                            if label_id < len(id2label):
                                label_str = id2label[label_id]
                                scores_for_this_span_by_label_str[label_str] = log_prob_tensor.item()
                        span_to_bert_log_score_map[orig_span_ij] = scores_for_this_span_by_label_str
        
        # ステップ4: CKYチャートの初期化
        chart = [[[] for _ in range(n + 1)] for _ in range(n + 1)]

        # ステップ5: スパン長1の初期化 + ユナリー規則
        for i in range(n):
            j = i + 1
            if (i, j) in span_to_bert_log_score_map:
                initial_entries = []
                for label_str, log_score_val in span_to_bert_log_score_map[(i, j)].items():
                    initial_entries.append((log_score_val, label_str, None, tokens[i], None))
                
                queue = list(initial_entries)
                best_scores = {entry[1]: entry[0] for entry in queue}
                head = 0
                while head < len(queue):
                    log_score_B, label_B, _, _, _ = queue[head]
                    head += 1
                    rhs_unary = (label_B,)
                    if rhs_unary in rhs_to_lhs:
                        for label_A in rhs_to_lhs[rhs_unary]:
                            bert_log_score_for_A = span_to_bert_log_score_map.get((i, j), {}).get(label_A, -float('inf'))
                            
                            # 複合ラベル (B-A) のスコアも取得
                            composite_label = f"{label_B}-{label_A}"
                            composite_log_score = span_to_bert_log_score_map.get((i, j), {}).get(composite_label, 0.0) # 存在しない場合は影響0
                            
                            # 3つのスコアを合算
                            new_log_score = log_score_B + bert_log_score_for_A + composite_log_score
                            bert_log_score_for_A = span_to_bert_log_score_map.get((i, j), {}).get(label_A, -float('inf'))
                            new_log_score = log_score_B + bert_log_score_for_A

                            if new_log_score > best_scores.get(label_A, -float('inf')):
                                best_scores[label_A] = new_log_score
                                new_entry = (new_log_score, label_A, None, queue[head-1], None)
                                queue.append(new_entry)
                
                queue.sort(key=lambda x: x[0], reverse=True)
                chart[i][j] = queue[:top_k]

        # ステップ6 & 7: スパン長2以上の処理 + ユナリー規則
        for length in range(2, n + 1):
            for i in range(n - length + 1):
                j = i + length
                candidate_entries_for_cell = []
                for k_split in range(i + 1, j):
                    if not chart[i][k_split] or not chart[k_split][j]:
                        continue
                    for left_log_score, left_label, _, _, _ in chart[i][k_split]:
                        for right_log_score, right_label, _, _, _ in chart[k_split][j]:
                            rhs = (left_label, right_label)
                            if rhs in rhs_to_lhs:
                                rule_log_score = left_log_score + right_log_score
                                for lhs_label in rhs_to_lhs[rhs]:
                                    bert_span_log_score_for_lhs = span_to_bert_log_score_map.get((i,j), {}).get(lhs_label, -float('inf'))
                                    combined_log_score = rule_log_score + bert_span_log_score_for_lhs
                                    candidate_entries_for_cell.append((
                                        combined_log_score, lhs_label, k_split,
                                        (i, k_split, left_label), (k_split, j, right_label)
                                    ))
                
                queue = list(candidate_entries_for_cell)
                best_scores = {entry[1]: entry[0] for entry in queue}
                head = 0
                while head < len(queue):
                    log_score_B, label_B, _, _, _ = queue[head]
                    head += 1
                    rhs_unary = (label_B,)
                    if rhs_unary in rhs_to_lhs:
                        for label_A in rhs_to_lhs[rhs_unary]:
                            bert_log_score_for_A = span_to_bert_log_score_map.get((i, j), {}).get(label_A, -float('inf'))
                            
                            # 複合ラベル (B-A) のスコアも取得
                            composite_label = f"{label_B}-{label_A}"
                            composite_log_score = span_to_bert_log_score_map.get((i, j), {}).get(composite_label, 0.0)

                            # 3つのスコアを合算
                            new_log_score = log_score_B + bert_log_score_for_A + composite_log_score
                            if new_log_score > best_scores.get(label_A, -float('inf')):
                                best_scores[label_A] = new_log_score
                                new_entry = (new_log_score, label_A, None, queue[head-1], None)
                                queue.append(new_entry)
                
                queue.sort(key=lambda x: x[0], reverse=True)
                chart[i][j] = queue[:top_k]
        return chart

    chart = cky_parse(tokens, model, id2label, tokenizer, rhs_to_lhs, device)

    # --- ▼▼▼ バグを修正した木構造構築ロジック ▼▼▼ ---

    def build_tree(chart, i, j, target_label):
        """
        チャート(i, j)の中から、target_labelを持つ最適なエントリを探して木を構築する。
        """
        best_entry_for_label = None
        for entry in chart[i][j]:
            if entry[1] == target_label:
                best_entry_for_label = entry
                break
        
        if best_entry_for_label is None:
            return None

        _score, label, split_k, left_desc, right_desc = best_entry_for_label

        if split_k is None:
            if isinstance(left_desc, tuple): # ユナリー規則 (A -> B) の場合
                child_entry = left_desc
                child_label = child_entry[1]
                child_tree = build_tree(chart, i, j, child_label)
                return Tree(label, [child_tree]) if child_tree else Tree(label, [])
            else: # 終端規則 (X -> word) の場合
                return Tree(label, [str(left_desc)])
        else: # 二項規則 (A -> B C) の場合
            left_i, left_j, left_label = left_desc
            right_i, right_j, right_label = right_desc
            
            left_subtree = build_tree(chart, left_i, left_j, left_label)
            right_subtree = build_tree(chart, right_i, right_j, right_label)
            
            children = []
            if left_subtree: children.append(left_subtree)
            if right_subtree: children.append(right_subtree)
                
            return Tree(label, children)

    # ---- 初期呼び出し ----
    if not chart or not chart[0][len(tokens)] or not chart[0][len(tokens)][0]:
        return Tree('TOP', [str(t) for t in tokens])

    root_label = chart[0][len(tokens)][0][1]
    tree = build_tree(chart, 0, len(tokens), root_label)

    if tree is None:
        return Tree('TOP', [str(t) for t in tokens])

    return tree