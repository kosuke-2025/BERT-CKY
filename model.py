import torch
import torch.nn as nn
from transformers import BertModel

class SpanScorer(nn.Module):
    def __init__(self, num_labels):
        super().__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        hidden_size = self.bert.config.hidden_size
        self.classifier = nn.Linear(hidden_size * 2, num_labels)

    def forward(self, input_ids, attention_mask, spans):
        # input_ids: [batch_size, seq_len] (CKYからは batch_size=1 で渡される)
        # attention_mask: [batch_size, seq_len] (CKYからは batch_size=1)
        # spans: List[List[Tuple[int, int]]]
        #        CKYからは [[(s1,e1), (s2,e2), ...]] の形で渡される

        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        h = outputs.last_hidden_state  # [batch_size, seq_len, hidden_size]

        # CKYからは常にバッチサイズ1で、spans = [ list_of_actual_span_tuples ] の形で来る想定
        if not spans or not spans[0]: # スパンリストが空、または最初の要素(リスト)が空の場合
            # num_labels は self.classifier.out_features で取得可能
            return torch.empty(0, self.classifier.out_features, device=h.device)

        # 実際に処理するスパンタプルのリストを取得
        actual_span_tuples_for_this_sentence = spans[0]

        # バッチの最初の（そして唯一の）要素の隠れ状態を取得
        h_for_this_sentence = h[0]  # [seq_len, hidden_size]

        span_reps_for_this_sentence = []
        
        for start, end in actual_span_tuples_for_this_sentence: # ここで (start, end) タプルを正しく反復処理
            # スパンの範囲チェック (サブワードインデックスが隠れ状態の長さを超えないように)
            # end は exclusive (含まない) と想定
            if start < end and start < h_for_this_sentence.shape[0] and end <= h_for_this_sentence.shape[0]:
                span_start_rep = h_for_this_sentence[start]
                # end-1 で inclusive な終了インデックスにアクセス
                span_end_rep = h_for_this_sentence[end - 1]
                span_reps_for_this_sentence.append(torch.cat([span_start_rep, span_end_rep]))
            else:
                print(f"Warning: Span ({start},{end}) for sentence (seq_len {h_for_this_sentence.shape[0]}) is out of bounds or invalid.")


        if not span_reps_for_this_sentence: # 有効なスパン表現が一つも生成されなかった場合
            return torch.empty(0, self.classifier.out_features, device=h.device)
            
        # スパン表現をスタックして分類器に入力
        concatenated_span_reps = torch.stack(span_reps_for_this_sentence)
        logits = self.classifier(concatenated_span_reps)
        
        return logits