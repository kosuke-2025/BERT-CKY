from nltk import Tree
from tqdm import tqdm

# PYEVALBライブラリから、正しいクラスと関数をインポート
from PYEVALB.scorer import Scorer
from PYEVALB.parser import create_from_bracket_string
from PYEVALB.summary import summary as summarize_results # summaryという名前が競合しないように別名でインポート

# debinarizeやparsing関数も必要に応じてインポート
from parsing import parsing
from data import debinarize 

def run_pyevalb_evaluation(model, test_trees, id2label, tokenizer, rhs_to_lhs, device):
    """
    公式ソースコードのAPIに完全に準拠し、データセット全体のPARSEVALスコアを集計する。
    """

    # 1. Scorerのインスタンスを最初に一つだけ作成
    scorer = Scorer()
    
    # 2. 文ごとの評価結果(Resultオブジェクト)を格納するリスト
    results_list = []

    for gold_nltk_tree in tqdm(test_trees, desc="PYEVALB Evaluation"):
        gold_tokens = gold_nltk_tree.leaves()
        if not gold_tokens:
            continue

        predicted_nltk_tree = parsing(gold_tokens, model, id2label, tokenizer, rhs_to_lhs, device)
        predicted_tokens = predicted_nltk_tree.leaves()

        if gold_tokens != predicted_tokens:
            print("a")
            continue
        
        if predicted_nltk_tree:
            debinarized_gold_tree = debinarize(gold_nltk_tree)
            debinarized_predicted_tree = debinarize(predicted_nltk_tree)

            predicted_nltk_tree.pretty_print()   # 予測の2
            debinarized_predicted_tree.pretty_print()    # 予測のタ
            gold_nltk_tree.pretty_print()   # 正解の2
            debinarized_gold_tree.pretty_print()    # 正解のタ

            gold_str = str(debinarized_gold_tree)
            pred_str = str(debinarized_predicted_tree)
            
            # PYEVALBの内部形式に変換
            gold_tree_internal = create_from_bracket_string(gold_str)
            pred_tree_internal = create_from_bracket_string(pred_str)

            # 3. Scorerで一文を評価し、Resultオブジェクトを取得してリストに追加
            single_result = scorer.score_trees(pred_tree_internal, gold_tree_internal)
            results_list.append(single_result)

    # 4. 全ての文の評価が終わったら、Resultのリストをsummary関数に渡して集計
    if not results_list:
        return {"f1": 0, "precision": 0, "recall": 0, "tag_accuracy": 0}

    corpus_summary = summarize_results(results_list)

    # 5. 集計結果から最終的なスコアを取り出して返す
    #    (ソースコード内の属性名に合わせて取得し、0-1の範囲に変換)
    return {
        "f1": corpus_summary.bracker_fmeasure / 100.0,
        "precision": corpus_summary.bracket_prec / 100.0,
        "recall": corpus_summary.bracket_recall / 100.0,
        "tag_accuracy": corpus_summary.tagging_accuracy / 100.0
    }