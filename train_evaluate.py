import torch
from torch.nn.functional import cross_entropy
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, classification_report, f1_score
import wandb

from data import prepare_input

def train_and_validate(model, train_trees, val_trees, epochs, label2id, tokenizer, optimizer, scheduler, device, patience):
    """
    モデルの訓練と検証を行い、早期終了を実装する。
    最も性能の良かったモデルの状態(state_dict)を返す。
    """
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state_dict = None

    for epoch in range(epochs):
        print(f"\n--- Epoch {epoch + 1}/{epochs} ---")

        # --- 訓練フェーズ ---
        model.train()
        train_total_loss = 0.0
        train_num_spans = 0
        for tree in tqdm(train_trees, desc="Training"):
            encoding, spans, labels = prepare_input(tree, label2id, tokenizer)
            if not spans: continue
            spans = [spans]
            input_ids, attention_mask = encoding["input_ids"].to(device), encoding["attention_mask"].to(device)
            label_tensor = torch.tensor(labels, dtype=torch.long).to(device)
            logits = model(input_ids=input_ids, attention_mask=attention_mask, spans=spans)
            if logits.shape[0] == 0: continue
            loss = cross_entropy(logits, label_tensor)
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            train_total_loss += loss.item() * len(labels)
            train_num_spans += len(labels)
        avg_train_loss = train_total_loss / train_num_spans if train_num_spans > 0 else 0

        # --- 検証フェーズ ---
        model.eval()
        val_total_loss = 0.0
        val_num_spans = 0
        all_val_labels = []
        all_val_predictions = []

        with torch.no_grad():
            for tree in tqdm(val_trees, desc="Validation"):
                encoding, spans, labels = prepare_input(tree, label2id, tokenizer)
                if not spans: continue
                spans = [spans]
                input_ids, attention_mask = encoding["input_ids"].to(device), encoding["attention_mask"].to(device)
                label_tensor = torch.tensor(labels, dtype=torch.long).to(device)
                logits = model(input_ids=input_ids, attention_mask=attention_mask, spans=spans)
                if logits.shape[0] == 0: continue
                loss = cross_entropy(logits, label_tensor)
                val_total_loss += loss.item() * len(labels)
                val_num_spans += len(labels)
                predictions = torch.argmax(logits, dim=1)
                all_val_labels.extend(label_tensor.cpu().numpy())
                all_val_predictions.extend(predictions.cpu().numpy())
        avg_val_loss = val_total_loss / val_num_spans if val_num_spans > 0 else 0
        val_f1 = f1_score(all_val_labels, all_val_predictions, average='weighted', zero_division=0) if all_val_labels else 0

        # --- 結果の表示 ---
        print(f"  Epoch {epoch + 1} Summary: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}")

        wandb.log({
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "val_f1-score": val_f1
        })

        # --- 早期終了のロジック ---
        if avg_val_loss < best_val_loss:
            print(f"    -> Validation loss improved from {best_val_loss:.4f} to {avg_val_loss:.4f}. Storing best model state...")
            best_val_loss = avg_val_loss
            best_model_state_dict = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"    -> Validation loss did not improve. Patience counter: {patience_counter}/{patience}")

        if patience_counter >= patience:
            print("Early stopping triggered. Stopping training.")
            break
    
    # 【変更】最も良かったモデルのstate_dictのみを返す
    return best_model_state_dict



def evaluate_model(model, test_trees, label2id, tokenizer, device):
    """
    指定されたデータセットでモデルを評価し、損失、正解率、適合率、再現率、F1スコアを返す。
    """
    
    model.eval()
    
    total_loss = 0.0
    
    # 【変更】全ての予測と正解ラベルを格納するリスト
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for tree in tqdm(test_trees, desc="Evaluating"):
            encoding, spans, labels = prepare_input(tree, label2id, tokenizer)

            if not spans:
                continue
            
            spans_batch = [spans]
            input_ids = encoding["input_ids"].to(device)
            attention_mask = encoding["attention_mask"].to(device)
            label_tensor = torch.tensor(labels, dtype=torch.long).to(device)

            logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                spans=spans_batch
            )

            if logits.shape[0] == 0:
                continue
            
            loss = cross_entropy(logits, label_tensor, reduction='sum')
            total_loss += loss.item()

            predictions = torch.argmax(logits, dim=1)
            
            # 【変更】ラベルと予測をリストに追加（scikit-learnで処理するためにCPUに移動し、numpy配列に変換）
            all_labels.extend(label_tensor.cpu().numpy())
            all_predictions.extend(predictions.cpu().numpy())
            
    # 【変更】scikit-learnを使って各指標を計算
    if not all_labels: # 1つも評価対象のスパンがなかった場合
        return print(f"Test Loss: {0}\nTest Accuarcy: {0}\nTest Precision: {0}\nTest Recall: {0}\nTest F1-score: {0}")

    # 全体の平均スコアを計算（重み付き平均）。ラベルの出現頻度の偏りを考慮する。
    precision, recall, f1_score, _ = precision_recall_fscore_support(
        all_labels, all_predictions, average='weighted', zero_division=0
    )
    
    # 正解率もscikit-learnで計算可能
    accuracy = accuracy_score(all_labels, all_predictions)
    
    # 平均損失
    avg_loss = total_loss / len(all_labels)

    wandb.summary["test_loss"] = avg_loss
    wandb.summary["test_accuracy"] = accuracy
    wandb.summary["test_precision"] = precision
    wandb.summary["test_recall"] = recall
    wandb.summary["test_F1-score"] = f1_score

    return print(f"Test Loss: {avg_loss:4f}\nTest Accuracy: {accuracy:4f}\nTest Precision: {precision:4f}\nTest Recall: {recall:4f}\nTest F1-score: {f1_score:4f}")



def generate_detailed_report(model, test_trees, label2id, id2label, tokenizer, device):
    """
    テストデータでモデルを評価し、各ラベルごとの詳細な性能レポートを表示する（簡易版）。
    """
    print("\n--- Detailed Classification Report ---")
    
    model.eval()
    all_labels = []
    all_predictions = []

    # 1. 予測と正解のラベルを収集する (この部分は変更なし)
    with torch.no_grad():
        for tree in tqdm(test_trees, desc="Generating Report"):
            encoding, spans, labels = prepare_input(tree, label2id, tokenizer)
            if not spans:
                continue
            
            spans_batch = [spans]
            input_ids = encoding["input_ids"].to(device)
            attention_mask = encoding["attention_mask"].to(device)
            
            logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                spans=spans_batch
            )
            if logits.shape[0] == 0:
                continue

            predictions = torch.argmax(logits, dim=1)
            all_labels.extend(labels)
            all_predictions.extend(predictions.cpu().numpy())
            
    if not all_labels:
        print("No data available to generate a report.")
        return

    # --- ▼▼▼【簡易版】ここからがレポート生成ロジック ▼▼▼ ---

    # 2. 実際にテストデータに出現したラベルIDのリストを作成する
    #    正解ラベルと予測ラベルの両方に含まれるIDを重複なく集め、ソートする
    present_label_ids = sorted(list(set(all_labels) | set(all_predictions)))

    # 3. 出現したラベルIDに対応する名前のリストを作成する
    present_target_names = [id2label[label_id] for label_id in present_label_ids]

    # 4. 絞り込んだリストを引数として渡す
    report = classification_report(
        y_true=all_labels, 
        y_pred=all_predictions, 
        labels=present_label_ids,           # レポート対象を、出現したラベルに限定
        target_names=present_target_names,
        digits=4, 
        zero_division=0
    )
    print(report)