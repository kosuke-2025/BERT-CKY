import torch
from torch.nn.functional import cross_entropy
from tqdm import tqdm
from sklearn.metrics import f1_score
import wandb

from data import prepare_input

def train_val_test_per_epoch(model, train_trees, val_trees, test_trees, epochs, label2id, tokenizer, optimizer, device,
                             patience):
    """
    【変更】毎エポックごとに訓練・検証・テストを行う。
    モデルの訓練と検証を行い、早期終了を実装する。
    検証ロスが最も良かったモデルの状態(state_dict)を返す。
    注意：毎エポックでテストデータを使用するため、テストデータが学習プロセスに
    間接的に影響を与える可能性があり、厳密な評価方法ではありません。
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
            optimizer.zero_grad()
            train_total_loss += loss.item() * len(labels)
            train_num_spans += len(labels)
        avg_train_loss = train_total_loss / train_num_spans if train_num_spans > 0 else 0

        # --- 検証・テスト共通評価関数 ---
        def evaluate(data_trees, desc, device):
            model.eval()
            total_loss = 0.0
            num_spans = 0
            all_labels = []
            all_predictions = []
            with torch.no_grad():
                for tree in tqdm(data_trees, desc=desc):
                    encoding, spans, labels = prepare_input(tree, label2id, tokenizer)
                    if not spans: continue
                    spans = [spans]
                    input_ids, attention_mask = encoding["input_ids"].to(device), encoding["attention_mask"].to(device)
                    label_tensor = torch.tensor(labels, dtype=torch.long).to(device)
                    logits = model(input_ids=input_ids, attention_mask=attention_mask, spans=spans)
                    if logits.shape[0] == 0: continue
                    loss = cross_entropy(logits, label_tensor)
                    total_loss += loss.item() * len(labels)
                    num_spans += len(labels)
                    predictions = torch.argmax(logits, dim=1)
                    all_labels.extend(label_tensor.cpu().numpy())
                    all_predictions.extend(predictions.cpu().numpy())
            
            avg_loss = total_loss / num_spans if num_spans > 0 else 0
            f1 = f1_score(all_labels, all_predictions, average='weighted', zero_division=0) if all_labels else 0
            return avg_loss, f1

        # --- 検証フェーズ ---
        avg_val_loss, val_f1 = evaluate(val_trees, "Validation", device)

        # --- 【変更】テストフェーズ ---
        avg_test_loss, test_f1 = evaluate(test_trees, "Testing", device)


        # --- 結果の表示 ---
        print(f"  Epoch {epoch + 1} Summary:")
        print(f"    Train: Loss = {avg_train_loss:.4f}")
        print(f"    Val  : Loss = {avg_val_loss:.4f}, F1 = {val_f1:.4f}")
        print(f"    Test : Loss = {avg_test_loss:.4f}, F1 = {test_f1:.4f}")


        # --- wandbにログを記録 ---
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "val_f1-score": val_f1,
            "test_loss": avg_test_loss,   # 【変更】テストのロスを追加
            "test_f1-score": test_f1    # 【変更】テストのF1スコアを追加
        })

        # --- 早期終了のロジック (検証ロスに基づく) ---
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
    
    # 最も良かったモデルのstate_dictのみを返す
    return best_model_state_dict