from transformers import BertTokenizerFast
import torch
from torch.optim import AdamW
import os
import wandb
from transformers import get_linear_schedule_with_warmup

from data import read_and_binarize, build_label2id, build_rhs_to_lhs, split_data
from train_evaluate import train_and_validate, evaluate_model, generate_detailed_report
from model import SpanScorer
from evalb import run_pyevalb_evaluation

def main():
    file_names = [
        '/home/higashi/workspace/BERT-CKY/multi-domain-parsing-analysis/data/MCTB_en/dialogue.cleaned.txt',
        '/home/higashi/workspace/BERT-CKY/multi-domain-parsing-analysis/data/MCTB_en/forum.cleaned.txt',
        '/home/higashi/workspace/BERT-CKY/multi-domain-parsing-analysis/data/MCTB_en/law.cleaned.txt',
        '/home/higashi/workspace/BERT-CKY/multi-domain-parsing-analysis/data/MCTB_en/literature.cleaned.txt',
        '/home/higashi/workspace/BERT-CKY/multi-domain-parsing-analysis/data/MCTB_en/review.cleaned.txt'
    ]

    config = {
        "epochs": 20,
        "learning_rate": 2e-5,
        "patience": 5,
        "model_name": "bert-base-uncased",
        "dataset": "MCTB",
        "optimizer": "AdamW",
        "weight_decay": 0.01,    #0.01がデフォルト値
        "learining_rate_scheduling": "linear_warmup"
    }

    wandb.init(
        project="BERT_CKY",  # プロジェクト名（自由に設定）
        config=config,
    )
    
    binarized_trees = read_and_binarize(file_names)

    train_trees, val_trees, test_trees = split_data(binarized_trees)
    label2id = build_label2id(train_trees)
    rhs_to_lhs = build_rhs_to_lhs(train_trees)
    id2label = {idx: label for label, idx in label2id.items()}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SpanScorer(num_labels=len(label2id)).to(device)
    tokenizer = BertTokenizerFast.from_pretrained(config['model_name'])
    wandb.watch(model, log_freq=100)
    optimizer = AdamW(model.parameters(), lr=wandb.config.learning_rate, weight_decay=wandb.config.weight_decay)
    epochs = wandb.config.epochs

    num_training_steps = len(train_trees) * epochs
    num_warmup_steps = int(0.1 * num_training_steps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )

    best_model_state_dict = train_and_validate(model, train_trees, val_trees, epochs, label2id, tokenizer, optimizer, scheduler, device, patience=wandb.config.patience)
    
    torch.save(best_model_state_dict, os.path.join("./", "model.pt"))
    model.load_state_dict(torch.load("./model.pt", map_location=device))

    evaluate_model(model, test_trees, label2id, tokenizer, device)
    wandb.finish()

    # 20, 161, 257
    a = []
    for i in range(0,500):
        if i in [20, 161, 257]:
            continue
        a.append(test_trees[i])
    evalb_result = run_pyevalb_evaluation(model, test_trees[9], id2label, tokenizer, rhs_to_lhs, device)

    print("\n--- Final Test Results (PARSEVAL with pyevalb) ---")
    print(f"  Precision: {evalb_result['precision']:.4f}")
    print(f"  Recall:    {evalb_result['recall']:.4f}")
    print(f"  F1-Score:  {evalb_result['f1']:.4f}")
    print(f"  Tagging Accuracy: {evalb_result['tag_accuracy']:.2%}")

if __name__ == "__main__":
    main()
