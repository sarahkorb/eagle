from eagle import EagleRanker
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from tqdm.auto import tqdm

load_dotenv()

def train_eagle(df_train, df_val, models, N=20, K=32, P = 0.5):
    print("Training Eagle Global ELO")
    ranker = EagleRanker(models=models, P=P, N=N, K=K)
    combined = pd.concat([df_train, df_val], ignore_index=True)
    print("Embedding and populating prompts dict")
    ranker.populate_prompts(combined)
    train_ids = df_train["prompt_id"].tolist()
    print("Updating GLOBAL ELO on training set data ")
    for pid in tqdm(train_ids, desc="Global ELO updates", unit="prompt"):
        ranker.train_global_elo(pid)
    return ranker

def evaluate_eagle(ranker, df, models):
    correct = 0
    total = 0

    for _, row in df.iterrows():
        prompt_text = row["prompt"] + " " + row.get("question", "")
        pred = ranker.make_model_prediction(prompt_text)

        if isinstance(pred, tuple) and pred[0] == "tie":
            pred_model = "tie"
        else:
            pred_model = pred

        score_a = row[f"{models[0]}/score"]
        score_b = row[f"{models[1]}/score"]

        if score_a == score_b and pred_model == "tie":
            correct += 1   # count ties as correct
            total += 1
            continue

        if score_a > score_b:
            true = models[0]
        else:
            true = models[1]

        if pred_model == true:
            correct += 1

        total += 1

    return correct / total if total > 0 else 0

def run_full_train_test(df_train, df_val, df_test, models, P_values=None, N=20, K=32):
    if P_values is None:
        P_values = np.linspace(0.0, 1.0, 11)

    ranker = train_eagle(df_train, df_val, models, N=N, K=K)
    print("\nEvaluating test accuracy with paper default P=0.5...")
    ranker.P = 0.5
    test_acc_paper = evaluate_eagle(ranker, df_test, models)
    print(f"Test accuracy (P=0.5): {test_acc_paper:.4f}")

    print("\nGrid search to optimize P...")
    best_P = 0.5
    best_val_acc = -1

    for P in tqdm(P_values, desc="P sweep"):
        ranker.P = float(P)
        val_acc = evaluate_eagle(ranker, df_val, models)

        print(f"P={P:.2f}  Validation accuracy={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_P = P

    print(f"\nBest P found on validation: {best_P:.3f}")
    print(f"Best validation accuracy: {best_val_acc:.4f}")

    # Final test with optimized P
    print(f"\nEvaluating test accuracy with optimized P={best_P}...")
    ranker.P = best_P
    test_acc_opt = evaluate_eagle(ranker, df_test, models)
    print(f"Test accuracy (P={best_P}): {test_acc_opt:.4f}")

    return {
        "paper_test_accuracy": test_acc_paper,
        "best_P": best_P,
        "best_val_accuracy": best_val_acc,
        "optimized_test_accuracy": test_acc_opt,
        "global_elo": ranker.global_elo
    }

# Login using e.g. `huggingface-cli login` to access this dataset
splits = {
    'train': 'data/train-00000-of-00001.parquet',
    'test':  'data/test-00000-of-00001.parquet',
    'val':   'data/val-00000-of-00001.parquet'
}

df_train = pd.read_parquet("hf://datasets/notdiamond/repliqa_gpt4o_gpt4omini_evals/" + splits["train"]).head(1000)
df_val   = pd.read_parquet("hf://datasets/notdiamond/repliqa_gpt4o_gpt4omini_evals/" + splits["val"]).head(100)
df_test  = pd.read_parquet("hf://datasets/notdiamond/repliqa_gpt4o_gpt4omini_evals/" + splits["test"]).head(100)

models = ["gpt-4o-2024-08-06", "gpt-4o-mini-2024-07-18"]

results = run_full_train_test(
    df_train=df_train,
    df_val=df_val,
    df_test=df_test,
    models=models,
    P_values = None,
    N=20,   # per the default values from the paper
    K=32    # ELO update constant
)

print(results)