import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler

# === 固定定義：-1=異常，1=正常 ===
ANOMALY_LABEL = -1
NORMAL_LABEL = 1
TEST_SETS = [1, 2, 3, 4, 5]

_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.join(_script_dir, "..")

# 資料集路徑
TRAIN_PATH = os.path.join(_project_root, "Dataset", "train_data", "train_data", "train_dataset.csv")
TEST_PATH = os.path.join(_project_root, "Dataset", "test_data", "test_data")

def load_test_df(i: int) -> pd.DataFrame:
    path = os.path.join(TEST_PATH, f"test_set_{i}.csv")
    df = pd.read_csv(path)
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])
    return df

def compute_metrics(y_true_bin: np.ndarray, y_pred_bin: np.ndarray):
    tp = int(np.sum((y_true_bin == 1) & (y_pred_bin == 1)))
    fp = int(np.sum((y_true_bin == 0) & (y_pred_bin == 1)))
    tn = int(np.sum((y_true_bin == 0) & (y_pred_bin == 0)))
    fn = int(np.sum((y_true_bin == 1) & (y_pred_bin == 0)))

    acc = (tp + tn) / max(tp + tn + fp + fn, 1)
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-12)
    return acc, prec, rec, f1

def render_table_figure(df: pd.DataFrame, out_path: str):
    fig, ax = plt.subplots(figsize=(10, 2.2))
    ax.axis("off")
    table = ax.table(
        cellText=np.round(df[["accuracy", "precision", "recall", "f1"]].values, 4),
        rowLabels=[f"test_set_{i}" for i in df["test_set"].tolist()],
        colLabels=["Accuracy", "Precision", "Recall", "F1-Score"],
        cellLoc="center",
        rowLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.4)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

def render_boxplot_figure(df: pd.DataFrame, out_path: str):
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    axes = axes.ravel()
    metric_specs = [
        ("accuracy", "Accuracy"),
        ("precision", "Precision"),
        ("recall", "Recall"),
        ("f1", "F1-Score"),
    ]
    for ax, (k, title) in zip(axes, metric_specs):
        vals = df[k].to_numpy()
        ax.boxplot(
            vals,
            widths=0.35,
            patch_artist=True,
            boxprops=dict(facecolor="#5ebd70", alpha=0.85), # 改成綠色區隔
            medianprops=dict(color="black"),
            whiskerprops=dict(color="#333"),
            capprops=dict(color="#333"),
            flierprops=dict(marker="o", markersize=4, markerfacecolor="black", alpha=0.7),
        )
        ax.set_title(title)
        ax.set_ylim(0.0, 1.0)
        ax.set_xticks([1])
        ax.set_xticklabels(["Isolation Forest"])
        ax.grid(axis="y", linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

def main():
    print("🔍 尋找最新的模型與 Scaler...")
    model_root = os.path.join(_script_dir, "Model storage space", "IF_Model")
    if not os.path.exists(model_root):
        print("❌ 找不到模型目錄！")
        return
    model_dirs = [d for d in os.listdir(model_root) if d.startswith("Model ")]
    if not model_dirs:
        print("❌ 尚無任何已儲存的 IF 模型！")
        return
    latest_model_dir = sorted(model_dirs, key=lambda x: int(x.split(" ")[1]))[-1]
    model_path = os.path.join(model_root, latest_model_dir)
    print(f"📂 載入模型來源: {model_path}")

    import pickle
    with open(os.path.join(model_path, "scaler.pkl"), "rb") as f:
        scaler = pickle.load(f)
    with open(os.path.join(model_path, "if_model.pkl"), "rb") as f:
        model = pickle.load(f)
    print("✅ 載入成功！開始評估測試集...\n")

    rows = []
    for i in TEST_SETS:
        df_test = load_test_df(i)
        y = df_test["class"].to_numpy()
        X_test = scaler.transform(df_test.iloc[:, :-1]).astype(np.float32)

        # sklearn 的預測: 1(正常), -1(異常)
        y_pred = model.predict(X_test)

        # 轉換為 0(正常) 與 1(異常) 來計算指標
        y_true_bin = (y == ANOMALY_LABEL).astype(int)
        y_pred_bin = (y_pred == ANOMALY_LABEL).astype(int)

        acc, prec, rec, f1 = compute_metrics(y_true_bin, y_pred_bin)
        rows.append({"test_set": i, "accuracy": acc, "precision": prec, "recall": rec, "f1": f1})
        print(f"test_set_{i}: acc={acc:.4f}, prec={prec:.4f}, rec={rec:.4f}, f1={f1:.4f}")

    dfm = pd.DataFrame(rows).sort_values("test_set")

    out_dir = os.path.join(_script_dir, "picture")
    os.makedirs(out_dir, exist_ok=True)

    table_path = os.path.join(out_dir, "if_metrics_table.png")
    render_table_figure(dfm, table_path)
    print(f"✅ saved: {table_path}")

    boxplot_path = os.path.join(out_dir, "if_metrics_boxplot.png")
    render_boxplot_figure(dfm, boxplot_path)
    print(f"✅ saved: {boxplot_path}")

if __name__ == "__main__":
    main()