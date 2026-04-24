import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

# === 固定定義：-1=異常，1=正常 ===
ANOMALY_LABEL = -1
NORMAL_LABEL = 1

TEST_SETS = [1, 2, 3, 4, 5]

# 這支腳本會載入已訓練的模型（由 Algorithms/OCNN/tlight_ocnn.py 存下來）
_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.join(_script_dir, "..")

# 自動找最新的 Model N 資料夾（編號最大）
_model_root = os.path.join(_project_root, "training_result", "Model storage space")
_existing = [
    d for d in os.listdir(_model_root)
    if os.path.isdir(os.path.join(_model_root, d)) and d.startswith("Model ")
]
if not _existing:
    raise FileNotFoundError(f"❌ 找不到任何模型資料夾，請先執行訓練：{_model_root}")
_latest_model = sorted(_existing, key=lambda x: int(x.split()[-1]))[-1]
MODEL_DIR  = os.path.join(_model_root, _latest_model)
print(f"📂 載入模型：{MODEL_DIR}")

MODEL_CKPT = os.path.join(MODEL_DIR, "ocnn.ckpt")
SCALER_PKL = os.path.join(MODEL_DIR, "scaler.pkl")
META_PKL   = os.path.join(MODEL_DIR, "meta.pkl")

# Dataset/test_data/test_data/
TEST_PATH = os.path.join(_project_root, "Dataset", "test_data", "test_data")

# TF2 相容 TF1 API
if hasattr(tf, "compat") and hasattr(tf.compat, "v1"):
    tf = tf.compat.v1
    tf.disable_v2_behavior()


def load_pipeline_and_meta():
    with open(SCALER_PKL, "rb") as f:
        pipeline = pickle.load(f)
    with open(META_PKL, "rb") as f:
        meta = pickle.load(f)
    return pipeline, meta


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


def build_graph(x_size: int, h_size: int):
    tf.reset_default_graph()
    X = tf.placeholder(tf.float32, shape=[None, x_size], name="X")

    w_1 = tf.Variable(tf.zeros([x_size, h_size], dtype=tf.float32), name="w_1")
    w_2 = tf.Variable(tf.zeros([h_size, 1], dtype=tf.float32), name="w_2")

    # 必須與 tlight_ocnn.py 的 nnScore 完全一致 (加上 ReLU)
    h = tf.nn.relu(tf.matmul(X, w_1))
    score_op = tf.matmul(h, w_2)
    saver = tf.train.Saver(var_list={"w_1": w_1, "w_2": w_2})
    return X, score_op, saver


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
            boxprops=dict(facecolor="#d08b5b", alpha=0.85),
            medianprops=dict(color="black"),
            whiskerprops=dict(color="#333"),
            capprops=dict(color="#333"),
            flierprops=dict(marker="o", markersize=4, markerfacecolor="black", alpha=0.7),
        )
        ax.set_title(title)
        ax.set_ylim(0.0, 1.0)
        ax.set_xticks([1])
        ax.set_xticklabels(["OCNN"])
        ax.grid(axis="y", linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main():
    pipeline, meta = load_pipeline_and_meta()
    x_size = int(meta["x_size"])
    h_size = int(meta["h_size"])
    
    # 判斷是新版中心距離法還是舊版 rstar
    if "score_center" in meta:
        score_center = float(meta["score_center"])
        dist_thresh = float(meta["distance_threshold"])
        old_mode = False
    else:
        rstar = float(meta["rstar"])
        old_mode = True

    X, score_op, saver = build_graph(x_size=x_size, h_size=h_size)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    saver.restore(sess, MODEL_CKPT)

    rows = []
    for i in TEST_SETS:
        df_test = load_test_df(i)
        y = df_test["class"].to_numpy()
        X_test = pipeline.transform(df_test.iloc[:, :-1]).astype(np.float32)

        scores = sess.run(score_op, feed_dict={X: X_test}).reshape(-1)
        
        if old_mode:
            decision = scores - rstar
            # ── 診斷：分數分布 vs rstar ──
            print(f"  [診斷] 舊版 rstar={rstar:.4f} | scores: min={scores.min():.4f}, "
                  f"mean={scores.mean():.4f}, max={scores.max():.4f} | "
                  f"低於rstar比例: {(scores < rstar).mean()*100:.1f}%")
            y_pred_bin = (decision < 0).astype(int)
        else:
            distances = np.abs(scores - score_center)
            decision = dist_thresh - distances
            print(f"  [新版距離診斷] 🎯 靶心={score_center:.4f}, 🛡️ 半徑={dist_thresh:.4f} | 實際距離: min={distances.min():.4f}, "
                  f"mean={distances.mean():.4f}, max={distances.max():.4f} | "
                  f"超出半徑比例 (異常率): {(distances > dist_thresh).mean()*100:.1f}%")
            # 超出容忍半徑 (decision < 0) 則視為異常 (歸類為 1)
            y_pred_bin = (decision < 0).astype(int)

        y_true_bin = (y == ANOMALY_LABEL).astype(int)
        acc, prec, rec, f1 = compute_metrics(y_true_bin, y_pred_bin)
        rows.append({"test_set": i, "accuracy": acc, "precision": prec, "recall": rec, "f1": f1})
        print(f"test_set_{i}: acc={acc:.4f}, prec={prec:.4f}, rec={rec:.4f}, f1={f1:.4f}")

    sess.close()

    dfm = pd.DataFrame(rows).sort_values("test_set")

    out_dir = os.path.join(_script_dir, "picture")
    os.makedirs(out_dir, exist_ok=True)

    # 圖 1：五個測試集的表格（圖片）
    table_path = os.path.join(out_dir, "ocnn_metrics_table.png")
    render_table_figure(dfm, table_path)
    print(f"✅ saved: {table_path}")

    # 圖 2：像作者那樣的箱型圖（四宮格）
    boxplot_path = os.path.join(out_dir, "ocnn_metrics_boxplot.png")
    render_boxplot_figure(dfm, boxplot_path)
    print(f"✅ saved: {boxplot_path}")


if __name__ == "__main__":
    main()