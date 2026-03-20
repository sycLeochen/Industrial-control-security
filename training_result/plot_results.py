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
MODEL_DIR = "./models/ocnn"
MODEL_CKPT = os.path.join(MODEL_DIR, "ocnn.ckpt")
SCALER_PKL = os.path.join(MODEL_DIR, "scaler.pkl")
META_PKL = os.path.join(MODEL_DIR, "meta.pkl")

TEST_PATH = "./test_data/test_data"

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

    def relu(x):
        return x

    score_op = tf.matmul(relu(tf.matmul(X, w_1)), w_2)
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
    rstar = float(meta["rstar"])

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
        decision = scores - rstar

        # 規則：decision < 0 判異常（1），>=0 判正常（0）
        y_true_bin = (y == ANOMALY_LABEL).astype(int)
        y_pred_bin = (decision < 0).astype(int)

        acc, prec, rec, f1 = compute_metrics(y_true_bin, y_pred_bin)
        rows.append({"test_set": i, "accuracy": acc, "precision": prec, "recall": rec, "f1": f1})
        print(f"test_set_{i}: acc={acc:.4f}, prec={prec:.4f}, rec={rec:.4f}, f1={f1:.4f}")

    sess.close()

    dfm = pd.DataFrame(rows).sort_values("test_set")

    out_dir = "training_result/picture"
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