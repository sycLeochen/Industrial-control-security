import os
import numpy as np
import pandas as pd
import pickle
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import MinMaxScaler

# === 請填入路徑 ===
_script_dir = os.path.dirname(os.path.abspath(__file__))
TRAIN_DATA_PATH = os.path.join(_script_dir, "..", "..", "Dataset", "train_data", "train_data", "train_dataset.csv")
TEST_DIR_PATH = os.path.join(_script_dir, "..", "..", "Dataset", "test_data", "test_data")  # 測試集所在的資料夾路徑

ANOMALY_LABEL = -1
NORMAL_LABEL = 1
TEST_SETS = [1, 2, 3, 4, 5]

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

def main():
    print("🚀 載入訓練資料並進行正規化...")
    train_df = pd.read_csv(TRAIN_DATA_PATH, index_col=0)
    
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(train_df)

    print("🧠 開始訓練 OCSVM 模型...")
    # nu=0.04 與你 OCNN 的設定保持一致，確保對比基準公平
    ocsvm_model = OneClassSVM(kernel='rbf', gamma='scale', nu=0.04)
    ocsvm_model.fit(X_train)
    print("✅ 訓練完成！開始評估測試集...\n")

    # === 模型與 Scaler 儲存邏輯 ===
    model_root = os.path.join(_script_dir, "..", "..", "training_result", "Model storage space", "OCSVM")
    os.makedirs(model_root, exist_ok=True)
    model_idx = 1
    while os.path.exists(os.path.join(model_root, f"Model {model_idx}")):
        model_idx += 1
    model_save_dir = os.path.join(model_root, f"Model {model_idx}")
    os.makedirs(model_save_dir, exist_ok=True)
    print(f"📂 本次 OCSVM 模型將儲存至: {model_save_dir}")

    with open(os.path.join(model_save_dir, "ocsvm_model.pkl"), "wb") as f:
        pickle.dump(ocsvm_model, f)
    with open(os.path.join(model_save_dir, "scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)
    print("💾 儲存完成！\n")

    for i in TEST_SETS:
        test_path = os.path.join(TEST_DIR_PATH, f"test_set_{i}.csv")
        df_test = pd.read_csv(test_path)
        
        if "Unnamed: 0" in df_test.columns:
            df_test = df_test.drop(columns=["Unnamed: 0"])
            
        # 分離特徵與標籤
        y_true = df_test["class"].to_numpy()
        X_test = scaler.transform(df_test.drop(columns=["class"]))

        # sklearn 的異常偵測模型預測: 1 為正常, -1 為異常
        y_pred = ocsvm_model.predict(X_test)

        # 轉換為 0(正常) 與 1(異常) 來計算指標
        y_true_bin = (y_true == ANOMALY_LABEL).astype(int)
        y_pred_bin = (y_pred == ANOMALY_LABEL).astype(int)

        acc, prec, rec, f1 = compute_metrics(y_true_bin, y_pred_bin)
        print(f"Test_Set_{i} (OCSVM) -> acc: {acc:.4f}, prec: {prec:.4f}, rec: {rec:.4f}, f1: {f1:.4f}")

if __name__ == "__main__":
    main()