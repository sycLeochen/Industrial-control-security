import os
import pickle
import numpy as np
import pandas as pd
import tensorflow.compat.v1 as tf
import warnings

# 忽略警告 (方便觀察結果)
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

tf.disable_v2_behavior()

_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.join(_script_dir, "..", "..")
_model_root = os.path.join(_project_root, "training_result", "Model storage space")

def get_latest_model_dir():
    existing = []
    if os.path.exists(_model_root):
        for d in os.listdir(_model_root):
            if os.path.isdir(os.path.join(_model_root, d)) and d.startswith('Model '):
                existing.append(d)
    if not existing:
        return None
    latest = max(existing, key=lambda x: int(x.split(' ')[1]))
    return os.path.join(_model_root, latest)

def run_custom_test():
    model_dir = get_latest_model_dir()
    if not model_dir:
        print("\n❌ 找不到訓練好的模型，請確認是否已經將模型儲存至 Model storage space/ 中。")
        return
    print(f"\n📂 成功載入最新模型: {model_dir}")

    # ===== 1. 載入訓練好的模型參數 =====
    meta_path = os.path.join(model_dir, "meta.pkl")
    with open(meta_path, "rb") as f:
        meta = pickle.load(f)
        
    x_size = meta["x_size"]
    h_size = meta["h_size"]
    
    if "score_center" in meta:
        score_center = float(meta["score_center"])
        dist_thresh = float(meta["distance_threshold"])
        old_mode = False
        print(f"📊 該模型的安全靶心 (Score Center) = {score_center:.4f}")
        print(f"🛡️ 該模型的安全容忍半徑 (Distance Threshold) = {dist_thresh:.4f}")
    else:
        rstar = float(meta["rstar"])
        old_mode = True
        print(f"📊 該模型的安全及格線 (rstar) = {rstar:.4f}")

    print("-" * 55)

    # ===== 2. 重建神經網路架構 (含我們加上的 ReLU) =====
    tf.reset_default_graph()
    tf.set_random_seed(42)
    np.random.seed(42)

    X = tf.placeholder(tf.float32, shape=[None, x_size], name="X")
    w_1 = tf.Variable(tf.zeros([x_size, h_size], dtype=tf.float32), name="w_1")
    w_2 = tf.Variable(tf.zeros([h_size, 1], dtype=tf.float32), name="w_2")

    # 對應 tlight_ocnn.py 中的隱藏層與激勵函數
    h = tf.nn.relu(tf.matmul(X, w_1))
    score_op = tf.matmul(h, w_2)
    saver = tf.train.Saver(var_list={"w_1": w_1, "w_2": w_2})

    # ===== 3. 自定義極端情境與 10 組真實正常狀況 =====
    scenarios = [
        "極端攻擊 1：燈號全亮 (36顆燈全為1.0，嚴重違規)", 
        "極端攻擊 2：只亮一半 (前18顆全為1.0，不合常理)"
    ]

    # 情境 1: 燈號全亮
    data_all_ones = np.ones((1, x_size), dtype=np.float32)

    # 情境 2: 只亮一半
    data_half_ones = np.zeros((1, x_size), dtype=np.float32)
    data_half_ones[0, :x_size//2] = 1.0

    raw_test_data = [data_all_ones, data_half_ones]

    # 情境 3 ~ 12: 直接從訓練集 (純淨正常資料) 裡面抽出 10 種真實正常的交通號誌狀態
    train_path = os.path.join(_project_root, "Dataset", "train_data", "train_data", "train_dataset.csv")
    try:
        # 只讀前 10 行真實的機台數據
        real_normal_df = pd.read_csv(train_path, nrows=10)
        
        # 移除不需要的欄位以對齊特徵數量
        if "Unnamed: 0" in real_normal_df.columns:
            real_normal_df = real_normal_df.drop(columns=["Unnamed: 0"])
        if "class" in real_normal_df.columns:
            real_normal_df = real_normal_df.drop(columns=["class"])
            
        real_normal_data = real_normal_df.to_numpy(dtype=np.float32)
        
        for i in range(10):
            scenarios.append(f"✅ 真實機台正常運作 {i+1} (來自訓練集)")
            raw_test_data.append(real_normal_data[i:i+1])
            
    except Exception as e:
        print(f"⚠️ 無法讀取真實訓練集: {e}")

    # 將所有測試資料打包成矩陣
    raw_test_data = np.vstack(raw_test_data)

    # ===== 4. 取出訓練用的 Scaler 進行數值還原與縮放 =====
    scaler_path = os.path.join(model_dir, "scaler.pkl")
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)

    # 嘗試抓取標準欄位名稱
    try:
        sample_csv = pd.read_csv(os.path.join(_project_root, "Dataset", "test_data", "test_data", "test_set_2.csv"), nrows=1)
        if "Unnamed: 0" in sample_csv.columns:
            sample_csv = sample_csv.drop(columns=["Unnamed: 0"])
        cols = sample_csv.columns[:-1] # 去掉 class 欄位
        df_custom = pd.DataFrame(raw_test_data, columns=cols)
        X_test = scaler.transform(df_custom).astype(np.float32)
    except Exception:
        X_test = scaler.transform(raw_test_data).astype(np.float32)

    # ===== 5. 使用神經網路進行打分與審判 =====
    with tf.Session() as sess:
        ckpt_path = os.path.join(model_dir, "ocnn.ckpt")
        saver.restore(sess, ckpt_path)

        # 把測試資料餵給網路，取得滿分
        scores = sess.run(score_op, feed_dict={X: X_test}).reshape(-1)

    print("\n🚀 【極端情境與真實情境：模型審判結果】\n")
    for i, scen in enumerate(scenarios):
        score = scores[i]
        
        if old_mode:
            is_normal = score >= rstar
            pred_text = "🟢 正常 (Normal)" if is_normal else "🔴 異常 (Anomaly)"
            
            print(f"情境 {i+1}: {scen}")
            print(f"   ► 模型輸出分數 (Score) ：{score:.4f}")
            print(f"   ► 突破了及格線嗎？ ({score:.4f} >= {rstar:.4f})  -> {'✔ 是' if is_normal else '✖ 否'}")
            print(f"   ► 最終判斷結果 ：【 {pred_text} 】\n")
        else:
            dist = abs(score - score_center)
            is_normal = dist <= dist_thresh
            pred_text = "🟢 正常 (Normal)" if is_normal else "🔴 異常 (Anomaly)"
            
            print(f"情境 {i+1}: {scen}")
            print(f"   ► 模型輸出原始分數 (Score)：{score:.4f}")
            print(f"   ► 計算偏離靶心距離       ：|{score:.4f} - {score_center:.4f}| = {dist:.4f}")
            print(f"   ► 距離是否在安全防護罩內？({dist:.4f} <= {dist_thresh:.4f}) -> {'✔ 是' if is_normal else '✖ 否'}")
            print(f"   ► 最終判斷結果         ：【 {pred_text} 】\n")

if __name__ == "__main__":
    run_custom_test()
