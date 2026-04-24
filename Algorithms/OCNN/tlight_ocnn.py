import os
import csv
import pickle
import numpy as np
import pandas as pd
import tensorflow.compat.v1 as tf
from itertools import zip_longest
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline

# 關閉 TF2 行為以相容舊版代碼
tf.disable_v2_behavior()

def write_decisionScores2Csv(path, filename, positiveScores, negativeScores):
    os.makedirs(path, exist_ok=True)
    newfilePath = os.path.join(path, filename)
    print(f"📁 儲存決策分數至: {newfilePath}")
    
    export_data = zip_longest(positiveScores.flatten(), negativeScores.flatten(), fillvalue='')
    with open(newfilePath, 'w', newline='') as myfile:
        wr = csv.writer(myfile)
        # Positive 代表訓練集(正常)，Negative 代表測試集(可能包含異常)
        wr.writerow(("Training_Score(>0 is Normal)", "Testing_Score(>0 is Normal)"))
        wr.writerows(export_data)

def get_mini_batches(X, batch_size):
    """將資料切分為多個 Mini-batch，提升訓練穩定度"""
    indices = np.random.permutation(len(X))
    for start_idx in range(0, len(X) - batch_size + 1, batch_size):
        excerpt = indices[start_idx:start_idx + batch_size]
        yield X[excerpt]

def tf_OneClass_NN_Relu(train_X, test_X, nu=0.04, epochs=300, batch_size=256, learning_rate=0.001, model_save_dir=None):
    tf.reset_default_graph()
    tf.set_random_seed(42)
    np.random.seed(42)

    x_size = train_X.shape[1]
    h_size = 32
    y_size = 1

    def relu(x):
        return tf.nn.relu(x)

    def init_weights(shape, name):
        weights = tf.random_normal(shape, mean=0.0, stddev=0.1)
        return tf.Variable(weights, name=name)

    X = tf.placeholder(tf.float32, shape=[None, x_size], name="X")
    r = tf.get_variable("r", shape=(), trainable=False, initializer=tf.constant_initializer(0.1))

    w_1 = init_weights((x_size, h_size), "w_1")
    w_2 = init_weights((h_size, y_size), "w_2")

    # 網路前向傳播
    def nnScore(X_input):
        h = relu(tf.matmul(X_input, w_1)) 
        return tf.matmul(h, w_2)

    scores = nnScore(X)

    # OCNN Loss 函數
    term1 = 0.5 * tf.reduce_sum(w_1 ** 2)
    term2 = 0.5 * tf.reduce_sum(w_2 ** 2)
    term3 = (1.0 / nu) * tf.reduce_mean(tf.nn.relu(r - scores))
    cost = term1 + term2 + term3 - r

    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    
    rvalue = 0.1

    print("🚀 開始訓練 OCNN 模型...")
    for epoch in range(epochs):
        epoch_loss = 0
        batches = 0
        
        for batch_X in get_mini_batches(train_X, batch_size):
            _, batch_cost = sess.run([optimizer, cost], feed_dict={X: batch_X, r: rvalue})
            epoch_loss += batch_cost
            batches += 1
            
        current_scores = sess.run(scores, feed_dict={X: train_X})
        # 更新 r 值: 確保大多數正常樣本的分數都在 r 之上
        rvalue = np.percentile(current_scores, q=100 * nu)
        
        if (epoch + 1) % 50 == 0 or epoch == 0:
            avg_loss = epoch_loss / max(batches, 1)
            print(f"Epoch = {epoch + 1:3d} | Avg Loss = {avg_loss:.4f} | r = {rvalue:.6f}")

    print("✅ 訓練完成！開始進行預測計算...")
    
    train_scores = sess.run(scores, feed_dict={X: train_X})
    test_scores = sess.run(scores, feed_dict={X: test_X})

    # ──────────────────────────────────────────
    # 【核心修復】中心距離打分法 (解決分數爆表誤判的問題)
    # ──────────────────────────────────────────
    # 1. 找出正常樣本輸出的中心點 (中位數較不受極端值影響)
    score_center = np.median(train_scores)
    
    # 2. 計算每個樣本偏離中心點的「絕對距離」 (距離越大，異常程度越高)
    train_distances = np.abs(train_scores - score_center)
    test_distances = np.abs(test_scores - score_center)

    # 3. 根據 nu 值決定「容忍的最大安全半徑 (閾值)」
    # nu=0.04 代表我們預期訓練集有 4% 可能是雜訊，所以取 96 百分位數作為邊界
    distance_threshold = np.percentile(train_distances, 100 * (1 - nu))
    
    print(f"\n📊 [統計資訊] 正常群體中心點: {score_center:.4f}")
    print(f"📏 [統計資訊] 最大容忍半徑: {distance_threshold:.4f}")

    # 4. 轉換為最終的決策分數 (大於 0 為正常，小於 0 為異常)
    # 邏輯：半徑(及格線) - 實際距離。若距離過大，相減就會是負數(異常)！
    pos_decisionScore = distance_threshold - train_distances
    neg_decisionScore = distance_threshold - test_distances

    # ──────────────────────────────────────────
    # 儲存模型與變數
    # ──────────────────────────────────────────
    if model_save_dir is not None:
        os.makedirs(model_save_dir, exist_ok=True)
        # 使用 list 存儲權重變數更穩定
        saver = tf.train.Saver([w_1, w_2])
        ckpt_path = os.path.join(model_save_dir, "ocnn.ckpt")
        saver.save(sess, ckpt_path)
        print(f"\n💾 模型權重已儲存至: {ckpt_path}")

        # 儲存預測時需要的 Meta 資訊 (將中心點與半徑存起來)
        meta = {
            "score_center": float(score_center),
            "distance_threshold": float(distance_threshold), 
            "x_size": int(x_size), 
            "h_size": int(h_size)
        }
        meta_path = os.path.join(model_save_dir, "meta.pkl")
        with open(meta_path, "wb") as f:
            pickle.dump(meta, f)
        print(f"💾 模型 meta 已儲存至: {meta_path}")

    sess.close()

    # 儲存決策分數 CSV
    _script_dir = os.path.dirname(os.path.abspath(__file__))
    decision_scorePath = os.path.join(_script_dir, "..", "..", "Dataset")
    write_decisionScores2Csv(decision_scorePath, "OC-NN_Relu.csv", pos_decisionScore, neg_decisionScore)

    return pos_decisionScore, neg_decisionScore

# =========== Main 執行區段 ==================
if __name__ == "__main__":
    _script_dir = os.path.dirname(os.path.abspath(__file__))
    _dataset_dir = os.path.join(_script_dir, "..", "..", "Dataset")
    
    # 【修復】更安全的路徑拼接方式
    train_csv_path = os.path.join(_dataset_dir, "train_data", "train_data", "train_dataset.csv")
    test_csv_path = os.path.join(_dataset_dir, "test_data", "test_data", "test_set_2.csv")
    
    try:
        train_df = pd.read_csv(train_csv_path, index_col=0)
        test_df = pd.read_csv(test_csv_path)
        
        # 排除可能存在的標籤欄位
        for col in ["class", "Unnamed: 0"]:
            if col in test_df.columns:
                test_df = test_df.drop(columns=[col])
                
    except Exception as e:
        print(f"❌ 讀取檔案失敗，請確認路徑: {e}")
        exit()

    # 資料正規化
    trans_pipeline = Pipeline([("scaler", MinMaxScaler())])
    train_data = trans_pipeline.fit_transform(train_df).astype(np.float32)
    test_data = trans_pipeline.transform(test_df).astype(np.float32)

    # 模型儲存根目錄
    _model_root = os.path.join(_script_dir, "..", "..", "training_result", "Model storage space")
    os.makedirs(_model_root, exist_ok=True)

    # 自動找下一個可用的編號資料夾
    _model_idx = 1
    while os.path.exists(os.path.join(_model_root, f"Model {_model_idx}")):
        _model_idx += 1
    _model_save_dir = os.path.join(_model_root, f"Model {_model_idx}")
    os.makedirs(_model_save_dir, exist_ok=True)
    print(f"📂 本次模型將儲存至: {_model_save_dir}")

    # 儲存 scaler pipeline
    scaler_path = os.path.join(_model_save_dir, "scaler.pkl")
    with open(scaler_path, "wb") as f:
        pickle.dump(trans_pipeline, f)

    # 執行模型
    ocnn_anomaly_scores = tf_OneClass_NN_Relu(
        train_data, test_data,
        nu=0.04,        
        epochs=300,     
        model_save_dir=_model_save_dir,
    )
    print("🎉 程式執行完畢！請查看生成的 CSV 檔案。")