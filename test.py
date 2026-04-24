import os
import numpy as np
import tensorflow.compat.v1 as tf
import warnings

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.disable_v2_behavior()

# ===== 1. 建立 1 萬筆 1~100 的隨機數字資料集 =====
dataset_file = "dummy_dataset.txt"
if not os.path.exists(dataset_file):
    print("📝 正在建立資料集 (1萬筆 1~100 的隨機整數)...")
    # 生成 1~100 的隨機數 (1萬筆)
    data = np.random.randint(1, 101, size=10000)
    np.savetxt(dataset_file, data, fmt="%d")

# ===== 2. 讀取並正規化資料 =====
print(f"\n📖 讀取資料集 {dataset_file}...")
raw_data = np.loadtxt(dataset_file).reshape(-1, 1).astype(np.float32)

# 神經網路對原始數值敏感，將 1~100 壓縮到 0.01 ~ 1.0 的範圍
train_X = raw_data / 100.0

# ===== 3. OCNN 神經網路訓練架構 =====
def train_simple_ocnn(X_data, nu=0.05, epochs=100, batch_size=256, learning_rate=0.01):
    tf.reset_default_graph()
    tf.set_random_seed(42)
    np.random.seed(42)
    
    x_size = 1  # 只有一個特徵 (數字)
    h_size = 8  # 小型的隱藏層
    
    X = tf.placeholder(tf.float32, shape=[None, x_size])
    
    # r (即為未來的 rstar)
    r = tf.get_variable("r", shape=(), trainable=False, initializer=tf.constant_initializer(0.1))
    
    w_1 = tf.Variable(tf.random_normal((x_size, h_size), stddev=0.1))
    b_1 = tf.Variable(tf.zeros([h_size]))
    w_2 = tf.Variable(tf.random_normal((h_size, 1), stddev=0.1))
    b_2 = tf.Variable(tf.zeros([1]))
    
    # 還原為純線性網路 (沒有激勵函數，完全模擬最原始的程式碼狀態)
    h = tf.matmul(X, w_1) + b_1
    scores = tf.matmul(h, w_2) + b_2
    
    # OCNN 損失函數
    reg = 0.5 * tf.reduce_sum(w_1 ** 2) + 0.5 * tf.reduce_sum(w_2 ** 2)
    hinge_loss = (1.0 / nu) * tf.reduce_mean(tf.nn.relu(r - scores))
    cost = reg + hinge_loss - r
    
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)
    
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    
    rvalue = 0.1
    print("\n🚀 開始訓練 OCNN 模型...")
    for epoch in range(epochs):
        indices = np.random.permutation(len(X_data))
        for start_idx in range(0, len(X_data)-batch_size+1, batch_size):
            batch_X = X_data[indices[start_idx:start_idx+batch_size]]
            sess.run(optimizer, feed_dict={X: batch_X, r: rvalue})
            
        # 每個 Epoch 結束後更新門檻
        current_scores = sess.run(scores, feed_dict={X: X_data})
        rvalue = np.percentile(current_scores, q=100 * nu)
        
        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1:3d} | 目前模型及格線 rstar = {rvalue:.6f}")
            
    print("✅ 訓練完成！\n")
    return sess, X, scores, rvalue

# 設定 nu=0.05 也就是允許 5% 的訓練資料被當成自然雜訊
sess, X_tensor, score_tensor, final_rstar = train_simple_ocnn(train_X, nu=0.05, epochs=100)

# ===== 4. 準備測試資料 =====
print("=" * 60)
print("📊 模型門檻 (rstar):", final_rstar)
print("=" * 60)

# 第一組測試：正常範圍內的 5 個數字 (介於 1 ~ 100)
normal_test_raw = np.random.randint(1, 101, size=(5, 1)).astype(np.float32)

# 第二組測試：異常極端數字 5 個 (超大正數 或 負數)
anomaly_test_raw = np.array([[-50], [-10], [150], [200], [500]], dtype=np.float32)

# ====== 5. 進行測試 ======
print("\n🎯 【開始測試：正常的隨機數字 (落在 1~100 之間)】")
normal_test_X = normal_test_raw / 100.0  # 跟訓練期一樣要除以 100 轉換
normal_scores = sess.run(score_tensor, feed_dict={X_tensor: normal_test_X}).flatten()

for i, n in enumerate(normal_test_raw.flatten()):
    s = normal_scores[i]
    pred = "🟢 正常 (Normal)" if s >= final_rstar else "🔴 異常 (Anomaly)"
    print(f"輸入數字: {n:5.0f} | 神經網路分數: {s:7.4f} | 判斷結果: {pred}")

print("\n\n🎯 【開始測試：邊界外的極端數字 (負數 或 大於100的正數)】")
anomaly_test_X = anomaly_test_raw / 100.0
anomaly_scores = sess.run(score_tensor, feed_dict={X_tensor: anomaly_test_X}).flatten()

for i, n in enumerate(anomaly_test_raw.flatten()):
    s = anomaly_scores[i]
    pred = "🟢 超級正常!! (偽陽性)" if s >= final_rstar else "🔴 異常 (Anomaly)"
    print(f"輸入數字: {n:5.0f} | 神經網路分數: {s:7.4f} | 判斷結果: {pred}")
    
print("\n💡 你可以在這個極端數字測試中，親眼看見 OCNN 面對大於 100 分的「超量衝擊」時發生的致命數學盲點！")

sess.close()