# ONE-CLASS NEURAL NETWORK

import os
import time
import random
import csv

import numpy as np
import pandas as pd
from sklearn import utils
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import tensorflow as tf
# 讓 TensorFlow 2.x 相容舊的 TF1 程式碼（Session / placeholder / reset_default_graph 等）
if hasattr(tf, "compat") and hasattr(tf.compat, "v1"):
    tf = tf.compat.v1
    tf.disable_v2_behavior()
from itertools import zip_longest
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
import pickle


df_ocnn_scores = {}
decision_scorePath = "../../Dataset/"

def write_decisionScores2Csv(path, filename, positiveScores, negativeScores):
    newfilePath = path + filename
    print("Writing file to ", path + filename)
    poslist = positiveScores.tolist()
    neglist = negativeScores.tolist()

    d = [poslist, neglist]
    export_data = zip_longest(*d, fillvalue='')
    with open(newfilePath, 'w') as myfile:
        wr = csv.writer(myfile)
        wr.writerow(("Training", "Testing"))
        wr.writerows(export_data)
    myfile.close()
    
    return

def tf_OneClass_NN_Relu(data_train, data_test):
    tf.reset_default_graph()

    train_X = data_train

    RANDOM_SEED = 42
    tf.set_random_seed(RANDOM_SEED)

    # Layer's sizes
    x_size = train_X.shape[1]  # Number of input nodes
    h_size = 32  # Number of hidden nodes
    y_size = 1
    D = x_size
    K = h_size

    theta = np.random.normal(0, 1, K + K * D + 1)
    rvalue = np.random.normal(0, 1, (len(train_X), y_size))
    nu = 0.04

    def init_weights(shape):
        """ Weight initialization """
        weights = tf.random_normal(shape, mean=0, stddev=0.00001)
        return tf.Variable(weights)

    def forwardprop(X, w_1, w_2):
        """
        Forward-propagation.
        """
        X = tf.cast(X, tf.float32)
        w_1 = tf.cast(w_1, tf.float32)
        w_2 = tf.cast(w_2, tf.float32)
        h = tf.nn.sigmoid(tf.matmul(X, w_1))
        yhat = tf.matmul(h, w_2)
        return yhat

    g = lambda x: relu(x)

    def nnScore(X, w, V, g):
        X = tf.cast(X, tf.float32)
        w = tf.cast(w, tf.float32)
        V = tf.cast(V, tf.float32)
        return tf.matmul(g((tf.matmul(X, w))), V)

    def relu(x):
        y = x
        #     y[y < 0] = 0
        return y

    def ocnn_obj(theta, X, nu, w1, w2, g, r):
        w = w1
        V = w2

        X = tf.cast(X, tf.float32)
        w = tf.cast(w1, tf.float32)
        V = tf.cast(w2, tf.float32)

        term1 = 0.5 * tf.reduce_sum(w ** 2)
        term2 = 0.5 * tf.reduce_sum(V ** 2)
        term3 = 1 / nu * tf.reduce_mean(relu(r - nnScore(X, w, V, g)))
        term4 = -r

        return term1 + term2 + term3 + term4

    # For testing the OCNN algorithm
    test_X = data_test

    X = tf.placeholder("float32", shape=[None, x_size])

    r = tf.get_variable("r", dtype=tf.float32, shape=(), trainable=False)

    # Weight initializations
    w_1 = init_weights((x_size, h_size))
    w_2 = init_weights((h_size, y_size))

    # Forward propagation
    yhat = forwardprop(X, w_1, w_2)
    predict = tf.argmax(yhat, axis=1)

    # Backward propagation
    cost = ocnn_obj(theta, X, nu, w_1, w_2, g, r)
    updates = tf.train.GradientDescentOptimizer(0.0001).minimize(cost)

    # Run SGD
    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)
    rvalue = 0.1

    for epoch in range(100):
        # Train with each example
        sess.run(updates, feed_dict={X: train_X, r: rvalue})
        rvalue = nnScore(train_X, w_1, w_2, g)
        with sess.as_default():
            rvalue = rvalue.eval()
            rvalue = np.percentile(rvalue, q=100 * 0.04)
        print("Epoch = %d, r = %f"
              % (epoch + 1, rvalue))

    train = nnScore(train_X, w_1, w_2, g)
    test = nnScore(test_X, w_1, w_2, g)
    
    with sess.as_default():
        arrayTrain = train.eval()
        arrayTest = test.eval()

    rstar = rvalue
    sess.close()
    print("Session Closed!!!")

    pos_decisionScore = arrayTrain - rstar
    neg_decisionScore = arrayTest - rstar
    print(pos_decisionScore)

    write_decisionScores2Csv(decision_scorePath, "OC-NN_Relu.csv", pos_decisionScore, neg_decisionScore)

    return [pos_decisionScore, neg_decisionScore]


# =========== Main ==================
# 只做「訓練並存模型」：評估/畫圖由 Dataset/plot_results.py 載入模型完成
train_path = "../../Dataset/train_data/train_data/"



# 訓練資料：只有正常樣本
train_df = pd.read_csv(train_path + "train_dataset.csv", index_col=0)

# Normalize（scaler 只用訓練集 fit）
trans_pipeline = Pipeline([("scaler", MinMaxScaler())])
train_data = trans_pipeline.fit_transform(train_df).astype(np.float32)



# ---- 建圖、訓練一次、存 checkpoint + r* ----
tf.reset_default_graph()
RANDOM_SEED = 42
tf.set_random_seed(RANDOM_SEED)

x_size = train_data.shape[1]
h_size = 32
y_size = 1
nu = 0.04

X = tf.placeholder(tf.float32, shape=[None, x_size], name="X")
r_ph = tf.placeholder(tf.float32, shape=(), name="r_ph")

def init_weights(shape, name):
    weights = tf.random_normal(shape, mean=0, stddev=0.00001)
    return tf.Variable(weights, name=name)

def relu(x):
    return x

g = lambda x: relu(x)

def nnScore(X, w, V, g):
    return tf.matmul(g(tf.matmul(X, w)), V)

def ocnn_obj(X, nu, w, V, g, r):
    term1 = 0.5 * tf.reduce_sum(w ** 2)
    term2 = 0.5 * tf.reduce_sum(V ** 2)
    term3 = 1 / nu * tf.reduce_mean(relu(r - nnScore(X, w, V, g)))
    term4 = -r
    return term1 + term2 + term3 + term4

w_1 = init_weights((x_size, h_size), name="w_1")
w_2 = init_weights((h_size, y_size), name="w_2")

score_op = nnScore(X, w_1, w_2, g)
cost = ocnn_obj(X, nu, w_1, w_2, g, r_ph)
updates = tf.train.GradientDescentOptimizer(0.0001).minimize(cost)



sess = tf.Session()
sess.run(tf.global_variables_initializer())

rvalue = 0.1
for epoch in range(100):
    _, loss_val = sess.run([updates, cost], feed_dict={X: train_data, r_ph: rvalue})
    r_scores = sess.run(score_op, feed_dict={X: train_data})
    rvalue = float(np.percentile(r_scores, q=100 * nu))
    print("Epoch = %3d | loss = %.6f | r = %.6f" % (epoch + 1, loss_val, rvalue))

rstar = float(rvalue)

sess.close()

print("\n===== Training Summary =====")
print("Final r* = %.6f" % rstar)
print("nu = %.4f" % nu)
print("Epochs = 100, lr = 0.0001, h_size = %d" % h_size)
print("Finished training (model NOT saved).")
