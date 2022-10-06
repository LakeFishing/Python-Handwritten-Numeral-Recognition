import cv2
import numpy as np
from keras.datasets import mnist

# 載入 MNIST 資料庫，訓練集60000張，測試集10000張，共10類
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 將訓練集的輸入及輸出資料編碼和格式轉換
x_train = x_train.reshape(x_train.shape[0], -1)
x_train = x_train.astype('float32') / 255
y_train = y_train.astype(np.float32)

# 將測試集的輸入及輸出資料編碼和格式轉換
x_test = x_test.reshape(x_test.shape[0], -1)
x_test = x_test.astype('float32') / 255
y_test = y_test.astype(np.float32)

# 建立 KNN 訓練方法
knn = cv2.ml.KNearest_create()

# 設定參數
knn.setDefaultK(125) # 設定鄰近 K 值
knn.setIsClassifier(True) # 分類或回歸模型選擇

# 進行訓練
knn.train(x_train, cv2.ml.ROW_SAMPLE, y_train)

# 儲存模型
knn.save('mnist_knn01.xml') # 0.9688 5
# 0.9691 1
# 0.9665 10
# 0.9609 25
# 0.9392 125

# 讀取測試集獲得準確率
test_pre = knn.predict(x_test)
test_ret = test_pre[1]
test_ret = test_ret.reshape(-1,)
test_sum = (test_ret == y_test)
acc = test_sum.mean()
print(acc)