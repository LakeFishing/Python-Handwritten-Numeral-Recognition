import cv2
import numpy as np

def knn_pre():
    knn = cv2.ml.KNearest_load('mnist_knn01.xml')
    cap = cv2.imread('images\output.png')

    # 顏色灰階
    cap = cv2.cvtColor(cap, cv2.COLOR_BGR2GRAY)

    # 從白底黑字轉成黑底白字
    res, cap = cv2.threshold(cap, 127, 255, cv2.THRESH_BINARY_INV)

    # 轉換編碼
    cap = cap.astype(np.float32)

    # 轉換成辨識用的格式
    cap = cap.reshape(-1, ) # 將 28 x 28 的二維陣列轉換成一維陣列
    cap = cap.reshape(1, -1) # 將一維陣列轉換成 1 x 784 的二維陣列
    cap = cap / 255 # 白 255 黑 0 轉換成 白 1 黑 0

    # 進行辨識
    pre = knn.predict(cap)

    # 返回預測結果
    print("KNN：", pre)
    num = str(int(pre[1][0][0]))
    return num

if __name__ == '__main__':
    knn_pre()