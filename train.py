
from PIL import Image
import numpy as np
import pickle
import datetime
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from feature import NPDFeature
from ensemble import AdaBoostClassifier
import os
# 设置当前路径
#os.chdir('F:/DataMining')  

# 这里参数命名写的是matrix，个人习惯，其实是ndarray类型
def extract_NPD(feature_matrix, label_matrix, path):
    label = None
    if (path.split('/')[2] == 'face'):
        label = [1]
    elif (path.split('/')[2] == 'nonface'):
        label = [-1]
    else:
        print('path error!')
        exit(1)
    
    for file in os.listdir(path):
        # 得到图片路径
        file_path = os.path.join(path, file)
        if not os.path.isdir(file_path):
            image_file = Image.open(file_path)
            # 将图片转为24*24的灰度图
            image_file = image_file.convert('L').resize((24, 24))
            # 获取图片的像素值
            data_matrix = np.array(image_file.getdata())
#            data_matrix = data_matrix.reshape(24, 24)
            # 调用feature.py里的NPDFeature抽取特征
            npd_feature = NPDFeature(data_matrix).extract()
            # 将特征作为一行加到参数feature_matrix中
            feature_matrix = np.vstack((feature_matrix, [npd_feature]))
            # 将对应标签加入到label_matrix中
            label_matrix = np.vstack((label_matrix, [label]))
    
    # 返回对应路径下的特征矩阵和标签矩阵(ndarray类型)        
    return feature_matrix, label_matrix
            
def save(data, filename):
        with open(filename, "wb") as f:
            pickle.dump(data, f)
            
def load(filename):
        with open(filename, "rb") as f:
            return pickle.load(f)

def pre_process():
    # 特征大小，计算方法是在feature.py中看到的
    features_size = 576 * (576-1) // 2
    
    # 保存特征用的空array
    face_matrix = np.zeros((0, features_size))
    nonface_matrix = np.zeros((0, features_size))
    
    # 保存标签用的空array
    face_labels = np.ones((0, 1))
    nonface_labels = np.ones((0, 1))
    
    # 抽取特征
    face_matrix, face_labels = extract_NPD(face_matrix, face_labels, 'datasets/original/face')
    nonface_matrix, nonface_labels = extract_NPD(nonface_matrix, nonface_labels, 'datasets/original/nonface')
    
    # 将人脸和非人脸特征存到同一个array
    mix_matrix = np.vstack((face_matrix, nonface_matrix))
    mix_labels = np.vstack((face_labels, nonface_labels))
    
    del face_matrix
    del nonface_matrix
    del face_labels
    del nonface_labels
    
    save(mix_matrix, 'datasets/features/mix_matrix')
    save(mix_labels, 'datasets/features/mix_labels') 
    

if __name__ == "__main__":
    # write your code here
    print(datetime.datetime.now())
    pre_process()
    print(datetime.datetime.now())
    
    X = load('datasets/features/mix_matrix')
    y = load('datasets/features/mix_labels')
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    del X
    del y
    
    print(datetime.datetime.now())
        
    ada_booster = AdaBoostClassifier(DecisionTreeClassifier, 5)
    ada_booster.fit(X_train, y_train)
    
    del X_train
    del y_train
    
    y_test = y_test.reshape(y_test.shape[0])
    
    y_predict = ada_booster.predict(X_test)
    
    report = classification_report(y_test, y_predict, target_names=["nonface", "face"])
    
    file = open('report.txt', 'w')
    file.write(report)
    file.close()
    
    print(report)
    
    print(datetime.datetime.now())
     