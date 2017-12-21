import pickle
import numpy as np
import math

class AdaBoostClassifier:
    '''A simple AdaBoost Classifier.'''

    alpha_array = None
    n_weakers_limit = None
    weak_classifier = None
    weak_classifiers = []
    
    def __init__(self, weak_classifier, n_weakers_limit):
        '''Initialize AdaBoostClassifier

        Args:
            weak_classifier: The class of weak classifier, which is recommend to be sklearn.tree.DecisionTreeClassifier.
            n_weakers_limit: The maximum number of weak classifier the model can use.
        '''
        
        self.n_weakers_limit = n_weakers_limit
        self.weak_classifier = weak_classifier
        
        for m in range(n_weakers_limit):
            self.weak_classifiers.append(weak_classifier(max_depth = 3))
        

    def is_good_enough(self):
        '''Optional'''
        pass


    def fit(self,X,y):
        '''Build a boosted classifier from the training set (X, y).

        Returns:
            X: An ndarray indicating the samples to be trained, which shape should be (n_samples,n_features).
            y: An ndarray indicating the ground-truth labels correspond to X, which shape should be (n_samples,1).
        '''
        w = 1/X.shape[0] * np.ones((X.shape[0]))
        alpha_array = np.empty((self.n_weakers_limit))
        
        for m in range(self.n_weakers_limit):
            # 训练弱分类器
            self.weak_classifiers[m].fit(X, y, sample_weight = w)
            # 用弱分类器预测训练集
            y_predict = np.sign(self.weak_classifiers[m].predict(X))

            error_rate = 0
            for i in range(y_predict.shape[0]):
                if (y_predict[i] != y[i][0]):
                    error_rate += w[i]
            
            if (error_rate > 0.5):
                print('There may be some error!')
                print('Error rate is bigger than 0.5')
                print('Training has stopped')
                break
            
            alpha = math.log((1 - error_rate) / error_rate) / 2
            alpha_array[m] = alpha
            
            Zm = 0
            exp_array = np.empty(w.shape)
            for i in range(w.shape[0]):
                e = math.exp(-alpha * y[i][0] * y_predict[i])  # y为二维数组，y_predict为一维数组
                exp_array[i] = e
                Zm += w[i] * e
                
            for i in range(w.shape[0]):
                w[i] = w[i] * exp_array[i] / Zm
        
        self.alpha_array = alpha_array
        

    def predict_scores(self, X):
        '''Calculate the weighted sum score of the whole base classifiers for given samples.

        Args:
            X: An ndarray indicating the samples to be predicted, which shape should be (n_samples,n_features).

        Returns:
            An one-dimension ndarray indicating the scores of differnt samples, which shape should be (n_samples,1).
        '''
        predict_result = np.zeros((X.shape[0]))
        
        for m in range(self.n_weakers_limit):
            y_predict = np.sign(self.weak_classifiers[m].predict(X))
            
            predict_result += y_predict * self.alpha_array[m]
        
        return predict_result
    

    def predict(self, X, threshold=0):
        '''Predict the catagories for geven samples.

        Args:
            X: An ndarray indicating the samples to be predicted, which shape should be (n_samples,n_features).
            threshold: The demarcation number of deviding the samples into two parts.

        Returns:
            An ndarray consists of predicted labels, which shape should be (n_samples,1).
        '''
        predict_result = np.zeros((X.shape[0]))
        
        for m in range(self.n_weakers_limit):
            y_predict = np.sign(self.weak_classifiers[m].predict(X))
            
            predict_result += y_predict * self.alpha_array[m]
            
        for i in range(X.shape[0]):
            if (predict_result[i] > threshold):
                predict_result[i] = 1
            elif (predict_result[i] < threshold):
                predict_result[i] = -1
            
        return predict_result


    @staticmethod
    def save(model, filename):
        with open(filename, "wb") as f:
            pickle.dump(model, f)

    @staticmethod
    def load(filename):
        with open(filename, "rb") as f:
            return pickle.load(f)
