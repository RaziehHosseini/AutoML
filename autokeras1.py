
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 08:55:38 2019

@author: s01hosse
"""
import sys
from keras.datasets import mnist, cifar10, cifar100, cifar, fashion_mnist
from autokeras.image.image_supervised import ImageClassifier
from sklearn.metrics import classification_report
import numpy as np
from keras.models import load_model
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot

def data_info(x_train, y_train,x_test, y_test):
    print ('dataset size: ', x_train.shape[0] + x_test.shape[0])
    print ('training set:', np.shape(x_train))
    print ('test set:', np.shape(x_test))
    print ('training set label:', np.shape(y_train))
    print ('training set label:', np.shape(y_test))
    
def preprocess(x_train, y_train,x_test, y_test):
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255

def save_report(report, seconds, score):
    p = os.path.sep.join('report', "{}.txt".format(seconds))
    f = open (p, "w")
    f.write(report)
    f.write("\nscore: {}".format(score))
    f.close()
    
    
def autokeras_mnist(time, **kwargs):
    labelNames = ["Class 0", "Class 1", "Class 3",
                  "Class 4", "Class 5", "Class 6", 
                  "Class 7", "Class 8", "Class 9"]
    for seconds in time:
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        data_info(x_train, y_train,x_test, y_test)
        x_train = x_train.reshape(x_train.shape +(1, ))
        x_test = x_test.reshape(x_test.shape +(1, ))
        preprocess(x_train, y_train,x_test, y_test)
        
        clf = ImageClassifier(verbose=True, searcher_args={'trainer_args':{'max_iter_num':5}})
        clf.fit(x_train, y_train, time_limit=seconds)
        clf.final_fit(x_train, y_train, retrain=True)
        score = clf.evaluate(x_test, y_test)
        predictions = clf.predict(x_test)
        reports =  classification_report( y_test, predictions, labelNames )
        print(score)
        save_report(reports, seconds, score)
        clf.load_searcher().load_best_model().produce_keras_model().dave("models/ak_mnist_{}.h5".format(seconds))
        clf.export_keras_model("models/ak_mnist_{}.h5".format(seconds))
        clf.export_autokeras_model("models/ak_mnist_{}.h5".format(seconds))
        MODEL_NAME = 'models/ak_mnist_'+np.str(seconds)+'.h5'
        model = load_model(MODEL_NAME)
        SVG(model_to_dot(model).create(prog='dot', format='svg'))

def autokeras_cifar10(time, **kwargs):
    """autokeras on cifar10"""
    
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    data_info(x_train, y_train,x_test, y_test)
    x_train = x_train.reshape(x_train.shape +(1, ))
    x_test = x_test.reshape(x_test.shape +(1, ))
    
    clf = ImageClassifier(verbose=True, searcher_args={'trainer_args':{'max_iter_num':5}})
    clf.fit(x_train, y_train, time_limit=time)
    clf.final_fit(x_train, y_train, retrain=True)
    y = clf.evaluate(x_test, y_test)
    
    print(y)
    clf.load_searcher().load_best_model().produce_keras_model().dave('models/ak_mnist_1.h5')

if __name__=='__main__':
    TRAINING_TIME = [ # input time in seconds
            1 * 60 * 60,
            2 * 60 * 60,
            4 * 60 * 60,
            8 * 60 * 60,
            10 * 60 * 60,
            16 * 60 * 60,
            20 * 60 * 60]
    kwargs = {'time' : TRAINING_TIME}
    autokeras_mnist(**kwargs)
    autokeras_cifar10(**kwargs)
