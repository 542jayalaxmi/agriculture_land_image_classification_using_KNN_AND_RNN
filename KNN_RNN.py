
from tkinter import messagebox
from tkinter import *
from tkinter import simpledialog
import tkinter
from tkinter import filedialog
import matplotlib.pyplot as plt
import numpy as np
from tkinter.filedialog import askopenfilename
import os
import re
import numpy as np 
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn import svm
from sklearn.model_selection import train_test_split

import cv2
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.models import model_from_json
import pickle
from sklearn.decomposition import PCA
from keras.layers import Dense,Activation,BatchNormalization,Dropout
import keras.layers

main = tkinter.Tk()
main.title("Agricultural Land Image Classification using KNN and compare with Recurrent Neural network for Feature Extraction")
main.geometry("1300x1200")

global filename
global knn_acc,rnn_acc
global classifier
global X,Y
global knn_x,knn_y
global X_train, X_test, y_train, y_test

labels = ['Urban Land','Agricultural Land','Range Land','Forest Land']

def upload():
    global filename
    filename = filedialog.askopenfilename(initialdir="model")
    pathlabel.config(text=filename)
    text.delete('1.0', END)
    text.insert(END,filename+" loaded\n");

def extractFeatures():
    global X,Y
    global knn_x,knn_y
    global X_train, X_test, y_train, y_test
    text.delete('1.0', END)
    X = np.load('model/X.txt.npy')
    Y = np.load('model/Y.txt.npy')
    knn_y = Y

    knn_x = np.reshape(X, (X.shape[0],(X.shape[1]*X.shape[2]*X.shape[3])))
    text.insert(END,"Number of features in images before applying PCA feature extraction : "+str(knn_x.shape[1])+"\n")
    pca = PCA(n_components = 500)
    knn_x = pca.fit_transform(knn_x)
    text.insert(END,"Number of features in images after applying PCA feature extraction : "+str(knn_x.shape[1])+"\n")
    X_train, X_test, y_train, y_test = train_test_split(knn_x, knn_y, test_size=0.2)
    text.insert(END,"Total Images Found in dataset : "+str(len(X))+"\n")
    

def runKNN():
    text.delete('1.0', END)
    global knn_acc
    global X_train, X_test, y_train, y_test
    cls = svm.SVC()
    cls.fit(X_train, y_train)
    prediction_data = cls.predict(X_test) 
    knn_acc = accuracy_score(y_test,prediction_data)*100
    text.insert(END,"KNN Accuracy After applying PCA Feature Extraction : "+str(knn_acc)+"\n")

def runRNN():
    global X,Y
    global rnn_acc
    global classifier
    Y1 = to_categorical(Y)
    if os.path.exists('model/model.json'):
        with open('model/model.json', "r") as json_file:
            loaded_model_json = json_file.read()
            classifier = model_from_json(loaded_model_json)
        classifier.load_weights("model/model_weights.h5")
        classifier._make_predict_function()   
        print(classifier.summary())
        f = open('model/history.pckl', 'rb')
        data = pickle.load(f)
        f.close()
        acc = data['accuracy']
        rnn_acc = acc[19] * 100
        text.insert(END,"RNN Accuracy After applying PCA Feature Extraction : "+str(rnn_acc)+"\n")
    else:
        X = np.reshape(X, (X.shape[0],X.shape[1],(X.shape[2]*X.shape[3])))
        classifier = Sequential()
        classifier.add(keras.layers.LSTM(100,input_shape=(64, 192)))
        classifier.add(Dropout(0.5))
        classifier.add(Dense(100, activation='relu'))
        classifier.add(Dense(4, activation='softmax'))        
        classifier.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        hist = classifier.fit(X, Y1, epochs=20, batch_size=32)
        print(classifier.summary())
        classifier.save_weights('model/model_weights.h5')            
        model_json = classifier.to_json()
        with open("model/model.json", "w") as json_file:
            json_file.write(model_json)
        f = open('model/history.pckl', 'wb')
        pickle.dump(hist.history, f)
        f.close()
        f = open('model/history.pckl', 'rb')
        data = pickle.load(f)
        f.close()
        acc = data['accuracy']
        rnn_acc = acc[19] * 100
        text.insert(END,"RNN Accuracy After applying PCA Feature Extraction : "+str(rnn_acc)+"\n")
    
    
def graph():
    height = [knn_acc,rnn_acc]
    bars = ('KNN Accuracy', 'RNN Accuracy')
    y_pos = np.arange(len(bars))
    plt.bar(y_pos, height)
    plt.xticks(y_pos, bars)
    plt.show()

def predict():
    filename = filedialog.askopenfilename(initialdir="sampleImages")
    image = cv2.imread(filename)
    img = cv2.resize(image, (64,64))
    im2arr = np.array(img)
    im2arr = im2arr.reshape(1,64,64,3)
    img = np.asarray(im2arr)
    img = img.astype('float32')
    img = img/255
    img = np.reshape(img, (img.shape[0],img.shape[1],(img.shape[2]*img.shape[3])))
    preds = classifier.predict(img)
    predict = np.argmax(preds)

    img = cv2.imread(filename)
    img = cv2.resize(img, (800,400))
    cv2.putText(img, 'Land Classified as : '+labels[predict], (10, 25),  cv2.FONT_HERSHEY_SIMPLEX,0.7, (0, 255, 0), 2)
    cv2.imshow('Land Classified as : '+labels[predict], img)
    cv2.waitKey(0)
    
font = ('times', 14, 'bold')
title = Label(main, text='Agricultural Land Image Classification using KNN and compare with Recurrent Neural network for Feature Extraction')
title.config(bg='yellow3', fg='white')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=0,y=5)

font1 = ('times', 13, 'bold')
uploadButton = Button(main, text="Upload Land Satellite Images", command=upload)
uploadButton.place(x=50,y=100)
uploadButton.config(font=font1)  

pathlabel = Label(main)
pathlabel.config(bg='brown', fg='white')  
pathlabel.config(font=font1)           
pathlabel.place(x=460,y=100)

featuresButton = Button(main, text="Extract Features from Images", command=extractFeatures)
featuresButton.place(x=50,y=150)
featuresButton.config(font=font1) 

svmButton = Button(main, text="Train & Validate KNN Algorithm", command=runKNN)
svmButton.place(x=310,y=150)
svmButton.config(font=font1) 

nn = Button(main, text="Train & Validate RNN", command=runRNN)
nn.place(x=650,y=150)
nn.config(font=font1) 

graphbutton = Button(main, text="Accuracy Comparison Graph", command=graph)
graphbutton.place(x=50,y=200)
graphbutton.config(font=font1) 

predictb = Button(main, text="Upload Test Image & Clasify Lands", command=predict)
predictb.place(x=310,y=200)
predictb.config(font=font1) 


font1 = ('times', 12, 'bold')
text=Text(main,height=20,width=150)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=10,y=250)
text.config(font=font1)


main.config(bg='burlywood2')
main.mainloop()
