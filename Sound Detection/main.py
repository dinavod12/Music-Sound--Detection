#First getting Data
import librosa
import matplotlib.pyplot as plt
filename = "Desktop/Data/genres_original/blues/blues.00002.wav"
audio_data,sample_rate = librosa.load(filename)
plt.plot(audio_data)

#Ploting Data using Pandas

import pandas as pd
meta_data = pd.read_csv( "Desktop/Data/features_30_sec.csv")
meta_data.head()

import numpy as np
import pandas as pd
import librosa
from tqdm import tqdm
import os

audio_path = "Desktop/Data/genres_original"

def feature_extractor(file):
    audio_data1,sample_rate1 = librosa.load(file_name, res_type = "kaiser_best")
    mfccs_features = librosa.feature.mfcc(y=audio_data1,sr=sample_rate1,n_mfcc= 40)
    mfccs_scaled_features = np.mean(mfccs_features.T,axis = 0)
    return mfccs_scaled_features


extracted_features = []
for index_num,row in tqdm(meta_data.iterrows()):
    try:
       file_name = os.path.join(os.path.abspath(audio_path),str(row["label"]) + "/",str(row["filename"]))
       file_class_labels = row["label"]
       data = feature_extractor(file_name)
       extracted_features.append([data,file_class_labels])
    except Exception as e:
        print(e)

dataframe = pd.DataFrame(extracted_features , columns = ["feature","class"])

#training model

import numpy as np
X = np.array(dataframe["feature"].tolist())
Y = np.array(dataframe["class"].tolist())
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
Y = le.fit_transform(Y)
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size = 0.2 ,random_state = 0)


from sklearn.ensemble import RandomForestClassifier
RN = RandomForestClassifier(n_estimators = 800 , criterion = "entropy" , random_state = 0)
RN.fit(X_train,Y_train)
y_pred = RN.predict(X_test)


#from sklearn.neighbors import KNeighborsClassifier
#KN = KNeighborsClassifier(n_neighbors = 5 , metric = "minkowski" , p=2)
#KN.fit(X_test,y_pred)
#y_pred1 = KN.predict(X_test)

#from sklearn.preprocessing import StandardScaler
#ss = StandardScaler()
#X_train[0] = ss.fit_transform(X_train[0])
#X_test[0] = ss.fit_transform(X_test[0])
#Y_train[0] = ss.fit_transform(Y_train[0])
#Y_test[0] = ss.fit_transform(Y_test[0])

import tensorflow as tf
ann = tf.keras.models.Sequential()
ann.add(tf.keras.layers.Dense(units = 200, activation = "relu"))
ann.add(tf.keras.layers.Dense(units = 200, activation = "relu"))
ann.add(tf.keras.layers.Dense(units = 1, activation = "sigmoid"))
ann.compile(optimizer = "adam",metrics = ["accuracy"] , loss = "categorical_crossentropy")
ann.fit(X_test,y_pred,batch_size = 32, epochs=100)
y_pred1 = ann.predict(X_test)

"""
Epoch 1/100
7/7 [==============================] - 1s 6ms/step - loss: 0.0000e+00 - accuracy: 0.0850
Epoch 2/100
7/7 [==============================] - 0s 5ms/step - loss: 0.0000e+00 - accuracy: 0.0850
Epoch 3/100
7/7 [==============================] - 0s 5ms/step - loss: 0.0000e+00 - accuracy: 0.0850
Epoch 4/100
7/7 [==============================] - 0s 5ms/step - loss: 0.0000e+00 - accuracy: 0.0850
Epoch 5/100
7/7 [==============================] - 0s 5ms/step - loss: 0.0000e+00 - accuracy: 0.0850
Epoch 6/100
7/7 [==============================] - 0s 5ms/step - loss: 0.0000e+00 - accuracy: 0.0850
Epoch 7/100
7/7 [==============================] - 0s 5ms/step - loss: 0.0000e+00 - accuracy: 0.0850
Epoch 8/100
7/7 [==============================] - 0s 5ms/step - loss: 0.0000e+00 - accuracy: 0.0850
Epoch 9/100
7/7 [==============================] - 0s 5ms/step - loss: 0.0000e+00 - accuracy: 0.0850
Epoch 10/100
7/7 [==============================] - 0s 5ms/step - loss: 0.0000e+00 - accuracy: 0.0850
Epoch 11/100
7/7 [==============================] - 0s 5ms/step - loss: 0.0000e+00 - accuracy: 0.0850
Epoch 12/100
7/7 [==============================] - 0s 4ms/step - loss: 0.0000e+00 - accuracy: 0.0850
Epoch 13/100
7/7 [==============================] - 0s 5ms/step - loss: 0.0000e+00 - accuracy: 0.0850
Epoch 14/100
7/7 [==============================] - 0s 5ms/step - loss: 0.0000e+00 - accuracy: 0.0850
Epoch 15/100
7/7 [==============================] - 0s 4ms/step - loss: 0.0000e+00 - accuracy: 0.0850
Epoch 16/100
7/7 [==============================] - 0s 4ms/step - loss: 0.0000e+00 - accuracy: 0.0850
Epoch 17/100
7/7 [==============================] - 0s 5ms/step - loss: 0.0000e+00 - accuracy: 0.0850
Epoch 18/100
7/7 [==============================] - 0s 5ms/step - loss: 0.0000e+00 - accuracy: 0.0850
Epoch 19/100
7/7 [==============================] - 0s 4ms/step - loss: 0.0000e+00 - accuracy: 0.0850
Epoch 20/100
7/7 [==============================] - 0s 4ms/step - loss: 0.0000e+00 - accuracy: 0.0850
Epoch 21/100
7/7 [==============================] - 0s 5ms/step - loss: 0.0000e+00 - accuracy: 0.0850
Epoch 22/100
7/7 [==============================] - 0s 5ms/step - loss: 0.0000e+00 - accuracy: 0.0850
Epoch 23/100
7/7 [==============================] - 0s 5ms/step - loss: 0.0000e+00 - accuracy: 0.0850
Epoch 24/100
7/7 [==============================] - 0s 5ms/step - loss: 0.0000e+00 - accuracy: 0.0850
Epoch 25/100
7/7 [==============================] - 0s 5ms/step - loss: 0.0000e+00 - accuracy: 0.0850
Epoch 26/100
7/7 [==============================] - 0s 5ms/step - loss: 0.0000e+00 - accuracy: 0.0850
Epoch 27/100
7/7 [==============================] - 0s 5ms/step - loss: 0.0000e+00 - accuracy: 0.0850
Epoch 28/100
7/7 [==============================] - 0s 5ms/step - loss: 0.0000e+00 - accuracy: 0.0850
Epoch 29/100
7/7 [==============================] - 0s 5ms/step - loss: 0.0000e+00 - accuracy: 0.0850
Epoch 30/100
7/7 [==============================] - 0s 5ms/step - loss: 0.0000e+00 - accuracy: 0.0850
Epoch 31/100
7/7 [==============================] - 0s 5ms/step - loss: 0.0000e+00 - accuracy: 0.0850
Epoch 32/100
7/7 [==============================] - 0s 6ms/step - loss: 0.0000e+00 - accuracy: 0.0850
Epoch 33/100
7/7 [==============================] - 0s 5ms/step - loss: 0.0000e+00 - accuracy: 0.0850
Epoch 34/100
7/7 [==============================] - 0s 6ms/step - loss: 0.0000e+00 - accuracy: 0.0850
Epoch 35/100
7/7 [==============================] - 0s 5ms/step - loss: 0.0000e+00 - accuracy: 0.0850
Epoch 36/100
7/7 [==============================] - 0s 5ms/step - loss: 0.0000e+00 - accuracy: 0.0850
Epoch 37/100
7/7 [==============================] - 0s 5ms/step - loss: 0.0000e+00 - accuracy: 0.0850
Epoch 38/100
7/7 [==============================] - 0s 4ms/step - loss: 0.0000e+00 - accuracy: 0.0850
Epoch 39/100
7/7 [==============================] - 0s 4ms/step - loss: 0.0000e+00 - accuracy: 0.0850
Epoch 40/100
7/7 [==============================] - 0s 5ms/step - loss: 0.0000e+00 - accuracy: 0.0850
Epoch 41/100
7/7 [==============================] - 0s 4ms/step - loss: 0.0000e+00 - accuracy: 0.0850
Epoch 42/100
7/7 [==============================] - 0s 4ms/step - loss: 0.0000e+00 - accuracy: 0.0850
Epoch 43/100
7/7 [==============================] - 0s 5ms/step - loss: 0.0000e+00 - accuracy: 0.0850
Epoch 44/100
7/7 [==============================] - 0s 5ms/step - loss: 0.0000e+00 - accuracy: 0.0850
Epoch 45/100
7/7 [==============================] - 0s 5ms/step - loss: 0.0000e+00 - accuracy: 0.0850
Epoch 46/100
7/7 [==============================] - 0s 4ms/step - loss: 0.0000e+00 - accuracy: 0.0850
Epoch 47/100
7/7 [==============================] - 0s 4ms/step - loss: 0.0000e+00 - accuracy: 0.0850
Epoch 48/100
7/7 [==============================] - 0s 5ms/step - loss: 0.0000e+00 - accuracy: 0.0850
Epoch 49/100
7/7 [==============================] - 0s 5ms/step - loss: 0.0000e+00 - accuracy: 0.0850
Epoch 50/100
7/7 [==============================] - 0s 4ms/step - loss: 0.0000e+00 - accuracy: 0.0850
Epoch 51/100
7/7 [==============================] - 0s 5ms/step - loss: 0.0000e+00 - accuracy: 0.0850
Epoch 52/100
7/7 [==============================] - 0s 5ms/step - loss: 0.0000e+00 - accuracy: 0.0850
Epoch 53/100
7/7 [==============================] - 0s 6ms/step - loss: 0.0000e+00 - accuracy: 0.0850
Epoch 54/100
7/7 [==============================] - 0s 4ms/step - loss: 0.0000e+00 - accuracy: 0.0850
Epoch 55/100
7/7 [==============================] - 0s 4ms/step - loss: 0.0000e+00 - accuracy: 0.0850
Epoch 56/100
7/7 [==============================] - 0s 5ms/step - loss: 0.0000e+00 - accuracy: 0.0850
Epoch 57/100
7/7 [==============================] - 0s 5ms/step - loss: 0.0000e+00 - accuracy: 0.0850
Epoch 58/100
7/7 [==============================] - 0s 4ms/step - loss: 0.0000e+00 - accuracy: 0.0850
Epoch 59/100
7/7 [==============================] - 0s 5ms/step - loss: 0.0000e+00 - accuracy: 0.0850
Epoch 60/100
7/7 [==============================] - 0s 5ms/step - loss: 0.0000e+00 - accuracy: 0.0850
Epoch 61/100
7/7 [==============================] - 0s 4ms/step - loss: 0.0000e+00 - accuracy: 0.0850
Epoch 62/100
7/7 [==============================] - 0s 4ms/step - loss: 0.0000e+00 - accuracy: 0.0850
Epoch 63/100
7/7 [==============================] - 0s 4ms/step - loss: 0.0000e+00 - accuracy: 0.0850
Epoch 64/100
7/7 [==============================] - 0s 5ms/step - loss: 0.0000e+00 - accuracy: 0.0850
Epoch 65/100
7/7 [==============================] - 0s 5ms/step - loss: 0.0000e+00 - accuracy: 0.0850
Epoch 66/100
7/7 [==============================] - 0s 5ms/step - loss: 0.0000e+00 - accuracy: 0.0850
Epoch 67/100
7/7 [==============================] - 0s 4ms/step - loss: 0.0000e+00 - accuracy: 0.0850
Epoch 68/100
7/7 [==============================] - 0s 5ms/step - loss: 0.0000e+00 - accuracy: 0.0850
Epoch 69/100
7/7 [==============================] - 0s 4ms/step - loss: 0.0000e+00 - accuracy: 0.0850
Epoch 70/100
7/7 [==============================] - 0s 4ms/step - loss: 0.0000e+00 - accuracy: 0.0850
Epoch 71/100
7/7 [==============================] - 0s 4ms/step - loss: 0.0000e+00 - accuracy: 0.0850
Epoch 72/100
7/7 [==============================] - 0s 5ms/step - loss: 0.0000e+00 - accuracy: 0.0850
Epoch 73/100
7/7 [==============================] - 0s 5ms/step - loss: 0.0000e+00 - accuracy: 0.0850
Epoch 74/100
7/7 [==============================] - 0s 5ms/step - loss: 0.0000e+00 - accuracy: 0.0850
Epoch 75/100
7/7 [==============================] - 0s 5ms/step - loss: 0.0000e+00 - accuracy: 0.0850
Epoch 76/100
7/7 [==============================] - 0s 5ms/step - loss: 0.0000e+00 - accuracy: 0.0850
Epoch 77/100
7/7 [==============================] - 0s 5ms/step - loss: 0.0000e+00 - accuracy: 0.0850
Epoch 78/100
7/7 [==============================] - 0s 5ms/step - loss: 0.0000e+00 - accuracy: 0.0850
Epoch 79/100
7/7 [==============================] - 0s 4ms/step - loss: 0.0000e+00 - accuracy: 0.0850
Epoch 80/100
7/7 [==============================] - 0s 5ms/step - loss: 0.0000e+00 - accuracy: 0.0850
Epoch 81/100
7/7 [==============================] - 0s 5ms/step - loss: 0.0000e+00 - accuracy: 0.0850
Epoch 82/100
7/7 [==============================] - 0s 4ms/step - loss: 0.0000e+00 - accuracy: 0.0850
Epoch 83/100
7/7 [==============================] - 0s 6ms/step - loss: 0.0000e+00 - accuracy: 0.0850
Epoch 84/100
7/7 [==============================] - 0s 5ms/step - loss: 0.0000e+00 - accuracy: 0.0850
Epoch 85/100
7/7 [==============================] - 0s 4ms/step - loss: 0.0000e+00 - accuracy: 0.0850
Epoch 86/100
7/7 [==============================] - 0s 5ms/step - loss: 0.0000e+00 - accuracy: 0.0850
Epoch 87/100
7/7 [==============================] - 0s 5ms/step - loss: 0.0000e+00 - accuracy: 0.0850
Epoch 88/100
7/7 [==============================] - 0s 4ms/step - loss: 0.0000e+00 - accuracy: 0.0850
Epoch 89/100
7/7 [==============================] - 0s 4ms/step - loss: 0.0000e+00 - accuracy: 0.0850
Epoch 90/100
7/7 [==============================] - 0s 4ms/step - loss: 0.0000e+00 - accuracy: 0.0850
Epoch 91/100
7/7 [==============================] - 0s 5ms/step - loss: 0.0000e+00 - accuracy: 0.0850
Epoch 92/100
7/7 [==============================] - 0s 4ms/step - loss: 0.0000e+00 - accuracy: 0.0850
Epoch 93/100
7/7 [==============================] - 0s 4ms/step - loss: 0.0000e+00 - accuracy: 0.0850
Epoch 94/100
7/7 [==============================] - 0s 4ms/step - loss: 0.0000e+00 - accuracy: 0.0850
Epoch 95/100
7/7 [==============================] - 0s 4ms/step - loss: 0.0000e+00 - accuracy: 0.0850
Epoch 96/100
7/7 [==============================] - 0s 4ms/step - loss: 0.0000e+00 - accuracy: 0.0850
Epoch 97/100
7/7 [==============================] - 0s 5ms/step - loss: 0.0000e+00 - accuracy: 0.0850
Epoch 98/100
7/7 [==============================] - 0s 5ms/step - loss: 0.0000e+00 - accuracy: 0.0850
Epoch 99/100
7/7 [==============================] - 0s 4ms/step - loss: 0.0000e+00 - accuracy: 0.0850
Epoch 100/100
7/7 [==============================] - 0s 4ms/step - loss: 0.0000e+00 - accuracy: 0.0850
7/7 [==============================] - 0s 4ms/step

"""
#Finding Accuracy
   
from sklearn.metrics import accuracy_score
x = accuracy_score(y_pred,y_pred1)
print(x)

#Predicting on different value

list1 = []
audio_data2,sample_rate2 = librosa.load("Desktop/cats.wav", res_type = "kaiser_best")
mfccs_features1 = librosa.feature.mfcc(y=audio_data2,sr=sample_rate2,n_mfcc= 40)
mfccs_scaled_features = np.mean(mfccs_features1.T,axis = 0)
list1.append(mfccs_scaled_features)
predict = RN.predict(list1)
pt = le.inverse_transform(predict)
print(pt)   