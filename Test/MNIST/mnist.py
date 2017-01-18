import os
import struct
import numpy as np
import csv
import pandas as pd
import matplotlib.pyplot as plt
import theano as t
from keras.utils import np_utils


def load_mnist(path, kind='train'):
    """Load MNIST data from `path`"""
    file_path = os.path.join(path, kind+'.csv')
    #print(file_path)

    df = pd.read_csv(file_path)
    #df.head(5)

    labels = df.iloc[:,0]
    images = df.iloc[:,1:]
    
    return labels, images

path = "C:\\kaggle\\mnist"
kind = "train"

y_train, X_train = load_mnist(path, "train")
print('Rows: %d, columns: %d' % (X_train.shape[0], X_train.shape[1]))

y_test, X_test = load_mnist(path, "test")
print('Rows: %d, columns: %d' % (X_test.shape[0], X_test.shape[1]))

new_data = X_train.iloc[1,:].reshape(28,28)
new_data = (new_data-127.5)/255
#print (new_data)
#print(labels)
#print(len(images.iloc[1,:]))



#Rows: 60000, columns: 784
#X_test, y_test = load_mnist('mnist', kind='t10k')
#print('Rows: %d, columns: %d' % (X_test.shape[0], X_test.shape[1]))
#Rows: 10000, columns: 784


#img_data = np.array(new_data, dtype=float)
#print(img_data)
plt.imshow(new_data,cmap='Greys')
plt.show()
#plt.imshow(recovered_dataset[1,:,:],cmap='Greys_r') 


#print(t.config)
print(os.path.expanduser('~/.theanorc.txt'))
t.config.cxx = "C:\\MinGW64\\x86_64-6.3.0-posix-seh-rt_v5-rev0\\mingw64\\bin\\g++.exe"
#print(t.config)

t.config.floatX = 'float32'
X_train = X_train.astype(t.config.floatX) 
X_test = X_test.astype(t.config.floatX)



print('First 3 labels: ', y_train[8])

y_train_ohe = np_utils.to_categorical(y_train) 
print('\nFirst 3 labels (one-hot):\n', y_train_ohe[:8])



from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import SGD


np.random.seed(1) 

model = Sequential()
model.add(Dense(input_dim=X_train.shape[1], 
                output_dim=50, 
                init='uniform', 
                activation='tanh'))

model.add(Dense(input_dim=50, 
                output_dim=50, 
                init='uniform', 
                activation='tanh'))

model.add(Dense(input_dim=50, 
                output_dim=y_train_ohe.shape[1], 
                init='uniform', 
                activation='softmax'))

print('Start Compiling...')
sgd = SGD(lr=0.001, decay=1e-7, momentum=.9)
model.compile(loss='categorical_crossentropy', optimizer=sgd)

print('X_train shape',X_train.shape[0])
print('Y_train shape',y_train_ohe.shape[1])

print('Start Training...')
#start to train
model.fit(X_train, 
          y_train_ohe, 
          nb_epoch=50, 
          batch_size=200, 
          verbose=1, 
          validation_split=0.1) 
#          show_accuracy=True)
