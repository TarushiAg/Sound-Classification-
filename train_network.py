from sklearn import preprocessing
import numpy as np
import matplotlib.pyplot as plt
import librosa
import keras
from keras.models import Sequential, Model
from keras.layers import Input, Dense, LSTM, Dropout, Activation
from keras.layers import Convolution2D, MaxPooling2D, Flatten
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import ELU
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
from keras import regularizers
import os
from os.path import isfile

mono=True

dict={}
def get_class_names(path="Preproc/"): 
    class_names=os.listdir(path)
    return class_names

def get_total_files(path="Preproc/",train_percentage=0.8): 
    sum_total=0
    sum_train=0
    sum_test=0
    subdirs=os.listdir(path)
    for subdir in subdirs:
        files=os.listdir(path+subdir)
        n_files=len(files)
        sum_total+=n_files
        n_train=int(train_percentage*n_files)
        n_test=n_files-n_train
        sum_train+=n_train
        sum_test+=n_test
    return sum_total,sum_train,sum_test

def get_sample_dimensions(path='Preproc/'):
    classname=os.listdir(path)[0]
    files=os.listdir(path+classname)
    infilename=files[0]
    audio_path=path+classname+'/'+infilename
    melgram=np.load(audio_path)
    print("   get_sample_dimensions: melgram.shape = ",melgram.shape)
    return melgram.shape
 

def encode_class(class_name,class_names):  
        idx=class_names.index(class_name)
        dict[idx]=class_name
        vec=np.zeros(len(class_names))
        vec[idx]=1
        return vec
    
def shuffle_XY_paths(X,Y,paths):   
    assert(X.shape[0]==Y.shape[0])
    idx=np.array(range(Y.shape[0]))
    np.random.shuffle(idx)
    newX=np.copy(X)
    newY=np.copy(Y)
    newpaths=paths
    for i in range(len(idx)):
        newX[i]=X[idx[i],:,:]
        newY[i]=Y[idx[i],:]
        newpaths[i]=paths[idx[i]]
    return newX,newY,newpaths

def build_datasets(train_percentage=0.8,preproc=False):
    path="Preproc/"
    class_names=get_class_names(path=path)
    print("class_names = ",class_names)
    total_files,total_train,total_test=get_total_files(path=path,train_percentage=train_percentage)
    print("total files = ",total_files)
    nb_classes=len(class_names)
    mel_dims=get_sample_dimensions(path=path)  
    X_train=np.zeros((total_train,mel_dims[1],mel_dims[2],mel_dims[3]))   
    Y_train=np.zeros((total_train,nb_classes))  
    X_test=np.zeros((total_test,mel_dims[1],mel_dims[2],mel_dims[3]))  
    Y_test=np.zeros((total_test,nb_classes))  
    paths_train=[]
    paths_test=[]
    train_count=0
    test_count=0
    for idx,classname in enumerate(class_names):
        this_Y=np.array(encode_class(classname,class_names))
        this_Y=this_Y[np.newaxis,:]
        class_files=os.listdir(path+classname)
        n_files=len(class_files)
        n_load=n_files
        n_train=int(train_percentage*n_load)
        for idx2,infilename in enumerate(class_files[0:n_load]):          
            audio_path=path+classname+'/'+infilename
            melgram=np.load(audio_path)
            sr=44100
            melgram=melgram[:,:,:,0:mel_dims[3]]   
            if (idx2<n_train):
                X_train[train_count,:,:]=melgram
                Y_train[train_count,:]=this_Y
                paths_train.append(audio_path)     
                train_count+=1
            else:
                X_test[test_count,:,:]=melgram
                Y_test[test_count,:]=this_Y
                paths_test.append(audio_path)
                test_count+=1

    print("Shuffling order of data...")
    X_train,Y_train,paths_train=shuffle_XY_paths(X_train,Y_train,paths_train)
    X_test,Y_test,paths_test=shuffle_XY_paths(X_test,Y_test,paths_test)
    return X_train,Y_train,paths_train,X_test,Y_test,paths_test,class_names,sr



def build_model(X,Y,nb_classes):
    nb_filters=32  
    pool_size=(2,2)  
    kernel_size=(3,3)  
    nb_layers=3
    input_shape=(1,X.shape[2],X.shape[3])

    model=Sequential()
    model.add(Convolution2D(nb_filters,kernel_size,kernel_regularizer=regularizers.l2(0.01),input_shape=input_shape,padding="same"))
    model.add(BatchNormalization(axis=1))
    model.add(Activation('relu'))

    for layer in range(nb_layers):
        model.add(Convolution2D(nb_filters,kernel_size[0],padding="same",kernel_regularizer=regularizers.l2(0.01)))
        model.add(BatchNormalization(axis=1))
        model.add(ELU(alpha=1.0))  
        model.add(MaxPooling2D(pool_size=pool_size,dim_ordering="th"))
        model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes))
    model.add(Activation("softmax"))
    return model
    

if __name__ == '__main__':
    np.random.seed(1)
    X_train,Y_train,paths_train,X_test,Y_test,paths_test,class_names,sr=build_datasets(preproc=True)
    print(X_train.shape)
    print(X_test.shape)
    model=build_model(X_train,Y_train,nb_classes=len(class_names))
    model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
    model.summary()
    checkpointer=ModelCheckpoint(filepath=checkpoint_filepath,verbose=1,save_best_only=True)
    batch_size=128
    nb_epoch=100
    history=model.fit(X_train,Y_train,batch_size=batch_size,nb_epoch=nb_epoch,verbose=1,validation_data=(X_test,Y_test),callbacks=[checkpointer])
    score=model.evaluate(X_test,Y_test,verbose=0)
    print(dict)
    model.save('model.h5')
    print('Test score:',score[0])
    print('Test accuracy:',score[1])
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'],loc='upper left')
    plt.show()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train','test'],loc='upper left')
    plt.show()

