import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score
from keras.models import load_model
from keras.utils import np_utils
import os
from os.path import isfile
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from timeit import default_timer as timer
from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn.utils.fixes import signature
import itertools
from sklearn.metrics import classification_report
from keras.utils.vis_utils import plot_model

dic={}

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
    dic[idx]=class_name
    vec = np.zeros(len(class_names))
    vec[idx]=1
    return vec

def decode_class(vec,class_names):  
    return int(np.argmax(vec))

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


def plot_confusion_matrix(cm,classes,title='Confusion matrix',cmap=plt.cm.Blues):
    print('Confusion matrix')
    print(cm)
    plt.imshow(cm,interpolation='nearest',cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks=np.arange(len(classes))
    plt.xticks(tick_marks,classes,rotation=45)
    plt.yticks(tick_marks,classes)
    fmt='d'
    thresh=cm.max()/2.
    for i,j in itertools.product(range(cm.shape[0]),range(cm.shape[1])):
        plt.text(j,i,format(cm[i,j],fmt),horizontalalignment="center",color="white" if cm[i,j]>thresh else "black")
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    np.random.seed(1)
    X_train, Y_train, paths_train, X_test, Y_test, paths_test, class_names, sr=build_datasets(preproc=True)
    print(Y_test.shape)
    print(X_test.shape)
    batch_size=128
    num_pred=X_test.shape[0]
    model=load_model('model.h5')

    print("Running model.evaluate...")
    scores=model.evaluate(X_test,Y_test,verbose=1,batch_size=batch_size)
    print('Test score:',scores[0])
    print('Test accuracy:',scores[1])

    
    print("Running predict_proba...")
    y_scores=model.predict_proba(X_test[0:num_pred,:,:,:],batch_size=batch_size)
    auc_score=roc_auc_score(Y_test, y_scores)
    print("AUC = ",auc_score)

    n_classes=len(class_names)

    print(" Counting mistakes ")
    mistakes=np.zeros(n_classes)
    for i in range(Y_test.shape[0]):
        pred=decode_class(y_scores[i],class_names)
        true=decode_class(Y_test[i],class_names)
        if (pred!=true):
            mistakes[true]+=1
    mistakes_sum=int(np.sum(mistakes))
    print("    Found",mistakes_sum,"mistakes out of",Y_test.shape[0],"attempts")
    print("      Mistakes by class: ",mistakes)

    print("Generating ROC curves...")
    fpr = dict()
    tpr=dict()
    roc_auc=dict()
    for i in range(n_classes):
        fpr[i],tpr[i],_=roc_curve(Y_test[:,i],y_scores[:,i])
        roc_auc[i]=auc(fpr[i],tpr[i])

    plt.figure()
    lw=2
    for i in range(n_classes):
        plt.plot(fpr[i], tpr[i],lw=lw,label='ROC curve of class {0} (area = {1:0.2f})'''.format(i,roc_auc[i]))
    plt.plot([0, 1],[0, 1],color='navy',lw=lw,linestyle='--')
    plt.xlim([0.0,1.0])
    plt.ylim([0.0,1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.show()

    
    Y_pred=model.predict(X_test)
    print(Y_test.shape)
    print(Y_pred.shape)
    for i in range(len(Y_test)):
        Y_test[i]=decode_class(Y_test[i],class_names)
        
        Y_pred[i]=decode_class(Y_pred[i],class_names)
        
    Y_test=np.delete(Y_test,np.s_[1:32],axis=1)
    Y_pred=np.delete(Y_pred,np.s_[1:32],axis=1)
    print(Y_test)
    print(Y_pred)
    print(Y_test.shape)
    print(Y_pred.shape)
    cnf_matrix=confusion_matrix(Y_test,Y_pred)
    np.set_printoptions(precision=2)
    plt.figure(figsize=(15,10),dpi=70)
    plot_confusion_matrix(cnf_matrix, classes=class_names,title='Confusion matrix')
    print(classification_report(Y_test,Y_pred))    
    
