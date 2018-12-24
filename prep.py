import numpy as np
import librosa
import librosa.display
import os
import soundfile as sf
import matplotlib
import pylab

def get_class_names(path="data/"):  
    class_names=os.listdir(path)
    return class_names

def preprocess_dataset(inpath="data/", outpath="Preproc/"):
    if not os.path.exists(outpath):
        os.mkdir(outpath,0o755);   
    if not os.path.exists('Melgrams/'):
        os.mkdir('Melgrams/',0o755);
    class_names=get_class_names(path=inpath)   
    nb_classes=len(class_names)
    print("class_names = ",class_names)
    for idx,classname in enumerate(class_names):  
        if not os.path.exists(outpath+classname):
            os.mkdir(outpath+classname,0o755);   
        if not os.path.exists('Melgrams/'+classname):
            os.mkdir('Melgrams/'+classname,0o755);
        class_files=os.listdir(inpath+classname)
        n_files=len(class_files)
        print(' class name = {:14s} - {:3d}'.format(classname,idx),", ",n_files," files in this class",sep="")
        print(class_files)
        for idx2,infilename in enumerate(class_files):
            audio_path=inpath+classname+'/'+infilename
            print(audio_path)
            if infilename!='desktop.ini':
                aud,sr=sf.read(audio_path)
                pylab.axis('off') 
                pylab.axes([0., 0., 1., 1.],frameon=False,xticks=[],yticks=[]) 
                melgram=librosa.core.amplitude_to_db(librosa.feature.melspectrogram(aud,sr=sr,n_mels=96))[np.newaxis,np.newaxis,:,:]
                librosa.display.specshow(librosa.core.amplitude_to_db(librosa.feature.melspectrogram(aud,sr=sr,n_mels=96)))
                save_path="Melgrams/"+classname+'/'+infilename+'.png'
                pylab.savefig(save_path,bbox_inches=None,pad_inches=0)
                pylab.close()
                outfile=outpath+classname+'/'+infilename+'.npy'
                np.save(outfile,melgram)

preprocess_dataset()
