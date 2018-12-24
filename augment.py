import librosa
import numpy as np
import matplotlib.pyplot as plt
import os
import soundfile as sf
def get_class_names(path="data/"): 
    class_names=os.listdir(path)
    return class_names
    
def augment():
    inpath="data/"
    outpath="data/"
    class_names=get_class_names(path=inpath)   
    nb_classes=len(class_names)
    for idx,classname in enumerate(class_names):   
        class_files=os.listdir(inpath+classname)
        n_files=len(class_files)
        n_load=n_files
        print(' class name = {:14s} - {:3d}'.format(classname,idx),", ",n_files," files in this class",sep="")
        for idx2,infilename in enumerate(class_files):
            audio_path=inpath+classname+'/'+infilename
            print(audio_path)
            if infilename!='desktop.ini':
                aud,sr=sf.read(audio_path)
                noise=np.random.randn(len(aud))
                data_noise=aud+0.05*noise
                data_roll=np.roll(aud,360000)
                data_stretch=librosa.effects.time_stretch(aud,0.6)
                if len(data_stretch)>len(aud):
                    data_stretch=data_stretch[:len(aud)]
                else:
                    data_stretch=np.pad(data_stretch,(0,max(0,len(aud)-len(data_stretch))),"constant")
                outfile1=outpath+classname+'/'+"noise"+infilename
                outfile2=outpath+classname+'/'+'shift'+infilename
                outfile3=outpath+classname+"/"+"strech"+infilename
                librosa.output.write_wav(outfile1,data_noise,sr)
                librosa.output.write_wav(outfile2,data_roll,sr)
                librosa.output.write_wav(outfile3,data_stretch,sr)                   
                      
augment()
