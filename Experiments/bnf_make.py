#Libraries
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import IPython.display as ipd
import os
#from BottleneckFeatureExtractor import bottleneck2posterior
import sys
sys.path.insert(1, '/u/home/a/asarkar/project-jflint/Language_diarization/BUT/')
import audio2bottleneck

class Audio:
    def __init__(self,audio_path):
        self.audio_path = audio_path
        self.x,self.sr = librosa.load(audio_path,sr=None)

    def mfcc(self):
        mfccs = librosa.feature.mfcc(y=self.x, sr=self.sr, n_mfcc=39)
        return mfccs

class create_bnf(Audio):
    def __init__(self, audio_path, nn):
        super().__init__(audio_path)
        self.nn = nn
        f_name = str(self.audio_path.split('/')[-1].split('.')[0])
        f_path = "/u/home/a/asarkar/scratch/BNF_Data_split1"
        self.bn_file=f_path+'/BNF_'+f_name+".csv"
        self.vad_file=""
        self.post_file=f_path+'/'+f_name+".h5"
        
    def bnf(self):
        audio2bottleneck.compute(self.nn,self.audio_path,self.bn_file,self.vad_file)
        #bottleneck2posterior.compute(self.nn, self.bn_file, self.post_file)
        #print("files created")
        #print("BN file", self.bn_file)
        #print("post file",self.post_file)

def Save_BNFs(directory):
    nn = "BabelMulti"
    for path in os.listdir(directory):
        audio_path = os.path.join(directory, path)
        if os.path.isfile(audio_path):
            print(audio_path)
            create_bnf(audio_path, nn).bnf()


directory = "/u/home/a/asarkar/scratch/vad_test/"
Save_BNFs(directory)