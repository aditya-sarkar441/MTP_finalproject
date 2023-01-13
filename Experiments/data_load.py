#Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import librosa
import librosa.display
import sys
sys.path.insert(1, '/u/home/a/asarkar/project-jflint/Language_diarization/BUT/')
import bottleneck2posterior
import uvector
import audio2bottleneck
import splitfolders
from pprint import pprint
import os

class prepare_data:
    def __init__(self,path):
        self.path = path
    
    def split_files(self):
        output = "/u/scratch/a/asarkar/BNF_Data_split/"
        print(self.path)
        splitfolders.fixed(self.path, output=output, seed=1337, fixed=(50, 50, 50), oversample=False)
        return output

class create_mfcc:
    def __init__(self,audio_path,audio, sr):
        self.audio = audio
        self.sr = sr
        self.ap = audio_path
   
    def mfcc(self):
        mfccs = librosa.feature.mfcc(y=self.audio, sr=self.sr, n_mfcc=39)
        mfccs = mfccs.reshape(-1,39)
        self.ap = self.ap.split('.')[0]
        ouputfile = '/u/scratch/a/asarkar/chunks_results/MFCC/'+'MFCC_'+self.ap.split('/')[-1].split('.')[0]+'.csv'
        print(ouputfile,"done")
        file = open(ouputfile, 'w+')
        np.savetxt(file, mfccs, delimiter=",") #save MFCCs as .csv
        file.close()
        return mfccs
        
class create_bnf:
    def __init__(self, audio_path, nn):
        self.ap = audio_path
        self.nn = nn
        self.bn_file='/u/scratch/a/asarkar/chunks_results/BNF/BNF_'+str(self.ap.split('/')[-1].split('.')[0])+"_bn.csv"
        self.vad_file=""
        
    def bnf(self):
        audio2bottleneck.compute(self.nn,self.ap,self.bn_file,self.vad_file)
        print("files created")
        print("BNF file", self.bn_file)
        
class create_uvector:
    def __init__(self,folpath,typ):
        self.folpath = folpath
        self.typ = typ
        
    def pred_bilstm(self):
        if self.typ == 'bnf':
            Tru, pred = uvector.model_use(self.folpath,self.typ).calculate_quants('/u/home/a/asarkar/project-jflint/Language_diarization/BUT/bnf_uvector_40.pth')
            print("LOADED",'/u/home/a/asarkar/project-jflint/Language_diarization/BUT/bnf_uvector_40.pth')
        else:
            Tru, pred = uvector.model_use(self.folpath,self.typ).calculate_quants('/u/home/a/asarkar/project-jflint/Language_diarization/BUT/mfcc_uvector_40.pth')
            print("LOADED",'/u/home/a/asarkar/project-jflint/Language_diarization/BUT/mfcc_uvector_40.pth')
        return Tru, pred
        
class create_xvector:
    def __init__(self, folpath, typ):
        self.folpath = folpath
        self.typ = typ
    
    def pred_tdnn(self):
        if self.typ == 'bnf':
            Tru, pred = uvector.model_use(self.folpath,self.typ).calculate_quants('/u/home/a/asarkar/project-jflint/Language_diarization/BUT/bnf_xvector_40.pth')
            print("LOADED",'/u/home/a/asarkar/project-jflint/Language_diarization/BUT/bnf_xvector_40.pth')
        else:
            Tru, pred = uvector.model_use(self.folpath,self.typ).calculate_quants('/u/home/a/asarkar/project-jflint/Language_diarization/BUT/mfcc_xvector_40.pth')
            print("LOADED",'/u/home/a/asarkar/project-jflint/Language_diarization/BUT/mfcc_xvector_40.pth')
        return Tru, pred 