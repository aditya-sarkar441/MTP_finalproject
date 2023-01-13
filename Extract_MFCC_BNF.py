#Libraries
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import IPython.display as ipd
import os
#from BottleneckFeatureExtractor import bottleneck2posterior
from BUT import audio2bottleneck

#Class Audio

## Look at the github repo shared.
class Audio:
    def __init__(self,audio_path):
        self.audio_path = audio_path
        self.x,self.sr = librosa.load(audio_path,sr=None)

    def play_audio(self):
      ipd.display(ipd.Audio(self.audio_path))
    
    def display(self):
      plt.figure(figsize=(12, 5))
      plt.grid(True,alpha=1/2)
      librosa.display.waveplot(self.x, sr=self.sr)  
      plt.title(self.audio_path)
      plt.show()
    
    def spectogram(self):
      X = librosa.stft(self.x)
      Xdb = librosa.amplitude_to_db(abs(X))
      plt.figure(figsize=(12, 5))
      librosa.display.specshow(Xdb, sr=self.sr, x_axis='time', y_axis='hz') 
      plt.colorbar()
      plt.show()

    def mfcc(self):
        mfccs = librosa.feature.mfcc(y=self.x, sr=self.sr, n_mfcc=39)
        return mfccs.reshape(-1,39)

    def save_mfcc(self):
      mfccs = self.mfcc()
      f_name = self.audio_path.split('/')[-1].split('.')[0]
      #print(f_name)
      path = 'Segment_2/MFCC/'+str(self.audio_path.split('/')[1])+'/'+self.audio_path.split('/')[2]+'/'
      #print(path,f_name)
      outputfile = path.split('.')[0]+'.csv'#os.path.join(path, f_name+".csv")  
      print(outputfile)
      file = open(outputfile, 'w+')
      np.savetxt(file, mfccs, delimiter=",") #save MFCCs as .csv
      file.close()


def Save_MFCCs(directory):
    for path in os.listdir(directory):
        audio_path = os.path.join(directory, path)
        if os.path.isfile(audio_path):
            Audio(audio_path).save_mfcc()

class create_bnf(Audio):
    def __init__(self, audio_path, nn):
        super().__init__(audio_path)
        self.nn = nn
        f_name = str(self.audio_path.split('/')[-1].split('.')[0])
        f_path = "BNF_YouTube_Data/"+str(self.audio_path.split('/')[1])+'/'+str(self.audio_path.split('/')[2])
        f_path = f_path.split('.')[0]
        self.bn_file=f_path+".csv"
        self.vad_file=""
        self.post_file=f_path+'/'+f_name+".h5"
        
    def bnf(self):
        BNF = audio2bottleneck.compute(self.nn,self.audio_path,self.bn_file,self.vad_file)
        outputfile = os.path.join(self.bn_file)  
        #print(outputfile)
        file = open(outputfile, 'w+')
        np.savetxt(file, BNF, delimiter=",") #save MFCCs as .csv
        file.close()
        #bottleneck2posterior.compute(self.nn, self.bn_file, self.post_file)
        #print("files created")
        #print(self.audio_path)
        #print("BN file", self.bn_file)
        #print("post file",self.post_file)


def Save_BNFs(directory):
    nn = "BabelMulti"
    for path in os.listdir(directory):
        audio_path = os.path.join(directory, path)
        if os.path.isfile(audio_path):
            print(audio_path)
            create_bnf(audio_path, nn).bnf()


lang = ['asm','ben','guj','hin','kan','mal','odi','tel']
for l in lang:
    print(l)
    directory = "IITMandi_YouTube/"+l+"/" #Change Language Directories
    #print(directory)
    #Save_MFCCs(directory)      #Uncomment when needed
    Save_BNFs(directory)


directory = "VAD/" #Change Language Directories
#Save_MFCCs(directory)      #Uncomment when needed
#Save_BNFs(directory)
'''
for path in os.listdir(directory):
    audio_path = os.path.join(directory, path)
    if os.path.isfile(audio_path):
        print(audio_path,end=' ')
        x,sr = librosa.load(audio_path,sr=None)
        print(len(x)/sr,sr)
'''