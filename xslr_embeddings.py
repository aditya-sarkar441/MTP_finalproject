import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import glob
import os
from towhee import pipeline

def prepare_label(f):
    f1 = os.path.splitext(f)[0]     

    lang = f1.split("/")[-1][0:3]
                
    if(lang == 'asm'):
        Y1 = [1,0,0,0,0,0,0,0]
        
    elif(lang == 'ben'):
        Y1 = [0,1,0,0,0,0,0,0]            
    
    elif(lang == 'guj'):
        Y1 = [0,0,1,0,0,0,0,0]
        
    elif(lang == 'hin'):
        Y1 = [0,0,0,1,0,0,0,0]
        
    elif(lang == 'kan'):
        Y1 = [0,0,0,0,1,0,0,0]
        
    elif(lang == 'mal'):
        Y1 = [0,0,0,0,0,1,0,0]
    
    elif(lang == 'odi'):
        Y1 = [0,0,0,0,0,0,1,0]
        
    elif(lang == 'tel'):
        Y1 = [0,0,0,0,0,0,0,1]    
    else:
        Y1 = [1,0,0,0,0,0,0,0]
    
    return(Y1)
    
    
def prepare_data(folder_name):
    files_list = []
    path = folder_name+'/*'
    X=[]
    Y=[]
    
    # XLSR embeddings
    embedding_pipeline = pipeline('towhee/audio-embedding-wav2vec2-xlsr53')
    
    # generate embeddings for all wavs
    for f in glob.glob(path):
        for fn in glob.glob(f+'/*.wav'):
            x = embedding_pipeline(str(f))
            X.append(x)
            Y.append(prepare_label(f))
            print(x.shape)
            
    X = np.asarray(X)
    Y = np.asarray(Y)
    
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    X_train = torch.from_numpy(X_train)
    X_test = torch.from_numpy(X_test)
    
    y_train = torch.from_numpy(y_train)
    y_test = torch.from_numpy(y_test)
       
    return(X_train, X_test, y_train, y_test)
            

a,b,c,d = prepare_data('')
torch.save(a, './xlsr_Xtrain.pt')
torch.save(b, './xslr_Xtest.pt')
torch.save(c, './xslr_ytrain.pt')
torch.save(d, './xslr_ytest.pt')