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

class se_encoder(nn.module):
    def __init__(self,input_size, output_size):
        super(NN,self).__init__()
        
        # attention block 1
        self.afc1 = nn.Linear(input_size, input_size//8)
        self.afc2 = nn.Linear(input_size//8, input_size)
        
        # attention block 2
        self.afc3 = nn.Linear(input_size, input_size//8)
        self.afc4 = nn.Linear(input_size//8, input_size)
        
        # learnable weights
        self.vfc1 = nn.Linear(input_size, input_size)
        self.softmax = self.Softmax(dim=0)
        
    def forward(self, x):
        # extracting h
        x = F.relu(self.afc1(x))
        h = F.relu(self.afc2(x))
        
        # extracting attention
        a = self.softmax(self.vfc1(h))
        
        # extracting m
        m = torch.bmm(a, h) 
        
        m1 = F.relu(self.afc3(m))
        s = self.softmax(self.afc4(m1))
        
        # extracted xhat
        xhat = torch.mul(s,x)
        
        return(xhat)
    
class LSTMNet(torch.nn.Module):
    def __init__(self):
        super(LSTMNet, self).__init__()
        self.lstm1 = nn.LSTM(80, 256,bidirectional=True)
        self.lstm2 = nn.LSTM(2*256, 32,bidirectional=True)
               
        self.fc_ha=nn.Linear(2*32,100) 
        self.fc_1= nn.Linear(100,1)           
        self.sftmax = torch.nn.Softmax(dim=1)

    def forward(self, x):
        x1, _ = self.lstm1(x) 
        x2, _ = self.lstm2(x1)
        ht = x2[-1]
        ht=torch.unsqueeze(ht, 0)        
        ha= torch.tanh(self.fc_ha(ht))
        alp= self.fc_1(ha)
        al= self.sftmax(alp) 
        
       
        T=list(ht.shape)[1]  
        batch_size=list(ht.shape)[0]
        D=list(ht.shape)[2]
        c=torch.bmm(al.view(batch_size, 1, T),ht.view(batch_size,T,D))
        c = torch.squeeze(c,0)        
        return (c)

class MSA_DAT_Net(nn.Module):
    def __init__(self, model1,model2):
        super(MSA_DAT_Net, self).__init__()
        self.model1 = model1
        self.model2 = model2

        self.att1=nn.Linear(2*32,100) 
        self.att2= nn.Linear(100,1)           
        self.bsftmax = torch.nn.Softmax(dim=1)

        self.lang_classifier= nn.Sequential()
        self.lang_classifier.add_module('fc1',nn.Linear(2*32,8,bias=True))     
        
        
    def forward(self, x1,x2):
        u1 = self.model1(x1)
        u2 = self.model2(x2)        
        ht_u = torch.cat((u1,u2), dim=0)  
        ht_u = torch.unsqueeze(ht_u, 0) 
        ha_u = torch.tanh(self.att1(ht_u))
        alp = torch.tanh(self.att2(ha_u))
        al= self.bsftmax(alp)
        Tb = list(ht_u.shape)[1] 
        batch_size = list(ht_u.shape)[0]
        D = list(ht_u.shape)[2]
        u_vec = torch.bmm(al.view(batch_size, 1, Tb),ht_u.view(batch_size,Tb,D))
        u_vec = torch.squeeze(u_vec,0)
        
        lang_output = self.lang_classifier(u_vec)   
        
        return (lang_output, u1, u2)
       
    
def prepare_data(path):
    X_train = torch.load('./xlsr_Xtrain.pt')
    X_test = torch.load('./xlsr_Xtest.pt')
    y_train = torch.load('./xlsr_ytrain.pt')
    y_test = torch.load('./xlsr_ytest.pt')
    return(Xtrain, y_train)

if __name__ == "__main__":
    
    # se encoder
    modelencd = se_encoder(input_size, output_size)
    
    # BiLSTMS
    model1 = LSTMNet()
    model2 = LSTMNet()
    
    model1.cuda()
    model2.cuda()
    
    # WSSL
    model = MSA_DAT_Net(model1,model2)
    model.cuda()
    
    optimizer = optim.SGD(model.parameters(),lr = 0.01, momentum= 0.9)

    loss_lang = torch.nn.CrossEntropyLoss(reduction='mean') # Primary loss
    loss_wssl = torch.nn.CosineSimilarity()
    
    loss_lang.cuda()
    
    n_epoch = 30
    
    xtrain, ytrain = prepare_data("add path")
    
    X = []
    i = 0
    for e in range(n_epoch):
        cost = 0. 
        
        # pass through SE
        for data, label in zip(X_train, y_train):
            data = data.to(device)
            label = label.to(device)

            data = data.reshape(data.shape[0],-1)

            X.append(modelencd(data).detach().numpy())
            
        # form two types of vectors
        for i in range(0,len(X)-look_back1,1):           
            a=X[i:(i+look_back1),:]        
            Xdata1.append(a)
        Xdata1=np.array(Xdata1)

        for i in range(0,len(X)-look_back2,2):        
            b=X[i:(i+look_back2):3,:]        
            Xdata2.append(b)
        Xdata2=np.array(Xdata2)

        # formed two datsets
        Xdata1 = torch.from_numpy(Xdata1).float()
        Xdata2 = torch.from_numpy(Xdata2).float()
        
        for i in range(min(len(X_data2),len(X_data1))):
            fl,e1,e2 = model.forward(torch.tensor(Xdata1[i]),torch.tensor(Xdata2[i]))      
            err_l = loss_lang(fl,Y1)
            err_wssl = abs(loss_wssl(e1,e2))               
            T_err = err_l + 0.25*err_wssl   # alpha = 0.25 in this case. Change it to get better output       

            T_err.backward()
            optimizer.step()
            cost = cost + T_err.item()

            print("ZWSSL5:  epoch "+str(e+1)+" completed files  "+str(i)+"/"+str(l)+" Loss= %.3f"%(cost/i)) 