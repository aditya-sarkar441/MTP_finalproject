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

def make_data(f):
    df = pd.read_csv(f)
    dt = df.astype(np.float32)
    dt = dt.T
    X=np.array(dt)
    
    Xdata1 = []
    Xdata2 = []
    Ydata1 = []
      
    mu = X.mean(axis=0)
    std = X.std(axis=0)
    np.place(std, std == 0, 1) 
    X = (X - mu) / std 
    f1 = os.path.splitext(f)[0]     

    lang = f1.split("/")[-1][0:3]
    
#     print(lang)
                
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
    
    Y1 = np.array(Y1)
    
    Xdata1 = np.array(X)    
    return(Xdata1, Y1)

def prepare_data(path):
    files_list = []
    path = path+'/*'
    X=[]
    Y=[]
    for f in glob.glob(path):
        i = 0    
        for fn in glob.glob(f+'/*.csv'):
            x,y = make_data(fn)
            X.append(x)
            Y.append(y)
        
    X = np.asarray(X)
    Y = np.asarray(Y)
    
    print(X.shape, Y.shape)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42)
    X_train = torch.from_numpy(X_train)
    X_test = torch.from_numpy(X_test)
    
    y_train = torch.from_numpy(y_train)
    y_test = torch.from_numpy(y_test)
       
    return(X_train, X_test, y_train, y_test)

class NN(nn.Module):
    def __init__(self,input_size, output_size):
        super(NN,self).__init__()
        self.fc1 = nn.Linear(input_size, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, output_size)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return(x)

def check_accuracy(loader, model):
    num_correct = 0
    num_samples = 0
    model.eval()
    
    with torch.no_grad():
        for x,y in loader:
            x = x.to(device)
            y = y.to(device)
            
            x = x.reshape(x.shape[0],-1)
            
            scores = model(x)
            _, predictions = scores.max(1)
            num_correct += (predictions==y).sum()
            num_samples += predictions.size(0)
         
        acc = float(num_correct)/float(num_samples)
        print("accuracy", acc)

# Code starts here
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

input_size = 511
output_size = 8
learning_rate = 0.001
num_epochs = 40
path = '/u/home/a/asarkar/project-jflint/Language_identification/bnf_xvector'

# Load data
# X_train, X_test, y_train, y_test = prepare_data(path)
# print(X_train.shape,y_train.shape)

# torch.save(X_train, './tensor_Xtrain.pt')
# torch.save(X_test, './tensor_Xtest.pt')
# torch.save(y_train, './tensor_ytrain.pt')
# torch.save(y_test, './tensor_ytest.pt')

X_train = torch.load('./tensor_Xtrain.pt')
X_test = torch.load('./tensor_Xtest.pt')
y_train = torch.load('./tensor_ytrain.pt')
y_test = torch.load('./tensor_ytest.pt')

model = NN(input_size, output_size).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    print("epoch = ",epoch)
    for data, label in zip(X_train, y_train):
        data = data.to(device)
        label = label.to(device)
        
        data = data.reshape(data.shape[0],-1)
        
        scores = model(data).reshape(8)
        
        scores = scores.to(torch.float64)
        label = label.to(torch.float64)
        
        loss = criterion(scores, label)
        
        optimizer.zero_grad()
        loss.backward()
        
        optimizer.step()
    print("loss = ",loss)

check_accuracy()