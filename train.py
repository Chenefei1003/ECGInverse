import math
import torch
import torch.nn as nn
import torch.nn.functional as F

import os
import torch 
import torch.nn as nn
import torchvision
import scipy.io as sio
from sklearn.model_selection import KFold
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from networks import ELM, KPCACNN, LSTM, BiGRU, PCA_BiGRU, CNN_BiGRU, PCA_CNN_BiGRU
from sklearn.decomposition import PCA, KernelPCA
from sklearn.decomposition import PCA

os.environ["CUDA_VISIBLE_DEVICES"] = "3"
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
seed = 55
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

# hyperparameter 
input_size = 24 # 64
num_classes = 257
hidden_size = 128
num_layers = 2
learning_rate=0.001

period = 500

def standardize(data):
    mu = np.mean(data,axis=0)
    sigma = np.std(data,axis=0)
    return (data - mu) / sigma

class ECGDataset(Dataset):
    def __init__(self,X = None,y = None,transform = None):
        self.X = X
        self.y = y
        self.transform = transform
    def __len__(self):
        return self.X.shape[0]
    def __getitem__(self,idx):
        X = self.X[idx]
        y = self.y[idx]
        return X,y 
    
def data_loader():
    tsfm = transforms.Compose([transforms.ToTensor()])
    train = ECGDataset(X = X_train, y = y_train, transform = tsfm)
    test = ECGDataset(X = X_test, y = y_test, transform = tsfm)
    train_loader = DataLoader(train,
                              batch_size = 15500,
                              shuffle = False,
                              num_workers = 0)
    test_loader = DataLoader(test,
                             batch_size = 3500,
                             shuffle = False,
                             num_workers = 0)
    return train_loader,test_loader

# Train & Test
def train(epoch):
    model.train()
    train_loss = 0
    train_len = 0
    for i, (X, y) in enumerate(train_loader):
        #X = X.reshape(1, X.shape[0],X.shape[1]).to(device)
        X = X.reshape(X.shape[0],1,X.shape[1]).to(device) # input for KPCA-CNN
        #X = X.reshape(X.shape[0],64).to(device) # input for ELM
        y = y.to(device)
        # 前向传播
        outputs = model(X)
        #print(outputs.shape)
        loss = criterion(outputs, y)

        # 反向传播和优化，注意梯度每次清零
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss = loss.item()
    
#     writer.add_scalar('Train/Loss', train_loss, epoch+1)
    if (epoch + 1) % 100 == 0:
        print ('Training Epoch [{}/{}] --------  Loss: {:.4f}'.format(epoch+1, num_epochs, train_loss))
    
def evaluate(epoch):
    model.eval()
    test_loss = 0
    test_len = 0
    global best_mre
    global best_mre_cc
    global best_cc
    global best_cc_mre
    global best_re_epoch
    global best_cc_epoch
    with torch.no_grad():
        for i, (X, y) in enumerate(test_loader):
            #X = X.reshape(1, X.shape[0],X.shape[1]).to(device)
            X = X.reshape(X.shape[0],1,X.shape[1]).to(device) # input for KPCA-CNN
            #X = X.reshape(X.shape[0],64).to(device) # input for ELM
            y = y.to(device)
            outputs = model(X)
            pred = outputs.cpu().numpy()
            label = y.cpu().numpy()
            loss = criterion(outputs, y)
            mre = np.mean(((np.sum((pred-label)**2,axis=1))**0.5)/((np.sum((label)**2,axis=1))**0.5))
            cc = 0
            for i in range(label.shape[0]):
                r = np.corrcoef(pred[i],label[i])
                cc = cc+r[0][1]
            cc = cc/label.shape[0]
            if(best_mre>mre):
                best_mre = mre
                best_mre_cc = cc
                best_re_epoch = epoch+1
                torch.save(model.state_dict(), 'model/PCA-CNN-BiGRU-re-01.pth')
            if(best_cc<cc):
                best_cc = cc
                best_cc_mre = mre
                best_cc_epoch = epoch+1
                torch.save(model.state_dict(), 'model/PCA-CNN-BiGRU-cc-01.pth')
            test_loss = loss.item()
            
#     writer.add_scalar('Test/Loss', test_loss, epoch+1)
#     writer.add_scalar('Test/RE', mre, epoch+1)
#     writer.add_scalar('Test/CC', cc, epoch+1)
    if (epoch + 1) % 100 == 0:
        print ('Testing  Epoch [{}/{}] -------- MRE: {:.4f} -------- CC: {:.4f}\n'.format(epoch+1, num_epochs, mre, cc))

# Training
kpca = KernelPCA(kernel="rbf", n_components=input_size)
pca = PCA(n_components=input_size)

matfnamex = 'data/70a'
matfnamey = 'data/70b'
mfc = sio.loadmat(matfnamex)['a'].T
art = sio.loadmat(matfnamey)['b'].T
totalsamples = len(mfc)
print(totalsamples)

X = standardize(mfc).astype("float32")[totalsamples % period:]
y = standardize(art).astype("float32")[totalsamples % period:]

X = pca.fit_transform(X)

print(X.shape, y.shape)

X_data = []
y_data = []

for i in range(totalsamples // period):
    X_data.append(X[i * period:(i+1) * period, :])
    y_data.append(y[i * period:(i+1) * period, :])

kf = KFold(5, shuffle=False)

tr_index = dict()
val_index = dict()

for i, (tr,val) in enumerate(kf.split(X_data)):
    tr_index[str(i)] = tr
    val_index[str(i)] = val

X_train = []
y_train = []
X_test = []
y_test = []
for index in tr_index['0']:  # fold: 0-4
    X_train.append(X_data[index])
    y_train.append(y_data[index])
for index in val_index['0']:  # fold: 0-4
    X_test.append(X_data[index])
    y_test.append(y_data[index])
X_train = np.concatenate(X_train, axis=0)
y_train = np.concatenate(y_train, axis=0)
X_test = np.concatenate(X_test, axis=0)
y_test = np.concatenate(y_test, axis=0)

print("--------------------")
print(X_train.shape,y_train.shape,X_test.shape,y_test.shape)

train_loader,test_loader = data_loader()

#Model

model = PCA_CNN_BiGRU(input_size, hidden_size, num_layers, num_classes).to(device)

criterion = torch.nn.MSELoss(reduction='mean')
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

#Training

num_epochs = 2000

best_mre = 100
best_mre_cc = 0
best_cc = -100
best_cc_mre = 0
best_re_epoch = 1
best_cc_epoch = 1

for epoch in range(num_epochs):
    train(epoch)
    evaluate(epoch)
print('best_mre: {:.6f}'.format(best_mre))
print('best_mre_cc: {:.6f}'.format(best_mre_cc))
print('best_re_epoch: {:.1f}'.format(best_re_epoch))
print("=========================")
print('best_cc: {:.6f}'.format(best_cc))
print('best_cc_mre: {:.6f}'.format(best_cc_mre))
print('best_cc_epoch: {:.1f}'.format(best_cc_epoch))