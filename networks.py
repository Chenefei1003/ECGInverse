import torch
import torch.nn as nn

''' ELM '''
class ELM(nn.Module):
    def __init__(self):
        super(ELM,self).__init__()
        self.fc1 = nn.Linear(64, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 257)
    
    def forward(self, x):
        out = self.fc2(self.relu(self.fc1(x)))
        return out

''' KPCA-CNN '''
class KPCACNN(nn.Module):
    def __init__(self):
        super(KPCACNN,self).__init__()
        self.conv1 = nn.Conv1d(1,32, kernel_size=5,padding = 1)
        self.conv2 = nn.Conv1d(32,64, kernel_size=5,padding = 1)
        self.conv3 = nn.Conv1d(64,128, kernel_size=3,padding = 1)
        
        self.mp1 = nn.MaxPool1d(kernel_size=2,stride = 2)
        self.mp2 = nn.MaxPool1d(kernel_size=2,stride = 2)
        self.mp3 = nn.MaxPool1d(kernel_size=2,stride = 2)
        self.bn1 = nn.BatchNorm1d(32)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(128)
        self.d1=nn.Dropout(0.1)
        self.d2=nn.Dropout(0.1)
        self.d3=nn.Dropout(0.1)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.relu3 = nn.ReLU()
        self.fc1 = nn.Linear(896,257)
        
    def forward(self,x):
        in_size = x.size(0)
        out = self.relu1(self.bn1(self.mp1(self.conv1(x))))
        out = self.relu2(self.bn2(self.mp2(self.conv2(out))))
        out = self.relu3(self.bn3(self.mp3(self.conv3(out))))
        out = out.view(in_size, -1)
        out = self.fc1(out)
        return out



''' LSTM '''
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,batch_first=True, bidirectional=True)
        self.fc = nn.Linear(256, num_classes)  # 隐层包含向前层和向后层两层，所以隐层共有两倍的Hidden_size
    
    def forward(self, x):
        # 前向传播 LSTM
        out, _ = self.lstm(x)  # LSTM输出大小为 (batch_size, seq_length, hidden_size*2)
        s,b,h = out.size()
        out = out.view(s*b,h)
        
        out = self.fc(out)
        return out
    
''' BiGRU '''
class BiGRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(BiGRU, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        
        self.fc = nn.Linear(256, num_classes)  # 隐层包含向前层和向后层两层，所以隐层共有两倍的Hidden_size
    
    def forward(self, x):
        # 前向传播 LSTM
        out, _ = self.lstm(x)  # LSTM输出大小为 (batch_size, seq_length, hidden_size*2)
        s,b,h = out.size()
        out = out.view(s*b,h)
        
        out = self.fc(out)
        return out
    
''' PCA-BiGRU '''
class PCA_BiGRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(PCA_BiGRU, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        
        self.fc = nn.Linear(256, num_classes)  # 隐层包含向前层和向后层两层，所以隐层共有两倍的Hidden_size
        
        #self.pca = PCA(n_components=input_size)
    
    def forward(self, x):
        # 前向传播 LSTM
        out, _ = self.lstm(x)  # LSTM输出大小为 (batch_size, seq_length, hidden_size*2)
        s,b,h = out.size()
        out = out.view(s*b,h)
        
        out = self.fc(out)
        return out
    
''' CNN-BiGRU '''
class CNN_BiGRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(CNN_BiGRU, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        
        self.fc = nn.Linear(416, num_classes)  # 隐层包含向前层和向后层两层，所以隐层共有两倍的Hidden_size
        
        self.conv1 = nn.Conv1d(1, 10, kernel_size=9,stride=2,padding=4)
        self.mp = nn.MaxPool1d(kernel_size=2,stride = 2)
        self.bn1 = nn.BatchNorm1d(10)
        self.dropout=nn.Dropout(0.3)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(input_size,128)
    
    def forward(self, x):
        
        in_size = x.size(1)

        x1 = x.reshape(x.shape[1],1,64).to(device)
        x1 = self.relu(self.bn1(self.mp(self.conv1(x1))))
        x1 = x1.view(in_size, -1)
        x1 = x1.reshape(x1.shape[0],x1.shape[1])
        
        # 初始话LSTM的隐层和细胞状态
        h0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).to(device) # 同样考虑向前层和向后层
    
        # 前向传播 LSTM
        out, _ = self.lstm(x, h0)  # LSTM输出大小为 (batch_size, seq_length, hidden_size*2)
        
        out1, out2 = torch.chunk(out, 2, dim=2)
        out_cat = torch.cat((out1[-1, :, :], out2[0, :, :]), 1)
        out_cat = torch.cat((x1,out_cat),1)

        out = self.fc(out_cat)
        return out
    
''' PCA-CNN-BiGRU '''
class PCA_CNN_BiGRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(PCA_CNN_BiGRU, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.GRU(input_size, hidden_size, num_layers,batch_first=True, bidirectional=True)
        
        self.fc = nn.Linear(316, num_classes)  # 隐层包含向前层和向后层两层，所以隐层共有两倍的Hidden_size
        
        self.conv1 = nn.Conv1d(1, 10, kernel_size=9,stride=2,padding=4)
        self.mp = nn.MaxPool1d(kernel_size=2,stride = 2)
        self.bn1 = nn.BatchNorm1d(10)
        self.dropout=nn.Dropout(0.3)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(input_size,128)
        
        #self.pca = PCA(n_components=input_size)
    
    def forward(self, x):
        
        in_size = x.size(1)
        
        #x = torch.tensor(pca.fit_transform(x.squeeze(0))[np.newaxis, :])
        
        x1 = x.reshape(x.shape[1],1,x.shape[2]).to(device)
        x1 = self.relu(self.bn1(self.mp(self.conv1(x1))))
        x1 = x1.view(in_size, -1)
        x1 = x1.reshape(x1.shape[0],x1.shape[1])
        
        # 初始话LSTM的隐层和细胞状态
        h0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).to(device) # 同样考虑向前层和向后层
    
        # 前向传播 LSTM
        out, _ = self.lstm(x, h0)  # LSTM输出大小为 (batch_size, seq_length, hidden_size*2)
        
        out1, out2 = torch.chunk(out, 2, dim=2)
        out_cat = torch.cat((out1[-1, :, :], out2[0, :, :]), 1)
        out_cat = torch.cat((x1,out_cat),1)
        out = self.fc(out_cat)
        return out