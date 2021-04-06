import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pywt
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tensorflow as tf

torch.manual_seed(10)

data = pd.read_csv('index_data/near_20_years.csv') 

raw_data = data['AClose^GSPC']

# time delay
tau = 3
# Embedded dimension
m = 5

batch = 16
input_size = m
timestep_size = 5
hidden_size = 32
Hidden_layers = 3
Dropout = 0.6
output_size = 1
learning_rate = 0.001


index = []
data = []
for i in range(len(raw_data)):
     x, y = i, raw_data[i]
     X = float(x)
     Y = float(y)
     index.append(X)
     data.append(Y)

# Create wavelet object and define parameters
w = pywt.Wavelet('coif1')
maxlev = pywt.dwt_max_level(len(data), w.dec_len)
#maxlev = 2 # Override if desired
print("maximum level is " + str(maxlev))
threshold = 0.1 # Threshold for filtering

# Decompose into wavelet components, to the level selected:
coeffs = pywt.wavedec(data, 'coif1', level=maxlev)


plt.figure()
for i in range(1, len(coeffs)):
     #plt.subplot(maxlev, 1, i)
     #plt.plot(coeffs[i])
     coeffs[i] = pywt.threshold(coeffs[i], threshold*max(coeffs[i]))
     #plt.plot(coeffs[i])


datarec = pywt.waverec(coeffs, 'coif1')


mintime = 1
maxtime = len(raw_data)

ps_data = []
for i in range( len(datarec) - (m-1)*tau  ):
    delayTime = []
    for j in range( m ):
        delayTime.append( datarec[i + j * tau ] )
    ps_data.append(delayTime)

ps_data = pd.DataFrame(ps_data)

#mm = MinMaxScaler()

#ps_data_normalized = mm.fit_transform(ps_data )
ps_data_normalized = preprocessing.normalize(ps_data.T , norm='l2').T

train_data = ps_data_normalized[0 : int(len(ps_data_normalized)*0.8) ]
train_ans = df = pd.DataFrame(np.zeros((1, m)))
train_ans = pd.concat([train_ans, pd.DataFrame(train_data)]).to_numpy()
test_data = ps_data_normalized[ int(len(ps_data_normalized)*0.8) : len(ps_data_normalized) ]

train_data = torch.Tensor(train_data)

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, Hidden_layers, dropout ):
        super(RNN, self).__init__()
        self.rnn = nn.LSTM(
            input_size = input_size,
            hidden_size = hidden_size,
            num_layers = Hidden_layers,
            dropout = dropout 
        )
        self.out = nn.Sequential(

            nn.Linear(hidden_size, hidden_size),
            #nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            #nn.Tanh(),
            nn.Linear(hidden_size, 1),
            nn.ReLU()
        )

    def forward(self, _x):
        x, _ = self.rnn(_x)     # _x is input, size (seq_len, batch, input_size)
        s, b, h = x.shape       # x is output, size (seq_len, batch, hidden_size)
        x = x.view(s * b, h)
        x = self.out(x)
        x = x.view(s, b, -1)  # 把形状改回来
        return x 

def Loss(out, y):
    ans = 0
    for i in range(len(out)):
        ans += (out[i].detach().numpy()-y[i])**2
    return ans / len(out)

model = RNN(input_size, hidden_size, Hidden_layers, Dropout)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

for epoch in range(30): 
    mean_loss = 0
    for i in range( len(train_data) ):
        input_x = []
        try:
            for j in range(i, i + timestep_size):
                input_x.append(train_data[j])
        except:
            continue
        input_x = torch.tensor([item.cpu().detach().numpy() for item in input_x])
        input_x = tf.reshape(input_x, [timestep_size, -1, input_size ]).numpy()
        output = model( torch.from_numpy(tf.convert_to_tensor(input_x).numpy()) )
        # loss = Loss(output, train_ans[i])
        loss = criterion( torch.tensor(output, requires_grad = True).float(), torch.tensor(train_ans[i], requires_grad = True).float() )
        mean_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
    print( mean_loss/len(train_data) )
