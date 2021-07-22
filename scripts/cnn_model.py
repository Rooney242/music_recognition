import numpy as np
import pandas as pd
import json
import torch
import sys
from torch import nn
from torch import optim
import matplotlib.pyplot as plt
import time
import random
import pickle

from sklearn.experimental import enable_iterative_imputer
from sklearn import preprocessing, impute, model_selection, metrics, neighbors, ensemble, feature_selection
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.svm import SVC
from sklearn.decomposition import PCA, KernelPCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn import mixture

import essentia 
import essentia.standard as es

ann_path = '../emomusic/annotations/'
mod_path = '../models/'
graph_path = '../graphs/'
clips_path = '../emomusic/clips/'
ext = '.mp3'
random_state = 222
sample_rate = 500
v_dim = sample_rate*44

df = pd.read_csv(ann_path+'songs_info.csv')
df = df[['song_id', 'Mediaeval 2013 set']]

train_idx = list(df[df['Mediaeval 2013 set'] == 'development']['song_id'])
valid_idx = random.sample(train_idx, int(np.floor(len(train_idx) * 0.2))) 
train_idx = [i for i in train_idx if i not in valid_idx]
test_idx = list(df[df['Mediaeval 2013 set'] == 'evaluation']['song_id'])




#######
'''all_idx = train_idx+valid_idx+test_idx
cnn_feats = pd.DataFrame(index=all_idx, columns=['audio', 'arousal_mean', 'valence_mean'])

stat_labs = pd.read_parquet(ann_path+'static_selected_features.pqt')[['arousal_mean', 'valence_mean']]
for clip_id in all_idx:
    clip_path = clips_path+str(clip_id)+ext

    loader = essentia.standard.MonoLoader(filename=clip_path, sampleRate=sample_rate)
    audio = loader()

    cnn_feats.loc[clip_id]['audio'] = audio[:v_dim]
    if cnn_feats.loc[clip_id]['audio'].shape[0] != v_dim: print('error')

    print(clip_id, cnn_feats.loc[clip_id]['audio'].shape)
    cnn_feats.loc[clip_id]['arousal_mean'] = stat_labs.loc[clip_id]['arousal_mean']
    cnn_feats.loc[clip_id]['valence_mean'] = stat_labs.loc[clip_id]['valence_mean']

cnn_feats.to_parquet(ann_path+'cnn_selected_features_long.pqt')
sys.exit()'''

data = pd.read_parquet(ann_path+'cnn_selected_features.pqt')

def create_loader(idx_list, batch_size):
    last = data.loc[train_idx].shape[0] - 1
    loader = []
    for i, r in enumerate(data.loc[train_idx].iterrows()):
        norm_idx = i % batch_size
        if not norm_idx:
            batch_feats = torch.empty(batch_size, 1, v_dim)
            batch_labs  = torch.empty(batch_size, 2)

        batch_feats[norm_idx, 0] = torch.from_numpy(r[1]['audio']).reshape((1,1,v_dim))
        batch_labs[norm_idx] = torch.Tensor([r[1]['arousal_mean'], r[1]['valence_mean']])

        if norm_idx == batch_size-1 or i == last:
            loader.append((batch_feats, batch_labs))
    return loader

batch_size = 16

trainloader = create_loader(train_idx, batch_size)
validloader = create_loader(valid_idx, batch_size)
testloader = create_loader(test_idx, 1)

########
epochs = 100
lr = 0.0005
pdropout = 0.5


channels = [1, 5, 12]
kernel_size = 30
stride = 15
padding = 0
max_pool = 2
neurons = [64, 32]
########

class audio_cnn(nn.Module):
    def __init__(self, dimx, nlabels, pdropout=0.4):
        super().__init__()

        self.conv1 = nn.Conv1d(in_channels=channels[0], out_channels=channels[1], 
                               kernel_size=kernel_size, stride=stride)
        
        self.conv2 = nn.Conv1d(in_channels=channels[1], out_channels=channels[2], 
                               kernel_size=kernel_size, stride=stride)
        
        # Max pool layer
        self.pool = nn.MaxPool1d(max_pool)

        # Spatial dimension of the Tensor at the output of the 2nd CNN
        self.final_dim = channels[2]*int((int((dimx - (kernel_size - 1))/stride/max_pool) - (kernel_size - 1))/stride/max_pool)

        # Linear layers
        self.linear1 = nn.Linear(self.final_dim, neurons[0]) 
        self.linear2 = nn.Linear(neurons[0], neurons[1]) 
        self.linear3 = nn.Linear(neurons[1], nlabels) 
    
        self.relu = nn.ReLU()
        
        self.logsoftmax = nn.LogSoftmax(dim=1) 
        
        self.dropout = nn.Dropout(p=pdropout)

        print(dimx,  self.final_dim)
    
    def forward(self, x):
        # Pass the input tensor through the CNN operations
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)

        # Flatten the tensor into a vector of appropiate dimension using self.final_dim
        x = x.view(-1, self.final_dim) 

        # Pass the tensor through the Dense Layers
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear3(x)

        return x


class audio_cnn_extended(audio_cnn):
    def __init__(self, dimx, nlabels, epochs=100,lr=0.001, pdropout=0.2):
        
        super().__init__(dimx, nlabels, pdropout)
        
        self.lr = lr #Learning Rate
        self.epochs = epochs
        
        self.optim = optim.Adam(self.parameters(), self.lr)
        self.criterion = nn.MSELoss()       
        
        # A list to store the loss evolution along training
        self.loss_during_training = [] 
        self.valid_loss_during_training = []
        
    def trainloop(self,trainloader,validloader):
        # Optimization Loop
        for e in range(int(self.epochs)):
            start_time = time.time()
            
            # Random data permutation at each epoch
            running_loss = 0.

            for images, labels in trainloader:
        
                #Reset Gradients!
                self.optim.zero_grad()  #YOUR CODE HERE
                out = self.forward(images)

                #Your code here
                loss = self.criterion(out,labels)
                running_loss += loss.item()

                #Compute gradients
                loss.backward()
                
                #SGD stem
                self.optim.step()           
            self.loss_during_training.append(running_loss/len(trainloader))
            
            # Validation Loss
            # Turn off gradients for validation, saves memory and computations
            with torch.no_grad():            
                running_loss = 0.
                self.eval() # evaluation mode so dropout prob = 0

                for images,labels in validloader:
                    
                    # Compute output for input minibatch
                    out = self.forward(images) #YOUR CODE HERE
            
                    #Your code here
                    loss = self.criterion(out, labels) #YOUR CODE HERE
                    running_loss += loss.item()   
                self.valid_loss_during_training.append(running_loss/len(validloader))    
                self.train() # back to train mode


            if(e % 1 == 0):
                print("Epoch %d. Training loss: %f, Validation loss: %f, Time per epoch: %f seconds" 
                      %(e,self.loss_during_training[-1],self.valid_loss_during_training[-1],
                       (time.time() - start_time)))
                
    def eval_performance(self, dataloader):
        loss = 0
        accuracy = 0

        # Turn off gradients for validation, saves memory and computations
        with torch.no_grad():
            ar_pred = []
            ar_real = []
            va_pred = []
            va_real = []
            loss = 0

            for audio,labels in dataloader:
                preds = self.forward(audio)
                ar_pred.append(preds.numpy()[0][0])
                ar_real.append(labels.numpy()[0][0])
                va_pred.append(preds.numpy()[0][1])
                va_real.append(labels.numpy()[0][1])

                loss += self.criterion(preds, labels).item()
    
            return (metrics.r2_score(ar_pred, ar_real), metrics.r2_score(va_pred, va_real), loss)

my_cnn= audio_cnn_extended(dimx=v_dim, nlabels=2, epochs=epochs, lr=lr, pdropout=pdropout)
my_cnn.trainloop(trainloader, validloader)
pickle.dump(my_cnn, open(mod_path+'cnn.pkl', 'wb'))


with open('models', 'r') as f:
    models = json.load(f)

r2_arousal, r2_valence, loss = my_cnn.eval_performance(testloader)
models['arousal_mean']['cnn']['r2'] = r2_arousal
models['valence_mean']['cnn']['r2'] = r2_valence

with open('models', 'w') as f:
    json.dump(models, f, indent = 4)
