# -*- coding: utf-8 -*-
"""
Created on Mon Nov 26 10:25:02 2018

@author: h.oberoi
"""

import torch
import torch.nn as pytorch
import torch.nn.functional as F
from torch.utils.data.dataset import Dataset
from sklearn.metrics import precision_recall_fscore_support
import pandas as pd
import pdb
import numpy as np


#Initialize Global variables
  
q_max_words=12
p_max_words=50
emb_dim=50


## The following LoadValidationSet method reads ctf format validation file and creates query, passage feature vectors and also copies labels for each pair.
## the created vectors will be useful to find metrics on validation set after training each epoch which will be useful to decide the best model 
'''def LoadValidationSet(validationfile):
	f = open(validationfile,'r',encoding="utf-8")
	for line in f:
		tokens = line.strip().split("|")  
		#tokens[0] will be empty token since the line is starting with |
		x1 = tokens[1].replace("qfeatures","").strip() #Query Features
		x2 = tokens[2].replace("pfeatures","").strip() # Passage Features
		y = tokens[3].replace("labels","").strip() # labels
		x1 = [float(v) for v in x1.split()]
		x2 = [float(v) for v in x2.split()]
		y = [int(w) for w in y.split()]        
		y = y[1] # label will be at index 1, i.e. if y = "1 0" then label=0 else if y="0 1" then label=1

		validation_query_vectors.append(x1)
		validation_passage_vectors.append(x2)
		validation_labels.append(y)

		#print("1")
	
	print("Validation Vectors are created")
'''	
	
class Dataset_load(Dataset):
    def __init__(self, path, chunksize, nb_samples):
        self.path = path
        self.chunksize = chunksize
        self.len = int(nb_samples / self.chunksize)
    def __getitem__(self, index):
        x = next(
			pd.read_csv(
				self.path,
				skiprows=index * self.chunksize, 
				chunksize=self.chunksize,
				names=['data']))
        x = (x.data.values).tolist()
        query_vectors = []
        passage_vectors = []
        labels = []
        for line in x:
            tokens = line.strip().split("|")  
			#tokens[0] will be empty token since the line is starting with |
            x1 = tokens[1].replace("qfeatures","").strip() #Query Features
            x2 = tokens[2].replace("pfeatures","").strip() # Passage Features
            y = tokens[3].replace("labels","").strip() # labels
            x1 = [float(v) for v in x1.split()]
            x2 = [float(v) for v in x2.split()]
            y = [int(w) for w in y.split()]        
            query_vectors.append(x1)
            passage_vectors.append(x2)
            if y[1]==1:
                y=1
            else:
                y=0
            labels.append(y)
        
        return query_vectors, passage_vectors, labels
    def __len__(self):
        return self.len

class cnn_network(pytorch.Module):
	def __init__(self):
		super(cnn_network, self).__init__()
		self.convA1 = pytorch.Conv2d(in_channels = 1, out_channels = 4, kernel_size = (3,10))                     #in: 1X12X50   out: 4X10X41
		self.poolA1 = pytorch.MaxPool2d(kernel_size = (2,3), stride = (2,3))                                      #in: 4X10X41   out: 4X5X13
		self.convA2 = pytorch.Conv2d(in_channels = 4, out_channels = 2, kernel_size = (2,4))                      #in: 4X5X13   out: 2X4X10
		self.poolA2 = pytorch.MaxPool2d(kernel_size = (2,2), stride = (2,2))                                      #in: 2X4X10   out: 2X2X5
		self.denseA = pytorch.Linear(20,4)                                                                        #in: 2*2*5    out: 4
		
		self.convB1 = pytorch.Conv2d(in_channels = 1, out_channels = 4, kernel_size = (5,10))                     #in: 1X50X50   out: 4X46X41
		self.poolB1 = pytorch.MaxPool2d(kernel_size = (5,5), stride = (5,5))                                      #in: 4X46X41   out: 4X9X8
		self.convB2 = pytorch.Conv2d(in_channels = 4, out_channels = 2, kernel_size = (3,3))                      #in: 4X9X8   out: 2X7X6
		self.poolB2 = pytorch.MaxPool2d(kernel_size = (2,2), stride = (2,2))                                      #in: 2X7X6   out: 2X3X3
		self.denseB = pytorch.Linear(18,4)                                                                        #in: 2*3*3   out: 4
		
		self.model = pytorch.Linear(4,2)                                                                          #in: 4      out: 2
	
	def forward(self,query_features,passage_features):
		query_features = self.poolA1(F.relu(self.convA1(query_features)))
		query_features = self.poolA2(F.relu(self.convA2(query_features)))
		query_features = query_features.view(-1,20)
		query_features = self.denseA(query_features)
		
		passage_features = self.poolB1(F.relu(self.convB1(passage_features)))
		passage_features = self.poolB2(F.relu(self.convB2(passage_features)))
		passage_features = passage_features.view(-1,18)
		passage_features = F.relu(self.denseB(passage_features))
		
		merged = query_features * passage_features
		
		output = self.model(merged)
		
		return output


def TrainAndValidate(trainfile):

	#*****Hyper-Parameters******
    q_max_words= 12
    p_max_words = 50
    emb_dim = 50
    num_classes = 2
    minibatch_size = 250
    total_samples =  500000
    total_epochs = 200
    learning_rate = 0.03
    
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    model = cnn_network().to(device)
    model = model.cuda()
    loss_function = pytorch.CrossEntropyLoss().cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate , weight_decay = 0.00005 )
    
    train_dataset = Dataset_load("TrainData.ctf", chunksize=1, nb_samples=total_samples)
    val_dataset = Dataset_load("ValidationData.ctf", chunksize=1, nb_samples=total_samples)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=minibatch_size, num_workers=0, shuffle=False)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, num_workers=0, shuffle=False)
    for epoch in range(total_epochs):
        total_loss = 0
        print("Epoch : {}".format(epoch))
        
        for i, (query_vectors, passage_vectors, labels) in enumerate(train_loader):
            #print(type(query_vectors))
            #print(query_vectors[i])
            #import pdb;pdb.set_trace()
            print(i)
            q = []
            p = []
            l = []
            for vector in query_vectors:
                temp = []
                for t in vector:
                    temp.append(t.numpy().tolist())
                q.append(temp)
             
            q = np.array(q)
            q = np.reshape(q,[1,12,50,minibatch_size])
            q = np.transpose(q,(3,0,1,2))
            
            for vector in passage_vectors:
                temp = []
                for t in vector:
                    temp.append(t.numpy().tolist())
                p.append(temp)
             
            p = np.array(p)
            p = np.reshape(p,[1,50,50,minibatch_size])
            p = np.transpose(p,(3,0,1,2))
            
            for vector in labels:
                temp = []
                for t in vector:    
                    temp.append(t.numpy().tolist())
                    
                l.append(temp)
            
            #print(np.array(l).shape)
            l = np.array(l)
            l = np.reshape(l,[1,minibatch_size])
            l = np.transpose(l,(1,0))
            
            l = np.squeeze(l)
            
            minibatch_X_query = torch.from_numpy(q).float()
            minibatch_X_passage = torch.from_numpy(p).float()
            minibatch_Y = torch.from_numpy(l).long()
			
            minibatch_X_query = minibatch_X_query.cuda()
            minibatch_X_passage = minibatch_X_passage.cuda()
            minibatch_Y = minibatch_Y.cuda()
            #print(minibatch_Y)
            forward_pass = model(minibatch_X_query , minibatch_X_passage)
            print('forward_pass done')
            
            #pdb.set_trace()
            loss = loss_function(forward_pass , minibatch_Y)
            print('after loss function')
            total_loss+=loss.item()
            optimizer.zero_grad()
            loss.backward()
            print('loss calculated')
            optimizer.step()
            print('optimization done')
        predicted_labels=[]
        validation_labels = []
        with torch.no_grad():
            for i, (query_vectors, passage_vectors, labels) in enumerate(val_loader):
                queryVec   = query_vectors[i].numpy().reshape(1,q_max_words,emb_dim)
                passageVec = passage_vectors[i].numpy().reshape(1,p_max_words,emb_dim)
                
                queryVec = torch.from_numpy(queryVec).float()
                passageVec = torch.from_numpy(passageVec).float()
                scores = model(queryVec,passageVec)   # do forward-prop on model to get score  
                if scores[1]>=scores[0]:
                    predictLabel = 1
                else:
                    predictLabel = 0
                predicted_labels.append(predictLabel) 
                validation_labels.append(labels[i])
        metrics = precision_recall_fscore_support(np.array(validation_labels), np.array(predicted_labels), average='binary')
		#print("precision : "+str(metrics[0])+" recall : "+str(metrics[1])+" f1 : "+str(metrics[2])+"\n")
        print("Epoch : {} , Loss : {} Validation Precision : {}  Recall : {}  F1 Score : {}".format(epoch,total_loss/500000,metrics[0],metrics[1],metrics[2]))
        torch.save(model.state_dict(), 'model{}_train.ckpt'.format(epoch))
	
	

## The following GetPredictionOnEvalSet method reads all query passage pair vectors from CTF file and does forward prop with trained model to get similarity score
## after getting scores for all the pairs, the output will be written into submission file. 
def GetPredictionOnEvalSet(model,testfile,submissionfile):
	global q_max_words,p_max_words,emb_dim

	f = open(testfile,'r',encoding="utf-8")
	all_scores={} # Dictionary with key = query_id and value = array of scores for respective passages
	for line in f:
		tokens = line.strip().split("|")  
		#tokens[0] will be empty token since the line is starting with |
		x1 = tokens[1].replace("qfeatures","").strip() #Query Features
		x2 = tokens[2].replace("pfeatures","").strip() # Passage Features
		query_id = tokens[3].replace("qid","").strip() # Query_id
		x1 = [float(v) for v in x1.split()]
		x2 = [float(v) for v in x2.split()]    
		queryVec   = np.array(x1,dtype="float32").reshape(1,q_max_words,emb_dim)
		passageVec = np.array(x2,dtype="float32").reshape(1,p_max_words,emb_dim)
		score = model(queryVec,passageVec) # do forward-prop on model to get score
		if(query_id in all_scores):
			all_scores[query_id].append(score)
		else:
			all_scores[query_id] = [score]
	fw = open(submissionfile,"w",encoding="utf-8")
	for query_id in all_scores:
		scores = all_scores[query_id]
		scores_str = [str(sc) for sc in scores] # convert all scores to string values
		scores_str = "\t".join(scores_str) # join all scores in list to make it one string with  tab delimiter.  
		fw.write(query_id+"\t"+scores_str+"\n")
	fw.close()



if __name__ == '__main__':
	trainSetFileName = "TrainData.ctf"
	validationSetFileName = "ValidationData.ctf"
	testSetFileName = "EvaluationData.ctf"
	submissionFileName = "answer.tsv"
	
	#LoadValidationSet(validationSetFileName)    #Load Validation Query, Passage Vectors from Validation CTF File
	model = TrainAndValidate(trainSetFileName) # Training and validation methods    
	GetPredictionOnEvalSet(model,testSetFileName,submissionFileName) # Get Predictions on Evaluation Set
