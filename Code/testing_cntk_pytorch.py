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
import cntk as C
from cntk.io import MinibatchSource, CTFDeserializer, StreamDef, StreamDefs, INFINITELY_REPEAT, FULL_DATA_SWEEP
from sklearn.metrics import precision_recall_fscore_support
from sklearn.exceptions import UndefinedMetricWarning
import warnings
import numpy as np

torch.multiprocessing.set_sharing_strategy('file_system')
#Initialize Global variables
  
q_max_words=12
p_max_words=50
emb_dim=50


class cnn_network(pytorch.Module):
    def __init__(self):
        super(cnn_network, self).__init__()
        self.convA1 = pytorch.Conv2d(in_channels = 1, out_channels = 4, kernel_size = (3,10))
        self.prelu1 = pytorch.PReLU(num_parameters=4)
        torch.nn.init.normal(self.convA1.weight,std=0.01)
        self.batch_norm1 = pytorch.BatchNorm2d(4)
        self.poolA1 = pytorch.MaxPool2d(kernel_size = (2,3), stride = (2,3))
        self.convA2 = pytorch.Conv2d(in_channels = 4, out_channels = 2, kernel_size = (2,4))
        self.prelu2 = pytorch.PReLU(num_parameters=2)         
        torch.nn.init.normal(self.convA2.weight,std=0.01)
        self.batch_norm2 = pytorch.BatchNorm2d(2)
        self.poolA2 = pytorch.MaxPool2d(kernel_size = (2,2), stride = (2,2))
        self.denseA = pytorch.Linear(20,4) 
        torch.nn.init.normal(self.denseA.weight,std=0.01)
        self.batch_norm3 = pytorch.BatchNorm1d(4)
        
        self.convB1 = pytorch.Conv2d(in_channels = 1, out_channels = 4, kernel_size = (5,10))
        self.prelu3 = pytorch.PReLU(num_parameters=4)                   
        torch.nn.init.normal(self.convB1.weight,std=0.01)
        self.batch_norm4 = pytorch.BatchNorm2d(4)
        self.poolB1 = pytorch.MaxPool2d(kernel_size = (5,5), stride = (5,5))                                     
        self.convB2 = pytorch.Conv2d(in_channels = 4, out_channels = 2, kernel_size = (3,3))
        self.prelu4 = pytorch.PReLU(num_parameters=2)                   
        torch.nn.init.normal(self.convB2.weight,std=0.01)
        self.batch_norm5 = pytorch.BatchNorm2d(2)
        self.poolB2 = pytorch.MaxPool2d(kernel_size = (2,2), stride = (2,2))                                      
        self.denseB = pytorch.Linear(18,4)                                                                        
        torch.nn.init.normal(self.denseB.weight,std=0.01)
        self.batch_norm6 = pytorch.BatchNorm1d(4)
        
        self.model = pytorch.Linear(4,2)                     
                                                             #in: 4      out: 2
        torch.nn.init.normal(self.model.weight,std=0.01)                                                                          #in: 4      out: 2
    def forward(self,query_features,passage_features):
        query_features = self.poolA1(self.batch_norm1(self.prelu1(self.convA1(query_features))))
        query_features = self.poolA2(self.batch_norm2(self.prelu2(self.convA2(query_features))))
        query_features = query_features.view(-1,20)
        query_features = self.batch_norm3(F.relu(self.denseA(query_features)))
		
        passage_features = self.poolB1(self.batch_norm4(self.prelu3(self.convB1(passage_features))))
        passage_features = self.poolB2(self.batch_norm5(self.prelu4(self.convB2(passage_features))))
        passage_features = passage_features.view(-1,18)
        passage_features = self.batch_norm6(F.relu(self.denseB(passage_features)))
		
        merged = query_features * passage_features
		
        output = self.model(merged)
		
        return output

model = cnn_network()
model.load_state_dict(torch.load('model29_train_LR.ckpt'))
#model = torch.load('model29_train.ckpt')
model.eval()
        
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
        queryVec   = np.array(x1,dtype="float32").reshape(1,1,q_max_words,emb_dim)
        passageVec = np.array(x2,dtype="float32").reshape(1,1,p_max_words,emb_dim)
        
        #queryVec = torch.from_numpy(query_Vec).float()
        
        queryVec = torch.from_numpy(queryVec).float()
        passageVec = torch.from_numpy(passageVec).float()
        
        
        
        score = model(queryVec,passageVec)[0][1].detach().numpy() # do forward-prop on model to get score
        #print(score)
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
	#model = TrainAndValidate(trainSetFileName) # Training and validation methods    
	GetPredictionOnEvalSet(model,testSetFileName,submissionFileName) # Get Predictions on Evaluation Set
