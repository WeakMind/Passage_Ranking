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
		query_features = F.relu(self.denseA(query_features))
		
		passage_features = self.poolB1(F.relu(self.convB1(passage_features)))
		passage_features = self.poolB2(F.relu(self.convB2(passage_features)))
		passage_features = passage_features.view(-1,18)
		passage_features = F.relu(self.denseB(passage_features))
		
		merged = query_features * passage_features
		
		output = self.model(merged)
		
		return output

def create_reader(path, is_training, query_total_dim, passage_total_dim, label_total_dim):
    return MinibatchSource(CTFDeserializer(path, StreamDefs( queryfeatures = StreamDef(field='qfeatures', shape=query_total_dim,is_sparse=False), 
                                                            passagefeatures = StreamDef(field='pfeatures', shape=passage_total_dim,is_sparse=False), 
                                                            labels   = StreamDef(field='labels', shape=label_total_dim,is_sparse=False)
                                                            )), 
                           randomize=is_training, max_sweeps = INFINITELY_REPEAT if is_training else FULL_DATA_SWEEP)

def TrainAndValidate(trainfile):

	#*****Hyper-Parameters******
    q_max_words= 12
    p_max_words = 50
    emb_dim = 50
    num_classes = 2
    minibatch_size = 250
    total_samples =  200000
    validation_samples = 500000
    total_epochs = 30
    learning_rate = 0.1
    
    
    query_total_dim = q_max_words*emb_dim
    label_total_dim = num_classes
    passage_total_dim = p_max_words*emb_dim
    
    
    #****** Create placeholders for reading Training Data  ***********
    query_input_var =  C.ops.input_variable((1,q_max_words,emb_dim),np.float32,is_sparse=False)
    passage_input_var =  C.ops.input_variable((1,p_max_words,emb_dim),np.float32,is_sparse=False)
    labels = C.input_variable(num_classes,np.float32,is_sparse = False)
    train_reader = create_reader("TrainData.ctf", True, query_total_dim, passage_total_dim, label_total_dim)
    validation_reader = create_reader("ValidationData.ctf", False, query_total_dim, passage_total_dim, label_total_dim)
    input_map = { query_input_var : train_reader.streams.queryfeatures, passage_input_var:train_reader.streams.passagefeatures, labels : train_reader.streams.labels}

    
    
    model = cnn_network().to('cuda')
    model = model.cuda()
    weights = [0.1, 0.9] #[ 1 / number of instances for each class]
    class_weights = torch.FloatTensor(weights).cuda()
    loss_function = pytorch.CrossEntropyLoss(weight=class_weights).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate , weight_decay = 0.00005 )
    
    
    for epoch in range(total_epochs):
        total_loss = 0
        print("Epoch : {}".format(epoch))
        sample_count=0
        while sample_count < total_samples:
            #sample_count+= minibatch_size
            #pdb.set_trace()
            
            mini_size = min(minibatch_size, total_samples - sample_count)
           
            data = train_reader.next_minibatch(mini_size, input_map=input_map) # fetch minibatch.
            sample_count += minibatch_size
            query_vectors = np.array(data[query_input_var].as_sequences(variable=None))
            query_vectors = np.reshape(query_vectors,(query_vectors.shape[0],1,12,50))
            
            passage_vectors = np.array(data[passage_input_var].as_sequences(variable=None))
            passage_vectors = np.reshape(passage_vectors,(passage_vectors.shape[0],1,50,50))
            
            output = np.array(data[labels].as_sequences(variable=None))
            output = np.reshape(output,(output.shape[0],2))
            
            temp=[]
            for l in output:
                if l[1]==1:
                    temp.append(1)
                else:
                    temp.append(0)
            output = np.array(temp)
            
            
            minibatch_X_query = torch.from_numpy(query_vectors).float()
            minibatch_X_passage = torch.from_numpy(passage_vectors).float()
            minibatch_Y = torch.from_numpy(output).long()
			
            minibatch_X_query = minibatch_X_query.cuda()
            minibatch_X_passage = minibatch_X_passage.cuda()
            minibatch_Y = minibatch_Y.cuda()
            #print(minibatch_Y)
            forward_pass = model(minibatch_X_query , minibatch_X_passage)
            
            #pdb.set_trace()
            loss = loss_function(forward_pass , minibatch_Y)
            total_loss+=loss.item()
            print(loss.item())
            optimizer.zero_grad()
            loss.backward()
            
            optimizer.step()
        print('Loss after {} epoch = {}'.format(epoch,(total_loss/total_samples)*minibatch_size))   
        predicted_labels=[]
        validation_labels = []
        #with torch.no_grad():
        sample_count = 0
        while sample_count < validation_samples:
            #sample_count+= minibatch_size
            data = validation_reader.next_minibatch(min(minibatch_size,validation_samples-minibatch_size), input_map=input_map)
            sample_count += minibatch_size
            query_vectors = np.array(data[query_input_var].as_sequences(variable=None))
            query_vectors = np.reshape(query_vectors,(query_vectors.shape[0],1,12,50))
            
            
            passage_vectors = np.array(data[passage_input_var].as_sequences(variable=None))
            passage_vectors = np.reshape(passage_vectors,(passage_vectors.shape[0],1,50,50))
        
            output = np.array(data[labels].as_sequences(variable=None))
            output = np.reshape(output,(output.shape[0],2))
            
            temp=[]
            for l in output:
                if l[1]==1:
                    temp.append(1)
                else:
                    temp.append(0)
            output = temp
            
            queryVec = torch.from_numpy(query_vectors).float()
            passageVec = torch.from_numpy(passage_vectors).float()
            
            queryVec = queryVec.cuda()
            passageVec = passageVec.cuda()
            scores = model(queryVec,passageVec)   # do forward-prop on model to get score  
            predictLabel = []
            for score in scores:
                if score[1] >= score[0]:
                    predictLabel.append(1)
                else:
                    predictLabel.append(0)
            
            predicted_labels = predicted_labels + predictLabel 
            validation_labels = validation_labels + output
        metrics = precision_recall_fscore_support(np.array(validation_labels), np.array(predicted_labels), average='binary')
		#print("precision : "+str(metrics[0])+" recall : "+str(metrics[1])+" f1 : "+str(metrics[2])+"\n")
        print("Epoch : {} , Loss : {} Validation Precision : {}  Recall : {}  F1 Score : {}".format(epoch,(total_loss/validation_samples)*minibatch_size,metrics[0],metrics[1],metrics[2]))
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
        queryVec   = np.array(x1,dtype="float32").reshape(1,1,q_max_words,emb_dim)
        passageVec = np.array(x2,dtype="float32").reshape(1,1,p_max_words,emb_dim)
        
        queryVec = torch.from_numpy(queryVec).float()
        passageVec = torch.from_numpy(passageVec).float()
        score = model(queryVec,passageVec)[0][1].detach().numpy()
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