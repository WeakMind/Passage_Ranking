import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from torch.utils.data.dataset import Dataset
import glob
import pickle
from torchvision import transforms
from model import BiMPM

class Dataset1(Dataset):
	def __init__(self, filename):
		self.data = pickle.load(open(filename, "rb"))
		self.max_query_len = 12
		self.max_passage_len = 50
		self.len = len(self.data)
		
	
	def __len__(self):
		return self.len

	def __getitem__(self, index):
		query, passage, label = self.data[index]
		query = np.array([float(v) for v in query.split()])
		passage = np.array([float(v) for v in passage.split()])
		label = int(label)

		query = query.reshape( self.max_query_len, -1)
		passage = passage.reshape( self.max_passage_len, -1)
		
		return query, passage, label
			

	


def train():
    num_epochs = 1000
    minibatch_size = 200
    weights = [0.01, 0.09] #[ 1 / number of instances for each class]
    validation_files = []
    epoch_size = 5
    
    model = BiMPM(300,20)
    model = model.cuda()
    class_weights = torch.FloatTensor(weights).cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.1).cuda()
    criterion = nn.CrossEntropyLoss(weight = class_weights).cuda()
    
    validation_files = random.sample(glob.glob("/media/Sonaal/Microsoft_challenge/*.pkl"), 10)
    
    for epoch in range(num_epochs):
        
        ## TRAINING 
        for i in range(epoch_size):
            for batch, file in enumerate(random.sample(glob.glob("/media/Sonaal/Microsoft_challenge/*.pkl"), 5)):
                if file in validation_files:
                    continue
                
                train_dataset = Dataset1(file)
                train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=minibatch_size, num_workers=4, shuffle=True)
    			
                for i, (query, passage, label) in enumerate(train_loader):
                    query = query.float().cuda()
                    passage = passage.float().cuda()
                    label = label.long().cuda()
                    
                    output = model(query,passage,True)
                    optimizer.zero_grad()
                    loss = criterion(output, label)
                    loss.backward()
                    optimizer.step()
                    
        ## VALIDATION
        predicted_labels = []
        actual_labels = []
        for batch,file in enumerate(validation_files):
            train_dataset = Dataset1(file)
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = minibatch_size, number_workers, shuffle = False)
            
            for i,(query,passage,label) in enumerate(train_loader):
                query = query.float().cuda()
                passage = passage.float().cuda()
                label = label.long()
                
                output = model(query,passage,False)
                temp = []
                for out in output:
                    if out[1]>=0.5:
                        temp.append(1)
                    else:
                        temp.append(0)
                predicted_labels+= temp
                actual_labels+= list(label) 
        metrics = precision_recall_fscore_support(np.array(actual_labels), np.array(predicted_labels), average='binary')
		 #print("precision : "+str(metrics[0])+" recall : "+str(metrics[1])+" f1 : "+str(metrics[2])+"\n")
        print("Epoch : {} , Loss : {} Validation Precision : {}  Recall : {}  F1 Score : {}".format(epoch,(total_loss/validation_samples)*minibatch_size,metrics[0],metrics[1],metrics[2]))
        torch.save(model.state_dict(), '/saved_weights/model{}_train.ckpt'.format(epoch))

if __name__ == '__main__':
	train()