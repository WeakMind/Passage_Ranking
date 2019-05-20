# Search_Engine
A Deep Learning model that ranks 10 passages based on similarity with the query.

DataSet - https://competitions.codalab.org/competitions/20616

Language used - Python

Framework used - Pytorch

Architecture - A bilateral multi-perspective matching (BiMPM) model is implemented. Given two sentences P and Q, the model first encodes them with a BiLSTM encoder. Next, it matches the two encoded sentences in two directions P against Q and Q against P. In each matching direction, each time step of one sentence is matched against all timesteps of the other sentence from multiple perspectives. Then, another BiLSTM layer is utilized to aggregate the matching results into a fixed-length matching vector. Finally, based on the matching vector, a decision is made through a fully connected layer.



