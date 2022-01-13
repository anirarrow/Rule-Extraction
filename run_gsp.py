from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
import pandas as pd
import pandas
import numpy as np

from trepan import Trepan,Oracle
import sys

###########################################
children=dict()

def load_landsat_data(filename):
	'''
	Utility function to load Boston housing dataset.
	num_classes= 2
	This functions
	- Reads the data
	- Generates one-hot vector labels
	'''

	#data = pandas.read_csv(filename, sep=r"\s+", header=None)
	df = pandas.read_csv(filename, sep=",")
	#Balance the data
	data_rejected=df.loc[(df['UW_DECISION'] == 0)]
	data_accepted=df.loc[(df['UW_DECISION'] == 1)].sample(n=data_rejected.shape[0])
	data=pd.concat([data_accepted,data_rejected]).drop_duplicates().reset_index(drop=True)
	
	#Shuffle data
	data=data.sample(frac=1).reset_index(drop=True)
	
	
	
	#data=df
	#data.to_csv("orig_data_only_rejection.csv")
	datax=data.drop(['UW_DECISION'],axis=1)
	dataX = datax.values
	dataY= np.array(data['UW_DECISION'])

	print(dataX.shape)
	print(dataY.shape)

	#dataX = np.array(data[:,range(data.shape[1]-1)])
	#dataY = np.array(data[np.arange(data.shape[0]),data.shape[1]-1])

	# convert dataY to one-hot, 6 classes
	num_classes = 2
	#dataY = np.array([x-2 if x==7 else x-1 for x in dataY]) # re-named class 7 to 6 as class 6 is empty
	dataY_onehot = np.zeros([dataY.shape[0], num_classes])
	#print(dataY_onehot)
	#print(dataY_onehot.shape[0])
	
	dataY=dataY.astype(float)
	#print len(dataY)
	#print(dataY[1])

	#Changing the output nodes manually based on whether y=0 or 1. This needs to be changed if num_classes!=2.
	for i in range(dataY.shape[0]):
		if dataY[i]==1:
			dataY_onehot[i,1]=1
		elif dataY[i]==0:
			dataY_onehot[i,0]=1
	#print(dataY_onehot)
	#dataY_onehot[np.arange(dataY_onehot.shape[0]), dataY] = 1
	
	return dataX, dataY_onehot

def create_model (trainX,trainY,num_classes,layer1_width,layer2_width,MAX_NODES):
	#Best combination 500*2, adagrad, epoch=50, batch_size=32
	model = Sequential()
	
	if layer2_width==None:	
		print("Building model with 1 layer of width:"+str(layer1_width))
		model.add(Dense(layer1_width, input_dim=trainX.shape[1], activation="sigmoid"))
		model.add(Dense(num_classes, activation="softmax"))
		model.compile(loss="categorical_crossentropy", optimizer="adagrad", metrics=['accuracy'])
		model.fit(trainX, trainY, epochs=30, batch_size=32,shuffle=True) # epochs=150
		model.save('../../models/model_1HL_'+str(layer1_width)+'_maxnodes'+str(MAX_NODES)+'_balanced_sigmoid.important.without_BR_ID.h5')
	else:
		print("Building model with 2 layers of widths:"+str(layer1_width)+','+str(layer2_width))
		model.add(Dense(layer1_width, input_dim=trainX.shape[1], activation="sigmoid"))
		model.add(Dense(layer2_width, activation="sigmoid"))
		model.add(Dense(num_classes, activation="softmax"))
		model.compile(loss="categorical_crossentropy", optimizer="adagrad", metrics=['accuracy'])
		model.fit(trainX, trainY, epochs=30, batch_size=32,shuffle=True) # epochs=150
		model.save('../../models/model_2HL_'+str(layer1_width)+'_'+str(layer2_width)+'_maxnodes'+str(MAX_NODES)+'_balanced_sigmoid.important.without_BR_ID.h5')
	return model	
	

def count_nodes(root,c,visited):
	if root.splitrule!=None and root.splitrule.splits not in visited:
		c+=1
		visited.append(root.splitrule.splits)
	if root.left_child!=None and root.left_child.splitrule!=None:
		return count_nodes(root.left_child,c,visited)
	if root.right_child!=None and root.right_child.splitrule!=None:
		return count_nodes(root.right_child,c,visited)
	return c

def get_oracle_and_rules(model,num_classes,trainX,layer1_width,layer2_width):
	print("Getting oracle.")
	##############################################
	#Here, the oracle mainly gets the feature distributions of the training data
	#Each column has a distribution, which is appended to a list. This list contains
	#dists for all columns. This list is one of the instance variables of the Oracle.
	oracle = Oracle(model,num_classes,trainX)
	##############################################

	#build tree with TREPAN
	MIN_EXAMPLES_PER_NODE = 30
	MAX_NODES=int(sys.argv[4])
	print("Building tree.")
	root=Trepan.build_tree(MIN_EXAMPLES_PER_NODE,MAX_NODES,trainX,oracle)
	print("Counting nodes.")
	num_nodes=count_nodes(root,0,[])
	print("Number of nodes in the tree: "+str(num_nodes))

	#calculate fidelity
	num_test_examples= testX.shape[0]
	correct=0
	rule_acc=0	#Accuracy of the rule (comparing rule o/p with ground labels)
	for i in range(0,num_test_examples):
		ann_prediction = oracle.get_oracle_label(testX[i,:])
		tree_prediction = root.classify(testX[i,:],layer1_width,layer2_width)

		print("Oracle prediction="+str(ann_prediction))
		print("Tree prediction="+str(tree_prediction))
		#Take the argmax of test labels to get the ground truth
		if testY[i][0]>testY[i][1]:
			gl=0.0
			print("Ground label:0.0")
		elif testY[i][0]<testY[i][1]:
			gl=1.0
			print("Ground label:1.0")
		correct += (ann_prediction==tree_prediction)
		if gl==tree_prediction:
			rule_acc+=1

	fidelity=float(correct)/num_test_examples
	rule_acc=rule_acc/float(num_test_examples)
	print("Fidelity of the model is : "+str(fidelity))
	print("Rule accuracy is:"+str(rule_acc))


trainX, trainY = load_landsat_data("../../data/data_gsp_ordinal_withdecision2.important.without_BR_ID.csv")
testX, testY = load_landsat_data("../../data/data_gsp_ordinal_withdecision2.important.without_BR_ID.csv")
num_classes = trainY.shape[1]
total_num_examples = trainX.shape[0]
print(num_classes,total_num_examples)



num_layers=int(sys.argv[1])

if num_layers>2 or num_layers<=0:
	print("Number of layers incorrect.")
	exit(0)
max_layer1_width=int(sys.argv[2])
if num_layers==1:
	max_layer2_width=0
else:
	max_layer2_width=int(sys.argv[3])


MAX_NODES=int(sys.argv[4])	

#build oracle
if num_layers==1:
	for layer1_width in range(100,max_layer1_width,50):
		model = create_model(trainX,trainY,num_classes,layer1_width,None,MAX_NODES)
		get_oracle_and_rules(model,num_classes,trainX,layer1_width,layer2_width)
elif num_layers==2:
	for layer1_width in range(10,max_layer1_width,50):
		for layer2_width in range(10,max_layer2_width,50):
			model = create_model(trainX,trainY,num_classes,layer1_width,layer2_width,MAX_NODES)
			get_oracle_and_rules(model,num_classes,trainX,layer1_width,layer2_width)
		#print(trainX)
		#exit(0)
		
