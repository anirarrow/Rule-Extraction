#Script to preprocess data for trepan
import pandas as pd
import random

def create_train_test_files(fname):
	fin=open(fname,'r')
	#POL_MPREM_AMT	TOT_RISK_COVER	CVG_MAT_XPRY_DUR	CVG_SUM_INS_AMT	CLI_EARN_INCM_AMT	CLI_OINS_TOT_AMT	age	BMI	CLI_SMKR_CD	UW_DECISION


	ftrain=open('./data/underwriting_trn.csv','w')
	ftest=open('./data/underwriting_tst.csv','w')
	lol=set(fin.readlines())
	tot=len(lol)
	train_size=int(tot*0.8)
	test_size=tot-train_size

	lotrl=set(random.sample(lol,train_size))
	for line in lotrl:
		ftrain.write(line)

	for line in lol:
		if line not in lotrl:
			ftest.write(line)
	fin.close()
	ftrain.close()
	ftest.close()
	print("Training and test files written successfully.")
def preprocess_insurance_data():
	'''
	Preprocess the encoded version of German Credit Dataset (obtained from Neurorule repo).
	'''
	#Remove odd kind of lines from the main data file
	fname='./data/Data_Trepan_new_numeric.csv'
	fin=open(fname,'r')
	fout=open('./data/underwriting.correct.data.csv','w')
	for line in fin:
		l=line.rstrip().split(',')
		if len(l)==47:
			#Swap the last and second last columns
			temp=l[len(l)-1]
			l[len(l)-1]=l[len(l)-2]
			l[len(l)-2]=temp
			fout.write(' '.join(l)+'\n')
	fin.close()
	fout.close()
	
	#Now read the clean data and generate training and test files
	fname='./data/underwriting.correct.data.csv'
		
	df=pd.read_csv(fname)
	fout=open('./data/underwriting_trepan.csv','w')
	op_df=pd.DataFrame()	#Output numeric dataframe
	list_of_dicts=[]
	#Get the list of dicts for object types (string data)
	for j in range(0,len(df.columns)):
		df_col=(df.iloc[:,j])
		if df_col.dtype=='object':
			print("Flushing dict.")
			dict_values=dict()
			c=0
			for item in df_col:
				if item not in dict_values.keys():
					c+=1
					dict_values[item]=c
			list_of_dicts.append(dict_values)
	#list_of_dicts contains the dicts corresponding to each object feature

	#Now write the numeric features in fout using the dicts in list_of_dicts
	fin=open(fname,'r')
	c=0
	for line in fin:
		if c==0:
			c+=1
			continue
		l=line.rstrip().split(' ')
		list_of_numeric_features=[]
		string=''
		for item in l:
			try:
				itm=int(item)
				string+=str(itm)+' '
			except:
				for d in list_of_dicts:
					if item in d.keys():
						itm=d[item]
						break
				string+=str(itm)+' '

		print(string)
		#Change the last label to 0 from 1, and 1 from 2.
		str_list=string.rstrip().split(' ')
		#print(str_list[len(str_list)-1])
		str_list[len(str_list)-1]=str(int(str_list[len(str_list)-1])-1)
		string=' '.join(str_list)
		fout.write(string.rstrip()+'\n')
	print("FILE WRITTEN.")
	fout.close()
	fin.close()

def preprocess_kaggle_data():
	'''
	Preprocess the encoded version of German Credit Dataset (obtained from Neurorule repo).
	'''
	fname='./data/test_Kaggle_german.csv'
	df=pd.read_csv(fname)
	fout=open('./data/german_kaggle.tst','w')
	op_df=pd.DataFrame()	#Output numeric dataframe
	list_of_dicts=[]
	#Get the list of dicts for object types (string data)
	for j in range(0,len(df.columns)):
		df_col=(df.iloc[:,j])
		if df_col.dtype=='object':
			print("Flushing dict.")
			dict_values=dict()
			c=0
			for item in df_col:
				if item not in dict_values.keys():
					c+=1
					dict_values[item]=c
			list_of_dicts.append(dict_values)
	#list_of_dicts contains the dicts corresponding to each object feature

	#Now write the numeric features in fout using the dicts in list_of_dicts
	fin=open(fname,'r')
	c=0
	for line in fin:
		if c==0:
			c+=1
			continue
		l=line.rstrip().split(',')
		list_of_numeric_features=[]
		string=''
		for item in l:
			try:
				itm=int(item)
				string+=str(itm)+' '
			except:
				for d in list_of_dicts:
					if item in d.keys():
						itm=d[item]
						break
				string+=str(itm)+' '
		#Change the last label to 0 from 1, and 1 from 2.
		str_list=string.rstrip().split(' ')
		#print(str_list[len(str_list)-1])
		str_list[len(str_list)-1]=str(int(str_list[len(str_list)-1])-1)
		string=' '.join(str_list)
		fout.write(string.rstrip()+'\n')
	print("FILE WRITTEN.")
	fout.close()
	fin.close()


	#print(list_of_dicts)
def preprocess_german_data():
	'''
	Preprocess the encoded version of German Credit Dataset (obtained from Neurorule repo).
	'''
	fin=open('german_new.tst')
	#ftrain=open('german.trn','w')
	ftest=open('german.tst','w')
	c=0
	for line in fin:
		c+=1
		l=line.rstrip().split(' ')
		label=l[0]
		features=l[1:len(l)]
		features.append(label)
		#if c<=800:
		#ftrain.write(' '.join(features)+'\n')
		#else:
		ftest.write(' '.join(features)+'\n')
	#ftrain.close()
	ftest.close()

if __name__ == '__main__':
	#preprocess_german_data()
	#preprocess_kaggle_data()
	#preprocess_insurance_data()
	create_train_test_files('data/Data_Trepan_Numeric.csv')