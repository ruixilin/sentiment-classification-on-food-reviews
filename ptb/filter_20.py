import pandas as pd
import numpy as np
import re


#raw_data_x_train = pd.read_csv(r'ptb.trainx20.txt', header = None, index_col = None)
#raw_data_x_train.columns = ['Text']

#raw_data_x_train = pd.read_csv(r'ptb.validx20.txt', header = None, index_col = None)
#raw_data_x_train.columns = ['Text']

raw_data_x_train = pd.read_csv(r'ptb.testx20.txt', header = None, index_col = None)
raw_data_x_train.columns = ['Text']


#raw_data_y_train = pd.read_csv(r'ptb.trainy20.txt', header = None, index_col = None)
#raw_data_y_train.columns = ['Score']

#raw_data_y_train = pd.read_csv(r'ptb.validy20.txt', header = None, index_col = None)
#raw_data_y_train.columns = ['Score']

raw_data_y_train = pd.read_csv(r'ptb.testy20.txt', header = None, index_col = None)
raw_data_y_train.columns = ['Score']

counter = []
for ind, row in raw_data_x_train.iterrows():
	count = len(re.findall(r'\s+', row['Text']))
	counter.append(count+1)	
raw_data_x_train['wordcount'] = pd.Series(counter)
raw_data_y_train = raw_data_y_train[raw_data_x_train['wordcount']<=87]
raw_data_x_train = raw_data_x_train[raw_data_x_train['wordcount']<=87]  
print raw_data_x_train.head() 
raw_data_x_train=raw_data_x_train.drop('wordcount',axis=1)
raw_data_x_train = raw_data_x_train.head(int(len(raw_data_x_train.index)/64)*64)
print len(raw_data_x_train.index)

#raw_data_x_train.to_csv(r'ptb.trainx20_1.txt', header = False, index = False)
#raw_data_y_train.to_csv(r'ptb.trainy20_1.txt', header = False, index = False)


raw_data_x_train.to_csv(r'ptb.testx20_1.txt', header = False, index = False)
raw_data_y_train.to_csv(r'ptb.testy20_1.txt', header = False, index = False)


#raw_data_x_train.to_csv(r'ptb.validx20_1.txt', header = False, index = False)
#raw_data_y_train.to_csv(r'ptb.validy20_1.txt', header = False, index = False)
