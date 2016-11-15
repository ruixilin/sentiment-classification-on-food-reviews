from collections import defaultdict

import numpy as np
import pandas as pd

class Vocab(object):
  def __init__(self):
    self.word_to_index = {} # word_to_index is a dictionary
    self.index_to_word = {}
    self.word_freq = defaultdict(int)
    self.total_words = 0
    self.unknown = '<unk>'
    self.add_word(self.unknown, count=0)  #initialze, put '<unk>' into the vocab first  #why add this first and not count as a word?

  '''add_word has bug!'''
  def add_word(self, word, count=1):
    if word not in self.word_to_index:  
      index = len(self.word_to_index)
      self.word_to_index[word] = index 
      self.index_to_word[index] = word
    self.word_freq[word] += count

  def construct(self, words):
    for word in words:
      self.add_word(word)
    self.total_words = float(sum(self.word_freq.values()))
    print '{} total words with {} uniques'.format(self.total_words, len(self.word_freq))

  def encode(self, word):
    if word not in self.word_to_index:
      word = self.unknown
    return self.word_to_index[word]

  def decode(self, index):
    return self.index_to_word[index]

  def __len__(self):
    return len(self.word_freq)

def calculate_perplexity(log_probs):
  # https://web.stanford.edu/class/cs124/lec/languagemodeling.pdf
  perp = 0
  for p in log_probs:
    perp += -p
  return np.exp(perp / len(log_probs))


def get_ptb_dataset(dataset, max_len):
  fn = 'data/ptb/ptb.{}.txt'
  countline = 0
  for line in open(fn.format(dataset)):
    len_line = 0
    countline+=1
    for word in line.split():
      len_line += 1
      yield word
    if len_line > max_len:
      max_len = len_line
    # Add token to the end of the line
    # Equivalent to <eos> in:
    # https://github.com/wojzaremba/lstm/blob/master/data.lua#L32
    # https://github.com/tensorflow/tensorflow/blob/master/tensorflow/models/rnn/ptb/reader.py#L31
    yield '<eos>'
  print "x max len: ", max_len
  print "num of comments",countline

def get_ptb_dataset_y(dataset, max_len):
  fn = 'data/ptb/ptb.{}.txt'
  for line in open(fn.format(dataset)):
    len_line = 0
    for word in line.split():
      len_line += 1
      yield word
    if len_line > max_len:
      max_len = len_line
    # Add token to the end of the line
    # Equivalent to <eos> in:
    # https://github.com/wojzaremba/lstm/blob/master/data.lua#L32
    # https://github.com/tensorflow/tensorflow/blob/master/tensorflow/models/rnn/ptb/reader.py#L31


def ptb_iterator(raw_data_x, raw_data_y, batch_size, num_steps,vocab):
  # Pulled from https://github.com/tensorflow/tensorflow/blob/master/tensorflow/models/rnn/ptb/reader.py#L82  

  raw_data_x = raw_data_x.reshape(-1)
  raw_data_x = raw_data_x.tolist()
  #print len(raw_data_x)
  #output = [vocab.decode(word_ind) for word_ind in raw_data_x]  
  #print output
  max_len = 88;
  counter = 0
  countery = 0;
  i=0
  eos_encode = vocab.encode('<eos>')
  step_data_y = [];
  num_of_comments= 0;
  while i < len(raw_data_x):
    if raw_data_x[i] != eos_encode:
      i += 1
      counter += 1
    else:
      #compensate to max length  
      num_zero = max_len - counter-1
      num_of_comments+=1
      for j in xrange(num_zero):
        raw_data_x.insert(i-counter,0)
        i += 1
      #append y at each step  
      for j in xrange(max_len/num_steps): 
        step_data_y.append(raw_data_y[countery])
      #print raw_data_x[i-counter-10:i]
      """
      output = [vocab.decode(word_ind) for word_ind in raw_data_x[i-counter-num_zero:i]]
      print output
      print raw_data_y[countery]
      
      if(countery==0):
        print num_zero
        print "setence: ",countery
        print i
        print len(raw_data_x)
        output = [vocab.decode(word_ind) for word_ind in raw_data_x[0:i]]
        print output
      """  
      counter = 0
      countery += 1
      i += 1 #skip eos
      
      
  raw_data_x = np.array(raw_data_x, dtype=np.int32)    
  raw_data_y = np.array(step_data_y,dtype = np.int32)
  data_len = len(raw_data_x)
  data_leny = len(raw_data_y)
  batch_len = data_len // batch_size
  batch_leny = data_leny//batch_size
  #print "batch_leny",batch_leny
  data_x = np.zeros([batch_size, batch_len], dtype=np.int32)
  data_y = np.zeros([batch_size, batch_leny], dtype=np.int32)
  for i in range(batch_size):
    data_x[i] = raw_data_x[batch_len * i:batch_len * (i + 1)]
    data_y[i] = raw_data_y[batch_leny * i:batch_leny * (i + 1)]
  #print raw_data_x.shape  
  #print data_y.shape  
  epoch_size = (batch_len - 1) // num_steps
  if epoch_size == 0:
    raise ValueError("epoch_size == 0, decrease batch_size or num_steps")
  print "number of comments",num_of_comments  
  print "batch_len", batch_len  
  print "data_len",data_len 
  print "epoch_size", epoch_size  
  for i in range(epoch_size):
    x = data_x[:, i * num_steps:(i + 1) * num_steps]
    y = data_y[:, i]-1# 
    """
    if i<5:
      sample = x[0:5,:].reshape(-1)   
      output = [vocab.decode(word_ind) for word_ind in sample]
      print output
      print y[0:5].reshape(-1)  
    """ 
    yield (x, y)

def sample(a, temperature=1.0):
    # helper function to sample an index from a probability array
    # from https://github.com/fchollet/keras/blob/master/examples/lstm_text_generation.py
    a = np.log(a) / temperature
    a = np.exp(a) / np.sum(np.exp(a))
    return np.argmax(np.random.multinomial(1, a, 1))

def data_iterator(orig_X, orig_y=None, batch_size=32, label_size=2, shuffle=False):
  # Optionally shuffle the data before training
  if shuffle:
    indices = np.random.permutation(len(orig_X))
    data_X = orig_X[indices]
    data_y = orig_y[indices] if np.any(orig_y) else None
  else:
    data_X = orig_X
    data_y = orig_y
  ###
  total_processed_examples = 0
  total_steps = int(np.ceil(len(data_X) / float(batch_size)))
  for step in xrange(total_steps):
    # Create the batch by selecting up to batch_size elements
    batch_start = step * batch_size
    x = data_X[batch_start:batch_start + batch_size]
    # Convert our target from the class index to a one hot vector
    y = None
    if np.any(data_y):
      y_indices = data_y[batch_start:batch_start + batch_size]
      y = np.zeros((len(x), label_size), dtype=np.int32)
      y[np.arange(len(y_indices)), y_indices] = 1
    ###
    yield x, y
    total_processed_examples += len(x)
  # Sanity check to make sure we iterated over all the dataset as intended
  assert total_processed_examples == len(data_X), 'Expected {} and processed {}'.format(len(data_X), total_processed_examples)
