import getpass
import sys
import time

import numpy as np
from copy import deepcopy

from utilsMaxlen import calculate_perplexity, get_ptb_dataset, get_ptb_dataset_y,Vocab
from utilsMaxlen import ptb_iterator, sample

import tensorflow as tf
from tensorflow.python.ops.seq2seq import sequence_loss
from model import LanguageModel

from collections import Counter

# Let's set the parameters of our model
# http://arxiv.org/pdf/1409.2329v4.pdf shows parameters that would achieve near
# SotA numbers

class Config(object):
  """Holds model hyperparams and data information.

  The config class is used to store various hyperparameters and dataset
  information parameters. Model objects are passed a Config() object at
  instantiation.
  """
  batch_size = 64
  embed_size = 50
  hidden_size = 100
  num_steps = 8
  max_epochs = 16
  early_stopping = 2
  dropout = 0.9#0.9#
  lr = 1e-5#1e-5#0.0001 0.001#0.01#0.0003
  l2 = 0.006#0.003
  label_size = 5

class RNNLM_Model(LanguageModel):

  def load_data(self, debug=False):
    """Loads starter word-vectors and train/dev/test data."""
    self.vocab = Vocab()
    """
    self.vocab.construct(get_ptb_dataset('trainx',0)) #???
    self.encoded_trainx = np.array(
        [self.vocab.encode(word) for word in get_ptb_dataset('trainx',0)],
        dtype=np.int32)
    self.encoded_trainy = np.array(
        [label for label in get_ptb_dataset_y('trainy',0)],
        dtype=np.int32)
    self.encoded_validx = np.array(
        [self.vocab.encode(word) for word in get_ptb_dataset('validx',0)],
        dtype=np.int32)
    self.encoded_validy = np.array(
        [label for label in get_ptb_dataset_y('validy',0)],
        dtype=np.int32)
    self.encoded_testx = np.array(
        [self.vocab.encode(word) for word in get_ptb_dataset('testx',0)],
        dtype=np.int32)
    self.encoded_testy = np.array(
        [label for label in get_ptb_dataset_y('testy',0)],
        dtype=np.int32)          
    """
 
    self.vocab.construct(get_ptb_dataset('trainx88_1',0)) 
    self.encoded_trainx = np.array(
        [self.vocab.encode(word) for word in get_ptb_dataset('trainx88_1',0)],
        dtype=np.int32)
    self.encoded_trainy = np.array(
        [label for label in get_ptb_dataset_y('trainy88_1',0)],
        dtype=np.int32)
    self.encoded_validx = np.array(
        [self.vocab.encode(word) for word in get_ptb_dataset('validx88_1',0)],
        dtype=np.int32)
    self.encoded_validy = np.array(
        [label for label in get_ptb_dataset_y('validy88_1',0)],
        dtype=np.int32)
    self.encoded_testx = np.array(
        [self.vocab.encode(word) for word in get_ptb_dataset('testx88_1',0)],
        dtype=np.int32)
    self.encoded_testy = np.array(
        [label for label in get_ptb_dataset_y('testy88_1',0)],
        dtype=np.int32)       
    if debug:
      num_debug = 1024
      self.encoded_trainx = self.encoded_trainx[:num_debug]
      self.encoded_validx = self.encoded_validx[:num_debug]
      self.encoded_testx = self.encoded_testx[:num_debug]
      self.encoded_trainy = self.encoded_trainy[:num_debug]
      self.encoded_validy = self.encoded_validy[:num_debug]
      self.encoded_testy = self.encoded_testy[:num_debug]

  def add_placeholders(self):
    """Generate placeholder variables to represent the input tensors

    These placeholders are used as inputs by the rest of the model building
    code and will be fed data during training.  Note that when "None" is in a
    placeholder's shape, it's flexible

    Adds following nodes to the computational graph.
    (When None is in a placeholder's shape, it's flexible)

    input_placeholder: Input placeholder tensor of shape
                       (None, num_steps), type tf.int32
    labels_placeholder: Labels placeholder tensor of shape
                        (None, num_steps), type tf.float32
    dropout_placeholder: Dropout value placeholder (scalar),
                         type tf.float32

    Add these placeholders to self as the instance variables
  
      self.input_placeholder
      self.labels_placeholder
      self.dropout_placeholder

    (Don't change the variable names)
    """
    ### YOUR CODE HERE
    #numbers signifying which row of embed to look into is stored in each element
    self.input_placeholder = tf.placeholder(tf.int32,shape = [None,self.config.num_steps])
    #self.labels_placeholder = tf.placeholder(tf.int32,shape = [None,self.config.num_steps]) 
    self.labels_placeholder = tf.placeholder(tf.int32,shape = [None]) 
    self.dropout_placeholder = tf.placeholder(tf.float32)
    ### END YOUR CODE
  
  def add_embedding(self):
    """Add embedding layer.

    Hint: This layer should use the input_placeholder to index into the
          embedding.
    Hint: You might find tf.nn.embedding_lookup useful.
    Hint: You might find tf.split, tf.squeeze useful in constructing tensor inputs
    Hint: Check the last slide from the TensorFlow lecture.
    Hint: Here are the dimensions of the variables you will need to create:

      L: (len(self.vocab), embed_size)

    Returns:
      inputs: List of length num_steps, each of whose elements should be
              a tensor of shape (batch_size, embed_size).
    """
    # The embedding lookup is currently only implemented for the CPU
    with tf.device('/cpu:0'):
      ### YOUR CODE HERE
      L= tf.get_variable('L',[len(self.vocab),self.config.embed_size],initializer = tf.random_uniform_initializer(minval=-1,maxval=1))
      #L = tf.random_uniform(,minval=-1.0,maxval=1.0,name = 'embedding')
      inputs_unlist = tf.nn.embedding_lookup(L,self.input_placeholder)#will project every number of input_placeholder into a vector
      #so dimension of input would be batch_size*numsteps*embed_size
      inputs = tf.split(1,self.config.num_steps,inputs_unlist)
      print "input shape before squeeze",inputs[0].get_shape()
      for i in xrange(len(inputs)):
        inputs[i] = tf.squeeze(inputs[i],[1])
      print "input shape",inputs[0].get_shape()
	  #inputs[i] =tf.squeeze(inputs[i])
      ### END YOUR CODE
      return inputs

  def add_projection(self, rnn_outputs):
    """Adds a projection layer.

    The projection layer transforms the hidden representation to a distribution
    over the 4 categories.

    Hint: Here are the dimensions of the variables you will need to
          create 
          
          U:   (hidden_size, label_size)
          b_2: (len(vocab),)

    Args:
    rnn_outputs: List of length num_steps, each of whose elements should be
                a tensor of shape (batch_size, hidden_size).
    Returns:
      outputs: List of length num_steps, each a tensor of shape
               (batch_size, len(vocab)
    """
    ### YOUR CODE HERE
    with tf.variable_scope("Projection_Layer"):#,initializer = tf.constant_initializer()):#,intializer = tf.random_uniform([len(self.vocab),self.config.embed_size]),-1.0,1.0)
        shapeU = (self.config.hidden_size,self.config.label_size)
        U = tf.get_variable("U",shapeU)
        shapeB = (self.config.label_size)
        b_2 = tf.get_variable("b2",shapeB)
    #only care about the last output of each epoch   
        for i in xrange(self.config.num_steps):    
            outputs = tf.matmul(rnn_outputs[i],U)+b_2
        """        
        for i in xrange(self.config.num_steps):
            outputs[i] = tf.nn.dropout(outputs[i],self.dropout_placeholder) 
        """        
    ### END YOUR CODE
    return outputs

  def add_loss_op(self, output):
    """Adds loss ops to the computational graph.

    Hint: Use tensorflow.python.ops.seq2seq.sequence_loss to implement sequence loss. 

    Args:
      output: A tensor of shape (None, self.vocab)
    Returns:
      loss: A 0-d tensor (scalar)
    """
    ### YOUR CODE HERE
    with tf.variable_scope("Projection_Layer",reuse = True):
        U = tf.get_variable("U")
    with tf.variable_scope('RNN',reuse = True): 
        H = tf.get_variable('H')
        I = tf.get_variable('I')
    loss_reg = tf.nn.l2_loss(U)+tf.nn.l2_loss(H)#+tf.nn.l2_loss(I)    
    loss = tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(output,self.labels_placeholder))+self.config.l2*tf.reduce_sum(loss_reg)
    
    ### END YOUR CODE
    return loss

  def add_training_op(self, loss):
    """Sets up the training Ops.

    Creates an optimizer and applies the gradients to all trainable variables.
    The Op returned by this function is what must be passed to the
    `sess.run()` call to cause the model to train. See 

    https://www.tensorflow.org/versions/r0.7/api_docs/python/train.html#Optimizer

    for more information.

    Hint: Use tf.train.AdamOptimizer for this model.
          Calling optimizer.minimize() will return a train_op object.

    Args:
      loss: Loss tensor, from cross_entropy_loss.
    Returns:
      train_op: The Op for training.
    """
    ### YOUR CODE HERE
    opt = tf.train.AdamOptimizer()
    train_op = opt.minimize(loss)
    return train_op
  
  def __init__(self, config):
    self.config = config
    self.load_data(debug=False)
    self.add_placeholders()
    self.inputs = self.add_embedding()
    self.input_layer = tf.concat(1,self.inputs)
    self.rnn_outputs = self.add_model(self.inputs)
    self.outputs = self.add_projection(self.rnn_outputs)
  
    #we want to check the accuracy of the prediction
    self.predictions = tf.nn.softmax(tf.cast(self.outputs, 'float64'))
    self.one_hot_prediction = tf.cast(tf.argmax(self.predictions, 1),tf.int32)
    correct_prediction = tf.equal(
    self.labels_placeholder, self.one_hot_prediction)
    self.correct_predictions = tf.reduce_sum(tf.cast(correct_prediction, 'int32'))
    # Reshape the output into len(vocab) sized chunks - the -1 says as many as
    # needed to evenly divide
    print "shape of output",self.outputs.get_shape()
    output = tf.reshape(self.outputs, [-1, self.config.label_size])
    self.calculate_loss = self.add_loss_op(output)
    self.train_step = self.add_training_op(self.calculate_loss)
    self.hidden_layer = tf.concat(1,self.rnn_outputs)
    
  def add_model(self, inputs):
    """Creates the RNN LM model.

    In the space provided below, you need to implement the equations for the
    RNN LM model. Note that you may NOT use built in rnn_cell functions from
    tensorflow.

    Hint: Use a zeros tensor of shape (batch_size, hidden_size) as
          initial state for the RNN. Add this to self as instance variable

          self.initial_state
  
          (Don't change variable name)
    Hint: Add the last RNN output to self as instance variable

          self.final_state

          (Don't change variable name)
    Hint: Make sure to apply dropout to the inputs and the outputs.
    Hint: Use a variable scope (e.g. "RNN") to define RNN variables.
    Hint: Perform an explicit for-loop over inputs. You can use
          scope.reuse_variables() to ensure that the weights used at each
          iteration (each time-step) are the same. (Make sure you don't call
          this for iteration 0 though or nothing will be initialized!)
    Hint: Here are the dimensions of the various variables you will need to
          create:
      
          H: (hidden_size, hidden_size) 
          I: (embed_size, hidden_size)
          b_1: (hidden_size,)

    Args:
      inputs: List of length num_steps, each of whose elements should be
              a tensor of shape (batch_size, embed_size).
    Returns:
      outputs: List of length num_steps, each of whose elements should be
               a tensor of shape (batch_size, hidden_size)
    """
    ### YOUR CODE HERE
    shapeOut = (self.config.batch_size,self.config.hidden_size)
    #print self.config.batch_size
    #shapeOut = (1,self.config.hidden_size)
    self.initial_state = tf.zeros(shapeOut)
    #self.initial_state = tf.get_variable('h_initial',shapeOut,tf.float32,tf.constant_initializer(0.0))
    rnn_outputs = [] 
    #z = []
    with tf.variable_scope('RNN') as scope: 
        shapeH = (self.config.hidden_size,self.config.hidden_size)
        H = tf.get_variable('H',shapeH,tf.float32)
        shapeI = (self.config.embed_size,self.config.hidden_size)
        I = tf.get_variable('I',shapeI,tf.float32)
        shapeB1 = (self.config.hidden_size)
        b_1 = tf.get_variable('B1', shapeB1, tf.float32)
        for i in xrange(self.config.num_steps):
            scope.reuse_variables()
            inputs[i] = tf.nn.dropout(inputs[i],self.dropout_placeholder)
            if not i==0:
                z = tf.matmul(rnn_outputs[i-1],H)+tf.matmul(inputs[i],I)+b_1
            else:
                print inputs[0].get_shape()
                z = tf.matmul(self.initial_state,H)+tf.matmul(inputs[i],I)+b_1
            temp = tf.sigmoid(z)
            rnn_outputs.append(temp)
    for i in xrange(len(rnn_outputs)):
            rnn_outputs[i] = tf.nn.dropout(rnn_outputs[i],self.dropout_placeholder)
    
    self.final_state = rnn_outputs[self.config.num_steps-1]        
    
    ### END YOUR CODE
    return rnn_outputs

  #add new judgement of whether to start from prev state by compare the label
  def run_epoch(self, session, data_x, data_y, train_op=None, verbose=100, Epoch = None,isTraining = False):
    config = self.config
    dp = config.dropout
    if not train_op:
      train_op = tf.no_op()
      dp = 1
    total_steps = sum(1 for x in ptb_iterator(data_x, data_y, config.batch_size, config.num_steps,self.vocab))
    total_loss = []
    total_correct_examples = 0
    total_processed_examples = 0
    state = self.initial_state.eval()
    y_prev = np.zeros((self.config.batch_size,))
    for step, (x, y) in enumerate(
      ptb_iterator(data_x, data_y, config.batch_size, config.num_steps,self.vocab)):
      # We need to pass in the initial state and retrieve the final state to give
      # the RNN proper history
      ###keep the state if label unchanged
      #print "yshape",y.shape
      #print "prevyshape",y_prev.shape
      keepstate = (y==y_prev)
      keepstate = keepstate.reshape(self.config.batch_size,1)
      #print "keep state shape",keepstate.shape
      state = keepstate*state
      #print "state shape",state.shape
      y_prev = y
      
      feed = {self.input_placeholder: x,
              self.labels_placeholder: y,
              self.initial_state: state,
              self.dropout_placeholder: dp}
             
      wordvector, hiddenlayer, y_hat, predict_label, loss, state,total_correct, _ = session.run(
          [self.input_layer, self.hidden_layer, self.outputs, self.one_hot_prediction,self.calculate_loss, self.final_state, self.correct_predictions,train_op], feed_dict=feed)
      """
      wordvector, hiddenlayer, y_hat, predict_label, loss, total_correct, _ = session.run([self.input_layer, self.hidden_layer, self.outputs, self.one_hot_prediction,self.calculate_loss,  self.correct_predictions,train_op], feed_dict=feed)
      """
      total_loss.append(loss)

      eos_encode = self.vocab.encode('<eos>')
      x_eight = x[:,-1]
      x_flat = x_eight.reshape(-1)
      y_flat = y.reshape(-1)
      y_pred_flat = predict_label.reshape(-1)
      end_sentence = x_flat==eos_encode
      total_correct_examples+=np.sum(y_flat[end_sentence]==y_pred_flat[end_sentence])
      total_processed_examples += np.sum(end_sentence)
      #total_correct_examples+=total_correct
      #total_processed_examples += len(y)
      if verbose and step % verbose == 0:
          sys.stdout.write('\r{} / {} : loss = {} accuracy = {}'.format(
              step, total_steps, np.mean(total_loss), total_correct_examples/float(total_processed_examples)))
          sys.stdout.flush()
          #print ""
      #if step == self.config.num_steps/2 or 1000 and isTraining:   
      if isTraining and Epoch == 7:
          with open('RNN_train_wordvector.dat','a') as f_handle:
            np.savetxt(f_handle,wordvector,'%.3f')
          with open('RNN_train_hiddenlayer.dat','a') as f_handle:  
            np.savetxt(f_handle,hiddenlayer,'%.3f')
          with open('RNN_train_yhat.dat','a') as f_handle:    
            np.savetxt(f_handle,y_hat,'%.3f')
          with open('RNN_train_label.dat','a') as f_handle:    
            np.savetxt(f_handle,y,'%.3f')
      if not isTraining and Epoch == 7:
          with open('RNN_valid_wordvector.dat','a') as f_handle:
            np.savetxt(f_handle,wordvector,'%.3f')
          with open('RNN_valid_hiddenlayer.dat','a') as f_handle:  
            np.savetxt(f_handle,hiddenlayer,'%.3f')
          with open('RNN_valid_yhat.dat','a') as f_handle:    
            np.savetxt(f_handle,y_hat,'%.3f')
          with open('RNN_valid_label.dat','a') as f_handle:    
            np.savetxt(f_handle,y,'%.3f') 
      if not isTraining and Epoch == None:
          with open('RNN_test_wordvector.dat','a') as f_handle:
            np.savetxt(f_handle,wordvector,'%.3f')
          with open('RNN_test_hiddenlayer.dat','a') as f_handle:  
            np.savetxt(f_handle,hiddenlayer,'%.3f')
          with open('RNN_test_yhat.dat','a') as f_handle:    
            np.savetxt(f_handle,y_hat,'%.3f')
          with open('RNN_test_label.dat','a') as f_handle:    
            np.savetxt(f_handle,y,'%.3f')       
         
      """
      if step == self.config.num_steps/2 and isTraining: 
          print "writing training data at step "+str(step)
          #print "encodedx is "+str(x)
          with open('epoch'+str(Epoch)+'wordvector.dat','w') as f_handle:
            np.savetxt(f_handle,wordvector,'%.3f')
          with open('epoch'+str(Epoch)+'hiddenlayer.dat','w') as f_handle:  
            np.savetxt(f_handle,hiddenlayer,'%.3f')
          with open('epoch'+str(Epoch)+'yhat.dat','w') as f_handle:    
            np.savetxt(f_handle,y_hat,'%.3f')
          with open('epoch'+str(Epoch)+'label.dat','w') as f_handle:    
            np.savetxt(f_handle,y,'%.3f')
      if step == self.config.num_steps/2 and not isTraining and Epoch != None:
          print "writing test data at step "+str(step)
          #print "encodedx is "+str(x)
          with open('test_epoch'+str(Epoch)+'wordvector.dat','w') as f_handle:
            np.savetxt(f_handle,wordvector,'%.3f')
          with open('test_epoch'+str(Epoch)+'hiddenlayer.dat','w') as f_handle:  
            np.savetxt(f_handle,hiddenlayer,'%.3f')
          with open('test_epoch'+str(Epoch)+'yhat.dat','w') as f_handle:    
            np.savetxt(f_handle,y_hat,'%.3f')
          with open('test_epoch'+str(Epoch)+'label.dat','w') as f_handle:    
            np.savetxt(f_handle,y,'%.3f')      
       """     
    if verbose:
      sys.stdout.write('\r')
    #return np.exp(np.mean(total_loss)) return loss instead of this
    return np.mean(total_loss)

def generate_text(session, model, config, starting_text='<eos>',
                  stop_length=100, stop_tokens=None, temp=1.0):
  """Generate text from the model.

  Hint: Create a feed-dictionary and use sess.run() to execute the model. Note
        that you will need to use model.initial_state as a key to feed_dict
  Hint: Fetch model.final_state and model.predictions[-1]. (You set
        model.final_state in add_model() and model.predictions is set in
        __init__)
  Hint: Store the outputs of running the model in local variables state and
        y_pred (used in the pre-implemented parts of this function.)

  Args:
    session: tf.Session() object
    model: Object of type RNNLM_Model
    config: A Config() object
    starting_text: Initial text passed to model.
  Returns:
    output: List of word idxs
  """
  state = model.initial_state.eval()
  # Imagine tokens as a batch size of one, length of len(tokens[0])
  tokens = [model.vocab.encode(word) for word in starting_text.split()]
  categories = []
  for j in xrange(len(tokens)):
        x = np.asarray(tokens[j])
        x = np.reshape(x,(1,1))
        print x.shape
        print state.shape
        feed = {model.input_placeholder: x,
              model.initial_state: state,
              model.dropout_placeholder : config.dropout}
        state = session.run(model.final_state,feed_dict=feed)
        #then start to predict
        x = np.asarray(tokens[-1])
        x = np.reshape(x,(1,1))
        feed = {model.input_placeholder: x,
            model.initial_state: state,
            model.dropout_placeholder : config.dropout} 
        #need to use run([]) to fetch stuff, they are returned as function value        
        state,y_pred = session.run([model.final_state,model.predictions],feed_dict=feed)
        print y_pred
        ### END YOUR CODE
        #category = sample(y_pred[0], temperature=temp)
        category = np.argmax(y_pred[0])
        categories.append(category)
  #return the most popular category
  print categories  
  print Counter(categories).most_common(1)
  output = [ite for ite, it in Counter(categories).most_common(1)]
  return output[0]

def categorize_sentence(session, model, config, *args, **kwargs):
  """Convenice to generate a sentence from the model."""
  return generate_text(session, model, config, *args, stop_tokens=['<eos>'], **kwargs)

def test_RNNLM():
  config = Config()
  gen_config = deepcopy(config)
  gen_config.batch_size = gen_config.num_steps = 1

  # We create the training model and generative model
  with tf.variable_scope('RNNLM') as scope:
    model = RNNLM_Model(config)
    # This instructs gen_model to reuse the same variables as the model above
    scope.reuse_variables()
    gen_model = RNNLM_Model(gen_config)

  init = tf.initialize_all_variables()
  saver = tf.train.Saver()

  with tf.Session() as session:
    best_val_pp = float('inf')
    best_val_epoch = 0
  
    session.run(init)
    for epoch in xrange(config.max_epochs):
      print 'Epoch {}'.format(epoch)
      start = time.time()
      ###
      print '---train---'*5
      train_pp = model.run_epoch(
          session, model.encoded_trainx,model.encoded_trainy,
          train_op=model.train_step,verbose = 10,Epoch = epoch,isTraining = True)
      print ""    
      print '---valid---'*5    
      valid_pp = model.run_epoch(session, model.encoded_validx, model.encoded_validy,verbose = 10,Epoch = epoch)
      print ""
      if valid_pp < best_val_pp:
        best_val_pp = valid_pp
        best_val_epoch = epoch
        saver.save(session, './ptb_rnnlm.weights')
      if epoch - best_val_epoch > config.early_stopping:
        break
      print 'Total time: {}'.format(time.time() - start)
      
    saver.restore(session, 'ptb_rnnlm.weights')
    print "Test----"*5
    test_pp = model.run_epoch(session, model.encoded_testx, model.encoded_testy)
    print ""
    print '=-=' * 5
    print 'Test loss: {}'.format(test_pp)
    print '=-=' * 5
    
    starting_text = 'I hate vegetables'
    while starting_text:
      print categorize_sentence(
          session, gen_model, gen_config, starting_text=starting_text, temp=1.0)
      starting_text = raw_input('> ')

if __name__ == "__main__":
    test_RNNLM()
