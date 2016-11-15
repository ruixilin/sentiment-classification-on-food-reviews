import getpass
import sys
import time

import numpy as np
from copy import deepcopy

from utilsGRU_zeropad import calculate_perplexity, get_ptb_dataset, get_ptb_dataset_y, Vocab
from utilsGRU_zeropad import ptb_iterator, sample

import tensorflow as tf
from tensorflow.python.ops.seq2seq import sequence_loss
from model import LanguageModel

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
  label_size = 5
  hidden_size = 100
  num_steps = 8
  max_epochs = 16#16
  early_stopping = 2
  dropout = 1#0.9
  lr = 1e-5#0.001
  l2 = 0.006#0.001
class RNNLM_Model(LanguageModel):

  def load_data(self, debug=False):
    """Loads starter word-vectors and train/dev/test data."""
    self.vocab = Vocab()
    self.vocab.construct(get_ptb_dataset('trainx88_1',0)) #???
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
    self.input_placeholder = tf.placeholder(tf.int32, shape=(None, self.config.num_steps))
    self.labels_placeholder = tf.placeholder(tf.int32, shape=(None,self.config.num_steps))
    self.dropout_placeholder = tf.placeholder(tf.float32)
    #raise NotImplementedError
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
      embeddings = tf.Variable(tf.random_uniform([len(self.vocab), self.config.embed_size], -1.0, 1.0))
      embeds = tf.nn.embedding_lookup(embeddings, self.input_placeholder)
      inputs = tf.split(1, self.config.num_steps, embeds)
      return [tf.squeeze(eachInput,[1]) for eachInput in inputs]
      ### END YOUR CODE
      return inputs

  def add_projection(self, rnn_outputs):
    """Adds a projection layer.

    The projection layer transforms the hidden representation to a distribution
    over the vocabulary.

    Hint: Here are the dimensions of the variables you will need to
          create 
          
          U:   (hidden_size, len(vocab))
          b_2: (len(vocab),)

    Args:
      rnn_outputs: List of length num_steps, each of whose elements should be
                   a tensor of shape (batch_size, hidden_size).
    Returns:
      outputs: List of length num_steps, each a tensor of shape
               (batch_size, len(vocab)
    """
    ### YOUR CODE HERE
    outputs = []
    shapeU = (self.config.hidden_size, self.config.label_size)
    U = tf.get_variable("U",  shapeU)
    b_2 = tf.get_variable("b_2",  self.config.label_size)
    tf.add_to_collection('myCollect1', U)
      
    for i in xrange(self.config.num_steps):
      outputs.append(tf.matmul(rnn_outputs[i], U)+b_2)

    #raise NotImplementedError
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
    U = tf.get_collection('myCollect1')
    print "shape of U", U[0].get_shape()
    with tf.variable_scope('RNN') as scope: 
        scope.reuse_variables()
        H = tf.get_variable('H')
        H_reset = tf.get_variable('H_reset')
        H_update = tf.get_variable('H_update')
    loss_reg = tf.nn.l2_loss(U[0])+tf.nn.l2_loss(H)+tf.nn.l2_loss(H_reset)+tf.nn.l2_loss(H_update)    
    weights = tf.ones([self.config.batch_size*self.config.num_steps])
    loss = sequence_loss([output], [tf.reshape(self.labels_placeholder,[-1])], [weights],average_across_batch=False)+self.config.l2*loss_reg
    
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
    ### END YOUR CODE
    return train_op
  
  def __init__(self, config):
    self.config = config
    self.load_data(debug=False)
    self.add_placeholders()
    self.inputs = self.add_embedding()
    self.rnn_outputs = self.add_model(self.inputs)
    self.outputs = self.add_projection(self.rnn_outputs)
    #output the layers for checking
    self.input_layer = tf.concat(1,self.inputs)
    self.hidden_layer = tf.concat(1,self.rnn_outputs)
    self.output_layer = tf.concat(1,self.outputs)
    # We want to check how well we correctly predict the next word
    # We cast o to float64 as there are numerical issues at hand
    # (i.e. sum(output of softmax) = 1.00000298179 and not 1)
    self.predictions = [tf.nn.softmax(tf.cast(o, 'float64')) for o in self.outputs]
    # Reshape the output into len(vocab) sized chunks - the -1 says as many as
    # needed to evenly divide
    output = tf.reshape(tf.concat(1, self.outputs), [-1, self.config.label_size])
    self.calculate_loss = self.add_loss_op(output)
    self.train_step = self.add_training_op(self.calculate_loss)

    #we want to check the accuracy of the prediction
    #self.one_hot_prediction = tf.cast(tf.argmax(self.predictions, 1),tf.int32)
    print"predictions shape",self.predictions[0].get_shape()
    print "len of prediction",len(self.predictions)
    self.one_hot_prediction = [tf.reshape(tf.argmax(o,1),(self.config.batch_size,1)) for o in self.predictions]
    self.one_hot_predictions = tf.concat(1,self.one_hot_prediction)
    self.correct_prediction = tf.equal(
    self.labels_placeholder, tf.cast(self.one_hot_predictions,'int32'))
    self.correct_predictions = tf.reduce_sum(tf.cast(self.correct_prediction, 'int32'))
    #self.accuracy = tf.cast(self.correct_predictions,'float64)'/float(self.config.batch_size)
    

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

        # define GRU weights and biases
        I_update = tf.get_variable('I_update',shapeI,tf.float32)
        H_update = tf.get_variable('H_update',shapeH,tf.float32)
        I_reset = tf.get_variable('I_reset',shapeI,tf.float32)
        H_reset = tf.get_variable('H_reset',shapeH,tf.float32)
        b1_update = tf.get_variable('B1_update', shapeB1, tf.float32)
        b1_reset = tf.get_variable('B1_reset', shapeB1, tf.float32)
        for i in xrange(self.config.num_steps):
            scope.reuse_variables()
            inputs[i] = tf.nn.dropout(inputs[i],self.dropout_placeholder)
            if not i==0:  
                updateGate = tf.sigmoid(tf.matmul(rnn_outputs[i-1],H_update)+tf.matmul(inputs[i],I_update))#+b1_update
                resetGate = tf.sigmoid(tf.matmul(rnn_outputs[i-1],H_reset)+tf.matmul(inputs[i],I_reset))#+b1_reset
                z = tf.mul(resetGate,tf.matmul(rnn_outputs[i-1],H))+tf.matmul(inputs[i],I)+b_1
                temp = tf.tanh(z)
                temp = tf.mul(updateGate,rnn_outputs[i-1])+tf.mul((1-updateGate),temp)
            else:
                #print inputs[0].get_shape()
                updateGate = tf.zeros((self.config.hidden_size,))#tf.sigmoid(tf.matmul(self.initial_state,H_update)+tf.matmul(inputs[i],I_update))+b1_update
                resetGate = tf.zeros((self.config.hidden_size,))#tf.sigmoid(tf.matmul(self.initial_state,H_reset)+tf.matmul(inputs[i],I_reset))+b1_reset
                z = tf.mul(resetGate,tf.matmul(self.initial_state,H))+tf.matmul(inputs[i],I)+b_1
                temp = tf.tanh(z)
                temp = tf.mul(updateGate,self.initial_state)+tf.mul((1-updateGate),temp)
            rnn_outputs.append(temp)
    for i in xrange(len(rnn_outputs)):
            rnn_outputs[i] = tf.nn.dropout(rnn_outputs[i],self.dropout_placeholder)
    
    self.final_state = rnn_outputs[self.config.num_steps-1] 
    ### END YOUR CODE
    return rnn_outputs


  def run_epoch(self, session, data_x, data_y, train_op=None, verbose=100, Epoch = None,isTraining = False):
    config = self.config
    dp = config.dropout
    if not train_op:
      train_op = tf.no_op()
      dp = 1
    total_steps = sum(1 for x in ptb_iterator(data_x, data_y,config.batch_size, config.num_steps,self.vocab))
    total_loss = []

    total_correct_examples = 0
    total_processed_examples = 0

    state = self.initial_state.eval()
    y_prev = np.zeros((self.config.batch_size,self.config.num_steps))
    for step, (x, y) in enumerate(
      ptb_iterator(data_x, data_y,config.batch_size, config.num_steps,self.vocab)):

      keepstate = (y[:,0]==y_prev[:,-1])
      keepstate = keepstate.reshape(self.config.batch_size,1)
      #state = keepstate*state
      y_prev = y

      # We need to pass in the initial state and retrieve the final state to give
      # the RNN proper history
      feed = {self.input_placeholder: x,
              self.labels_placeholder: y,
              self.initial_state: state,
              self.dropout_placeholder: dp}

      input_layer, hidden_layer, output_layer, loss, state,y_pred, _ = session.run(
          [self.input_layer, self.hidden_layer, self.output_layer, self.calculate_loss, self.final_state, self.one_hot_predictions,train_op], feed_dict=feed)

      total_loss.append(loss)
      #now try to calculate accuracy by sentence
      """
      total_correct_examples+=total_correct
      total_processed_examples += len(y)
      """
      eos_encode = self.vocab.encode('<eos>')
      x_flat = x.reshape(-1)
      y_flat = y.reshape(-1)
      y_pred_flat = y_pred.reshape(-1)
      end_sentence = x_flat==eos_encode
      total_processed_examples += np.sum(end_sentence)
      total_correct_examples += np.sum((y_flat[end_sentence==True] == y_pred_flat[end_sentence==True]))
      if(total_processed_examples==0): accuracy = 0;
      else: accuracy = float(total_correct_examples)/total_processed_examples
      if verbose and step % verbose == 0:
          """
          print y_pred_flat
          """  
          sys.stdout.write('\r{} / {} : loss = {}, accuracy = {}, correct{}/{}'.format(
              step, total_steps, np.mean(total_loss), accuracy, total_correct_examples, total_processed_examples))
          sys.stdout.flush()
      """      
      if step == self.config.num_steps/2 and isTraining: 
          print "writing training data at step "+str(step)
          with open('epoch'+str(Epoch)+'wordvector_GRU.dat','w') as f_handle:
            np.savetxt(f_handle,input_layer,'%.3f')
          with open('epoch'+str(Epoch)+'hiddenlayer_GRU.dat','w') as f_handle:  
            np.savetxt(f_handle,hidden_layer,'%.3f')
          with open('epoch'+str(Epoch)+'yhat_GRU.dat','w') as f_handle:    
            np.savetxt(f_handle,output_layer,'%.3f')
          with open('epoch'+str(Epoch)+'label_GRU.dat','w') as f_handle:    
            np.savetxt(f_handle,y,'%.3f')
      if step == self.config.num_steps/2 and not isTraining and Epoch != None:
          print "writing test data at step "+str(step)
          #print "encodedx is "+str(x)
          with open('test_epoch'+str(Epoch)+'wordvector_GRU.dat','w') as f_handle:
            np.savetxt(f_handle,input_layer,'%.3f')
          with open('test_epoch'+str(Epoch)+'hiddenlayer_GRU.dat','w') as f_handle:  
            np.savetxt(f_handle,hidden_layer,'%.3f')
          with open('test_epoch'+str(Epoch)+'yhat_GRU.dat','w') as f_handle:    
            np.savetxt(f_handle,output_layer,'%.3f')
          with open('test_epoch'+str(Epoch)+'label_GRU.dat','w') as f_handle:    
            np.savetxt(f_handle,y,'%.3f')  
      """
      if isTraining and Epoch == 7:
          with open('RNN_train_yhat_GRU.dat','a') as f_handle:    
            np.savetxt(f_handle,y_pred_flat[end_sentence==True].reshape((len(y_pred_flat[end_sentence]),1)),'%.3f')
          with open('RNN_train_label_GRU.dat','a') as f_handle:    
            np.savetxt(f_handle,y_flat[end_sentence==True].reshape((len(y_flat[end_sentence]),1)),'%.3f')
      if not isTraining and Epoch == 7:
          with open('RNN_valid_yhat_GRU.dat','a') as f_handle:    
            np.savetxt(f_handle,y_pred_flat[end_sentence==True].reshape((len(y_pred_flat[end_sentence]),1)),'%.3f')
          with open('RNN_valid_label_GRU.dat','a') as f_handle:    
            np.savetxt(f_handle,y_flat[end_sentence==True].reshape((len(y_flat[end_sentence]),1)),'%.3f') 
      if not isTraining and Epoch == None:
          with open('RNN_test_yhat_GRU.dat','a') as f_handle:    
            np.savetxt(f_handle,y_pred_flat[end_sentence==True].reshape((len(y_pred_flat[end_sentence]),1)),'%.3f')
          with open('RNN_test_label_GRU.dat','a') as f_handle:    
            np.savetxt(f_handle,y_flat[end_sentence].reshape((len(y_flat[end_sentence]),1)),'%.3f')           
    if verbose:
      sys.stdout.write('\r')      
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
        """
        print x.shape
        print state.shape
      
        feed = {model.input_placeholder: x,
              model.initial_state: state,
              model.dropout_placeholder : config.dropout}
        state = session.run(model.final_state,feed_dict=feed)
        """
        #then start to predict
        x = np.asarray(tokens[-1])
        x = np.reshape(x,(1,1))
        feed = {model.input_placeholder: x,
            model.initial_state: state,
            model.dropout_placeholder : config.dropout} 
        #need to use run([]) to fetch stuff, they are returned as function value        
        state,y_pred = session.run([model.final_state, model.one_hot_predictions],feed_dict=feed)
        print y_pred
        ### END YOUR CODE
        #category = sample(y_pred[0], temperature=temp)
        #category = np.argmax(y_pred[0])
        categories.append(y_pred)
  #return the most popular category
  print categories  
  return categories[-1]

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
      print 'Training loss: {}'.format(train_pp)
      print 'Validation loss: {}'.format(valid_pp)
      if valid_pp < best_val_pp:
        best_val_pp = valid_pp
        best_val_epoch = epoch
        saver.save(session, './ptb_rnnlm.weights')
      if epoch - best_val_epoch > config.early_stopping:
        break
      print 'Total time: {}'.format(time.time() - start)
    saver.restore(session, 'ptb_rnnlm.weights')
    print '---test---'*5
    test_pp = model.run_epoch(session, model.encoded_testx, model.encoded_testy)
    print ""
    print '=-=' * 5
    print 'Test Loss: {}'.format(test_pp)
    print '=-=' * 5
    starting_text = 'This is my favorite biscuit'
    print starting_text
    while starting_text:
      print categorize_sentence(
          session, gen_model, gen_config, starting_text=starting_text, temp=1.0)
      starting_text = raw_input('> ')

if __name__ == "__main__":
    test_RNNLM()
