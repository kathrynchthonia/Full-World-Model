# -*- coding: utf-8 -*-
# Building the MDN-RNN Model

# Importing the Libraries
import numpy as np
import tensorflow as tf

# Building the MDN-RNN Model within a class

class MDNRNN(object):
    
    #Initializing all the parameters and the variables of the MDNRNN Class
    def __init__(self, hps,  reuse=False. gpu_mode=False):
        self.hps = hps
        with tf.variable_scope('mdn-rnn', reuse=reuse):
            if not gpu_mode:
                with tf.device('/cpu:0'):
                    tf.logging.info('Model using cpu...')
                    self.g = tf.Graph()
                    with self.g.as_default():
                        self.build_model(hps)
            else:
                tf.logging.info('Model using gpu...')
                self.g = tf.Graph()
                with self.g.as_default():
                        self.build_model(hps)
        self._init_session()
        
        
    # Making a Method that creates the MDN-RNN model architecture itself# -*- coding: utf-8 -*-
    def build_model(self, hps):
        # building the RNN
        self.num_mixture = hps.num_mixture
        KMIX = self.num_mixture
        INWIDTH = hps.input_seq_width
        OUTWIDTH = hps.output_seq_width
        LENGTH = self.hps.max_seq_len
        if hps.is_training:
            self.global_step = tf.Variable(0, name='global_step', trainable=False)
