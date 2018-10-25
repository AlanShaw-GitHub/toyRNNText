import sys
import json
from dataloader import Loader

import tensorflow as tf


class Model(object):

    def __init__(self, loader):
        self.loader = loader
        self.build_model()


    def build_model(self):
        self._questions = tf.placeholder(tf.int32, [None, self.loader.max_ques_len])
        self.ques_len = tf.placeholder(tf.int32, [None])
        self.tags = tf.placeholder(tf.float32, [None,self.loader.max_tag_num])
        self.embeddings = tf.Variable(self.loader.embeddings,dtype=tf.float32,trainable=True)
        self.questions = tf.nn.embedding_lookup(self.embeddings, self._questions)
        self.is_training = tf.placeholder(tf.bool)
        with tf.variable_scope('model'):
            rnn_cell_fw = tf.nn.rnn_cell.LSTMCell(1024,name='rnn_cell_fw')
            # initial_state_fw = rnn_cell_fw.zero_state(self.loader.batch_size, dtype=tf.float32)
            rnn_cell_bw = tf.nn.rnn_cell.LSTMCell(1024,name='rnn_cell_bw')
            # initial_state_bw = rnn_cell_bw.zero_state(self.loader.batch_size, dtype=tf.float32)
            _, output_states = tf.nn.bidirectional_dynamic_rnn(rnn_cell_fw,rnn_cell_bw,self.questions,
                    self.ques_len, dtype=tf.float32)
            # self.outputs = tf.concat(outputs,axis=-1)
            self.output_states = tf.concat((tf.concat(output_states[0], axis=-1),tf.concat(output_states[1], axis=-1)), axis=-1)
            self.output_states = tf.contrib.layers.dropout(self.output_states, 0.5, is_training=self.is_training)
            # _predict_tags = tf.layers.dense(self.output_states,1024,activation=None)
            self.predict_tags = tf.layers.dense(self.output_states,self.loader.max_tag_num,activation=None)
            epsilon = 0.5
            # weight =  ((1-epsilon) * self.tags) + (epsilon / self.loader.max_tag_num)
            # weight = tf.nn.softmax(self.tags)
            variables = tf.trainable_variables()
            self.regularization_cost = tf.reduce_sum([tf.nn.l2_loss(v) for v in variables])
            loss = tf.square(tf.subtract(self.tags,self.predict_tags))
            # self.predict_tags = tf.nn.softmax(self.predict_tags)
            # loss = tf.nn.softmax_cross_entropy_with_logits(labels=self.tags,logits=self.predict_tags)
            self.loss = tf.reduce_sum(loss)
            self.train_loss = self.loss + 0.00005 * self.regularization_cost


if __name__ == '__main__':
    loader = Loader()
    model = Model()



