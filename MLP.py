'''
Created on Oct 26, 2016

@author: anton

This module keeps all code for Tensorflow to implement MLP
'''

import time, os
import tensorflow as tf
from sklearn.utils import shuffle

class MLP:
    
    def __init__(self, learning_rate, layers, last_layer='softmax',
                 optimizer_name=None, beta=0.0):
        print('create neural network...')
        log_msg = 'learning rate: {:g}, function: {:s}, optimizer: {:s}'
        log_msg = log_msg.format(learning_rate, last_layer, optimizer_name)
        print(log_msg)
        print('layers: {:s}'.format(str(layers)))
        
        self.learning_rate = learning_rate
        self.optimizer_name = optimizer_name
        self.activation_function = last_layer
        self.num_input = layers[0]
        self.num_hidden = layers[1:-1]
        self.num_output = layers[-1]
        
        self.x = tf.placeholder(tf.float32, [None, self.num_input])
        self.y_ = tf.placeholder(tf.float32, [None, self.num_output])
        self.keep_prob = tf.placeholder(tf.float32)
        
        if len(self.num_hidden) == 0:
            y, r1 = self.add_layer(self.x,
                                   [self.num_input, self.num_output],
                                   None, '', self.keep_prob)
            regularizers = r1
            
        elif len(self.num_hidden) == 1:
            h, r1 = self.add_layer(self.x,
                                   [self.num_input, self.num_hidden[0]],
                                   tf.nn.sigmoid, '1', self.keep_prob)
            y, r2 = self.add_layer(h,
                                   [self.num_hidden[0], self.num_output],
                                   None, '2', self.keep_prob)
            regularizers = r1 + r2
            
        elif len(self.num_hidden) == 2:
            h1, r1 = self.add_layer(self.x,
                                    [self.num_input, self.num_hidden[0]],
                                    tf.nn.relu, '1', self.keep_prob)
            h2, r2 = self.add_layer(h1,
                                    [self.num_hidden[0], self.num_hidden[1]],
                                    tf.nn.relu, '2', self.keep_prob)
            y, r3 = self.add_layer(h2,
                                    [self.num_hidden[1], self.num_output],
                                    None, '3', self.keep_prob)
            regularizers = r1 + r2 + r3
            
        elif len(self.num_hidden) == 3:
            h1, r1 = self.add_layer(self.x,
                                    [self.num_input, self.num_hidden[0]],
                                    tf.nn.relu, '1', self.keep_prob)
            h2, r2 = self.add_layer(h1,
                                    [self.num_hidden[0], self.num_hidden[1]],
                                    tf.nn.relu, '2', self.keep_prob)
            h3, r3 = self.add_layer(h2,
                                    [self.num_hidden[1], self.num_hidden[2]],
                                    tf.nn.relu, '3', self.keep_prob)
            y, r4 = self.add_layer(h3,
                                    [self.num_hidden[2], self.num_output],
                                    None, '4', self.keep_prob)
            regularizers = r1 + r2 + r3 + r4
            
        if last_layer == 'softmax':
            self.loss = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(labels=self.y_,
                                                        logits=y))
        elif last_layer == 'sigmoid':
            self.loss = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(labels=self.y_,
                                                        logits=y))
        
        self.loss = tf.reduce_mean(self.loss + beta * regularizers)
        
        if optimizer_name == 'GradientDescent':
            optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        elif optimizer_name == 'Adadelta':
            optimizer = tf.train.AdadeltaOptimizer(learning_rate)
        elif optimizer_name == 'Adagrad':
            optimizer = tf.train.AdagradOptimizer(learning_rate)
        elif optimizer_name == 'Momentum':
            optimizer = tf.train.MomentumOptimizer(learning_rate, 0.9,
                                                   use_nesterov=True) 
        elif optimizer_name == 'Adam':
            optimizer = tf.train.AdamOptimizer(learning_rate)
        elif optimizer_name == 'Ftrl':
            optimizer = tf.train.FtrlOptimizer(learning_rate)
        elif optimizer_name == 'RMSProp':
            optimizer = tf.train.RMSPropOptimizer(learning_rate)
        
        self.train_step = optimizer.minimize(self.loss)
        
        # grads_and_vars = optimizer.compute_gradients(self.loss)
        # grad_norms = [tf.nn.l2_loss(g) for g, _ in grads_and_vars]
        # self.grad_norm = tf.add_n(grad_norms)
        
        self.y_final = None
        if   last_layer == 'softmax': self.y_final = tf.nn.softmax(y)
        elif last_layer == 'sigmoid': self.y_final = tf.nn.sigmoid(y)
        else: raise ValueError('wrong function  value: ' + last_layer)
        
        y_true = tf.equal(tf.argmax(self.y_final, 1), tf.argmax(self.y_, 1))
        self.accuracy = tf.reduce_mean(tf.cast(y_true, tf.float32))
    
    def add_layer(self, x, shape, activation_function, postfix_name, keep_prob):
        w = tf.Variable(tf.truncated_normal(shape, stddev=0.1),
                        name='weights_' + postfix_name)
        b = tf.Variable(tf.constant(0.1, shape=[shape[1]]),
                        name='biases_' + postfix_name)
        l2_loss = tf.nn.l2_loss(w)
        if activation_function is None:
            return tf.matmul(x, w) + b, l2_loss
        else:
            layer = activation_function(tf.matmul(x, w) + b)
            layer = tf.nn.dropout(layer, keep_prob)
            return layer, l2_loss
    
    def train(self, steps, trn, vld=None, batch_size=None, keep_prob=1.0,
              log=True):
        """
        if early stopping not None output is
            [model_file, step, loss, trn_acc, vld_acc, area] for best_vld_acc,
             best_area0, best_area1 and best_area2
        """
        self.clean_model_dir()
        self.log_init(steps, batch_size, log)
        
        saver = tf.train.Saver()
        save_path = 'saver/model.ckpt'
        dt = 0
        counter = 0
        best_vld_acc = 0
        step = 0
        with tf.Session() as sess:
            tf.global_variables_initializer().run()
            for step in range(steps + 1):
                # train 
                start_time = time.time()
                x, y = shuffle(trn.x, trn.y)
                if batch_size is None:
                    sess.run(self.train_step,
                             feed_dict={self.x: x, self.y_: y,
                                        self.keep_prob: keep_prob})
                else:
                    for i in range(0, trn.x.shape[0], batch_size):
                        sess.run(self.train_step,
                                 feed_dict={self.x: x[i:i + batch_size],
                                            self.y_: y[i:i + batch_size],
                                            self.keep_prob: keep_prob})
                finish_time = time.time()
                dt += (finish_time - start_time)
                if step % 100 == 0:
                    loss, trn_acc, vld_acc = \
                        self.log_step_info(sess, trn, vld, step, dt, log)
                    dt = 0
                    # early stopping
                    if vld is not None:
                        if vld_acc > best_vld_acc:
                            best_vld_acc = vld_acc
                            counter = 0
                            saver.save(sess, save_path)
                            info = [save_path, step, loss, trn_acc, vld_acc]
                        else:
                            counter += 1
                        if counter >= 10:
                            break
            
            if vld is None:
                info = [save_path, step, loss, trn_acc, vld_acc]
                saver.save(sess, save_path)
        
        self.log_finish()
        
        return info
    
    def score(self, tst, filename='saver/model.ckpt'):
        saver = tf.train.Saver()
        with tf.Session() as sess:
            saver.restore(sess, filename)
            return sess.run(self.accuracy,
                            feed_dict={self.x: tst.x,
                                       self.y_: tst.y,
                                       self.keep_prob: 1.0})
    
    def predict_proba(self, x, filename='saver/model.ckpt'):
        if x is None: return None
        saver = tf.train.Saver()
        with tf.Session() as sess:
            saver.restore(sess, filename)
            return sess.run(self.y_final, feed_dict={self.x: x,
                                                     self.keep_prob: 1.0})
    
    def predict(self, x, filename='saver/model.ckpt'):
        saver = tf.train.Saver()
        with tf.Session() as sess:
            saver.restore(sess, filename)
            return sess.run(self.y_final, feed_dict={self.x: x})
    
    def clean_model_dir(self):
        dirPath = 'saver/'
        fileList = os.listdir(dirPath)
        for fileName in fileList:
            os.remove(dirPath + '/' + fileName)
    
    def log_init(self, steps, batch_size, logging):
        self.log_file = None
        if logging:
            filename = 'tests/{:s}_{:s}_{:g}_{:d}_{:d}_{:s}_{:d}.csv'
            filename.format(self.activation_function, self.optimizer_name,
                            self.learning_rate, steps, batch_size,
                            str(self.num_hidden), int(time.time()))
            self.log_file = open(filename, 'w+')
            print('Step,Loss,Train accuracy,Validation accuracy,'\
                  'ROC AUC: output threshold,ROC AUC: differential threshold,'\
                  'ROC AUC: ratio threshold,Time', file=self.log_file)
        
        log_msg = 'learing rate: {:f}; steps: {:d}'
        log_msg = log_msg.format(self.learning_rate, steps)
        if batch_size is not None:
            log_msg += '; batch size: {:d}'.format(batch_size)
        print(log_msg)
    
    def log_step_info(self, sess, trn, vld, step, train_time, logging):
        loss, trn_acc = sess.run([self.loss, self.accuracy],
                                 feed_dict={self.x: trn.x,
                                            self.y_: trn.y,
                                            self.keep_prob: 1.0})
        if vld is not None:
            vld_acc = sess.run(self.accuracy,
                               feed_dict={self.x: vld.x,
                                          self.y_: vld.y,
                                          self.keep_prob: 1.0})
        else: vld_acc = 0
        
        log_msg = '{:8d}, {:9f}, {:9f}, {:9f}, {:f}'
        log_msg = log_msg.format(step, loss, trn_acc, vld_acc, train_time)
        if logging: print(log_msg, file=self.log_file)
        print(log_msg)
        
        return loss, trn_acc, vld_acc
    
    def log_finish(self):
        print('train finished')