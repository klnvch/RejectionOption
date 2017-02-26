'''
Created on Oct 26, 2016

@author: anton
'''

import time, os
import numpy as np
import tensorflow as tf
from sklearn.utils import shuffle
from sklearn import metrics
from data_utils import rejection_score
from tensorflow.examples.tutorials.mnist.mnist import loss

class MLP:
    
    def __init__(self, learning_rate, layers, activation_function='softmax', optimizer_name=None, regularization_penalty=0.0):
        print('init...')
        print('learning rate: {:g}, activation function: {:s}, optimizer: {:s}'.format(learning_rate, activation_function, optimizer_name))
        print('layers: {:s}'.format(str(layers)))
        
        self.learning_rate = learning_rate
        self.optimizer_name = optimizer_name
        self.activation_function = activation_function
        self.num_input = layers[0]
        self.num_hidden = layers[1:-1]
        self.num_output = layers[-1]
        
        self.x = tf.placeholder(tf.float32, [None, self.num_input])
        self.y_ = tf.placeholder(tf.float32, [None, self.num_output])

        if len(self.num_hidden) == 1:
            w_h = tf.Variable(tf.truncated_normal([self.num_input, self.num_hidden[0]], stddev=0.1), name="hidden_weights")
            b_h = tf.Variable(tf.constant(0.1, shape=[self.num_hidden[0]]), name="hidden_biases")
            
            w_o = tf.Variable(tf.truncated_normal([self.num_hidden[0], self.num_output], stddev=0.1), name="output_weights")
            b_o = tf.Variable(tf.constant(0.1, shape=[self.num_output]), name="outputs_biases")
            
            y = tf.matmul(tf.nn.sigmoid(tf.matmul(self.x, w_h) + b_h), w_o) + b_o
            
            if regularization_penalty > 0:
                regularizer = regularization_penalty * (tf.nn.l2_loss(w_h) + tf.nn.l2_loss(b_h) + tf.nn.l2_loss(w_o) + tf.nn.l2_loss(b_o))
            else:
                regularizer = 0
            
        elif len(self.num_hidden) == 2:
            w_h1 = tf.Variable(tf.truncated_normal([self.num_input, self.num_hidden[0]], stddev=0.1), name="hidden_1_weights")
            b_h1 = tf.Variable(tf.constant(0.1, shape=[self.num_hidden[0]]), name="hidden_1_biases")
            
            w_h2 = tf.Variable(tf.truncated_normal([self.num_hidden[0], self.num_hidden[1]], stddev=0.1), name="hidden_2_weights")
            b_h2 = tf.Variable(tf.constant(0.1, shape=[self.num_hidden[1]]), name="hidden_2_biases")
            
            w_o = tf.Variable(tf.truncated_normal([self.num_hidden[1], self.num_output], stddev=0.1), name="output_weights")
            b_o = tf.Variable(tf.constant(0.1, shape=[self.num_output]), name="outputs_biases")
            
            y = tf.matmul(tf.nn.sigmoid(tf.matmul(tf.nn.sigmoid(tf.matmul(self.x, w_h1) + b_h1), w_h2) + b_h2), w_o) + b_o
            
            if regularization_penalty > 0:
                regularizer = regularization_penalty * (tf.nn.l2_loss(w_h1) + tf.nn.l2_loss(b_h1) + tf.nn.l2_loss(w_h2) + tf.nn.l2_loss(b_h2) + tf.nn.l2_loss(w_o) + tf.nn.l2_loss(b_o))
            else:
                regularizer = 0

        if activation_function == 'softmax':
            self.cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, self.y_)) + regularizer
        elif activation_function == 'sigmoid':
            self.cross_entropy = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.y_, logits=y))
            
        if optimizer_name == 'GradientDescent':
            optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        elif optimizer_name == 'Adadelta':
            optimizer = tf.train.AdadeltaOptimizer(learning_rate)
        elif optimizer_name == 'Adagrad':
            optimizer = tf.train.AdagradOptimizer(learning_rate)
        elif optimizer_name == 'Momentum':
            optimizer = tf.train.MomentumOptimizer(learning_rate, 0.9, use_nesterov=True) 
        elif optimizer_name == 'Adam':
            optimizer = tf.train.AdamOptimizer(learning_rate)
        elif optimizer_name == 'Ftrl':
            optimizer = tf.train.FtrlOptimizer(learning_rate)
        elif optimizer_name == 'RMSProp':
            optimizer = tf.train.RMSPropOptimizer(learning_rate)
            
        self.train_step = optimizer.minimize(self.cross_entropy)
        
        #grads_and_vars = optimizer.compute_gradients(self.cross_entropy)
        #grad_norms = [tf.nn.l2_loss(g) for g, _ in grads_and_vars]
        #self.grad_norm = tf.add_n(grad_norms)
    
        self.y_final = None
        if activation_function == 'softmax':
            self.y_final = tf.nn.softmax(y)
        elif activation_function == 'sigmoid':
            self.y_final = tf.nn.sigmoid(y)
        self.correct_prediction = tf.equal(tf.argmax(self.y_final, 1), tf.argmax(self.y_, 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))
        
        
    def train(self, steps, trn_x, trn_y, vld_x, vld_y, batch_size=None, early_stopping=None, logging=True):
        """
        if early stopping not None output is
            [model_file, step, loss, trn_acc, vld_acc, area] for best_vld_acc, best_area0, best_area1 and best_area2
        """
        self.clean_model_dir()
        self.log_init(steps, batch_size, early_stopping, logging)
        
        #       [model_file, step, loss, trn_acc, vld_acc, area]
        if early_stopping is not None:
            result=[['saver/model_best_vld_acc.ckpt', 0, 0, 0, 0, 0],
                    ['saver/model_best_area_0.ckpt',  0, 0, 0, 0, 0],
                    ['saver/model_best_area_1.ckpt',  0, 0, 0, 0, 0],
                    ['saver/model_best_area_2.ckpt',  0, 0, 0, 0, 0]]


        saver = tf.train.Saver()
        train_time = 0
        counter = 0
        best_vld_acc = 0
        with tf.Session() as sess:
            tf.global_variables_initializer().run()
            for step in range(steps+1):
                # train 
                start_time = time.time()
                x, y = shuffle(trn_x, trn_y)
                if batch_size is None:
                    sess.run(self.train_step, feed_dict={self.x: x, self.y_: y})
                else:
                    for i in range(0, trn_x.shape[0], batch_size):
                        sess.run(self.train_step, feed_dict={self.x: x[i:i+batch_size], self.y_: y[i:i+batch_size]})
                finish_time = time.time()
                train_time += (finish_time - start_time)
                if step%10 == 0:
                    loss, trn_acc, vld_acc = self.log_step_info(sess, trn_x, trn_y, vld_x, vld_y, step, train_time, logging)
                    train_time = 0
                    if vld_acc > best_vld_acc:
                        best_vld_acc = vld_acc
                        counter = 0
                    else:
                        counter += 1
                    if counter >= 10:
                        break
                        
                    
            saver.save(sess, 'saver/model.ckpt')
        
        self.log_finish()
    
        if early_stopping is None:
            return ['saver/model.ckpt', steps, loss, trn_acc]
        else:
            return result
    
    def test(self, x, y, filename='saver/model.ckpt'):
        saver = tf.train.Saver()
        with tf.Session() as sess:
            saver.restore(sess, filename)
            return sess.run(self.accuracy, feed_dict={self.x: x, self.y_: y})
        
    def predict_proba(self, x, filename='saver/model.ckpt'):
        saver = tf.train.Saver()
        with tf.Session() as sess:
            saver.restore(sess, filename)
            return sess.run(self.y_final, feed_dict={self.x: x})
        
            
    def clean_model_dir(self):
        dirPath = "saver/"
        fileList = os.listdir(dirPath)
        for fileName in fileList:
            os.remove(dirPath+"/"+fileName)
    
    def log_init(self, steps, batch_size, early_stopping, logging):
        
        self.log_file = None
        if logging:
            filename = 'tests/{:s}_{:s}_{:g}_{:d}_{:d}_{:s}_{:d}.csv'.format(self.activation_function, self.optimizer_name, self.learning_rate, steps, batch_size, str(self.num_hidden), int(time.time()))
            self.log_file = open(filename, 'w+')
            print('Step,Loss,Train accuracy,Validation accuracy,ROC AUC: output threshold,ROC AUC: differential threshold,ROC AUC: ratio threshold,Time', file=self.log_file)
    
        log_msg = 'learing rate: {:f}; steps: {:d}'.format(self.learning_rate, steps)
        if batch_size is not None:
            log_msg += '; batch size: {:d}'.format(batch_size)
        if early_stopping is not None:
            log_msg += '; early_stopping: {:d}'.format(early_stopping)
        print(log_msg)
        
    def log_step_info(self, sess, x_trn, y_trn, x_vld, y_vld, step, train_time, logging):
        loss, trn_acc = sess.run([self.cross_entropy, self.accuracy], feed_dict={self.x: x_trn, self.y_: y_trn})
        vld_acc, outputs = sess.run([self.accuracy, self.y_final], feed_dict={self.x: x_vld, self.y_: y_vld})
        #
        predictions = np.array([np.argmax(o) for o in outputs])
        y_ideal = np.array([np.argmax(o) for o in y_vld])
        auc = dict()
        y_true = [a==b for a, b in zip(y_ideal, predictions)]
        for i in [0,1,2]:
            y_score = rejection_score(outputs, i)
            auc[i] = metrics.roc_auc_score(y_true, y_score)
        #
        log_msg = '{:8d}, {:9f}, {:9f}, {:9f}, {:9f}, {:9f}, {:9f}, {:f}'.format(step, loss, trn_acc, vld_acc, auc[0], auc[1], auc[2], train_time)
        if logging: print(log_msg, file=self.log_file)
        print(log_msg)
        #
        return loss, trn_acc, vld_acc
    
            
    def log_accuracy(self, i, loss, trn_acc, vld_acc=None, areas=None, mark=None, logging=True):
        
        if vld_acc != 0:
            log_msg = 'step{:8d}, loss:{:9f}, trn_acc:{:9f}'.format(i, loss, trn_acc)
            if vld_acc is not None:
                log_msg += ', vld_acc{:9f}'.format(vld_acc)
            if areas is not None:
                log_msg += ', r0:{:12f}, r1:{:12f}, r2:{:12f}'.format(areas[1], areas[2], areas[3])
            if mark is not None:
                log_msg += ', {:s}'.format(mark)
            if logging: print(log_msg, file=self.log_file)
            print(log_msg)
        else:
            log_msg = "step %d; loss %g; train accuracy: %g"%(i, loss, trn_acc)
            if logging: print(log_msg, file=self.log_file)
            else:       print(log_msg)
            
    def log_results(self, threshold, c, e , r_c, r_e, logging):
        
        log_msg = "threshold %g; correct: %d; errors: %d; reject correct: %d; reject error: %d"%(threshold, len(c), len(e), len(r_c), len(r_e))
        if logging: print(log_msg, file=self.log_file)
        else:       print(log_msg)
        
    
    def log_finish(self):
        print('train finished')