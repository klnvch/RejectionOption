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
    
    def __init__(self, learning_rate, layers, activation_function='softmax'):
        print('init...')
        print('learning rate: {:f}, activation function: {:s}'.format(learning_rate, activation_function))
        print('layers: [{:d},{:d},{:d}]'.format(layers[0], layers[1], layers[2]))
        
        self.learning_rate = learning_rate
        self.activation_function = activation_function
        self.num_input = layers[0]
        self.num_hidden = layers[1]
        self.num_output = layers[2]
        
        self.x = tf.placeholder(tf.float32, [None, self.num_input])
        self.y_ = tf.placeholder(tf.float32, [None, self.num_output])

        w_h = tf.Variable(tf.truncated_normal([self.num_input, self.num_hidden], stddev=0.1), name="hidden_weights")
        b_h = tf.Variable(tf.constant(0.1, shape=[self.num_hidden]), name="hidden_biases")
        w_o = tf.Variable(tf.truncated_normal([self.num_hidden, self.num_output], stddev=0.1), name="output_weights")
        b_o = tf.Variable(tf.constant(0.1, shape=[self.num_output]), name="outputs_biases")

        y = tf.matmul(tf.nn.sigmoid(tf.matmul(self.x, w_h) + b_h), w_o) + b_o

        self.cross_entropy = None
        if activation_function == 'softmax':
            self.cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, self.y_))
        elif activation_function == 'sigmoid':
            self.cross_entropy = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(y, self.y_))
        self.train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(self.cross_entropy)
    
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
            counter = [0, 0, 0, 0]

        saver = tf.train.Saver()
        train_time = 0
        with tf.Session() as sess:
            tf.global_variables_initializer().run()
            for step in range(steps):
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
                    loss, trn_acc = self.log_step_info(sess, trn_x, trn_y, vld_x, vld_y, step, train_time, logging)
                    train_time = 0
                    
                    if early_stopping is not None and step > 0:
                        vld_acc = sess.run(self.accuracy, feed_dict={self.x: vld_x, self.y_: vld_y})
                        r_0, c_0, _, _ = self.test_rejection_internal(sess, vld_x, vld_y, None, 0, 100)
                        r_1, c_1, _, _ = self.test_rejection_internal(sess, vld_x, vld_y, None, 1, 100)
                        r_2, c_2, _, _ = self.test_rejection_internal(sess, vld_x, vld_y, None, 2, 100)
                        areas = [vld_acc, np.trapz(c_0, r_0), np.trapz(c_1, r_1), np.trapz(c_2, r_2)]
                        mark=''
                        for i in [0,1,2,3]:
                            if counter[i] is not None:
                                if areas[i] > result[i][5]:
                                    result[i][1:6] = [step, loss, trn_acc, vld_acc, areas[i]]
                                    counter[i] = 0
                                    saver.save(sess, result[i][0], write_meta_graph=False)
                                    mark += '+'
                                else:
                                    counter[i] += 1
                                    if counter[i] > early_stopping:
                                        counter[i] = None
                                    mark += '-'
                            else:
                                mark += '.'
                        self.log_accuracy(step, loss, trn_acc, vld_acc, areas, mark, logging)
                        if counter == [None, None, None, None]:
                            break
        
            if early_stopping is None:
                saver.save(sess, 'saver/model.ckpt', write_meta_graph=False) 
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
            filename = 'tests/test_{:s}_{:f}_{:d}_{:d}_{:d}_{:d}.txt'.format(self.activation_function, self.learning_rate, steps, batch_size, self.num_hidden, int(time.time()))
            self.log_file = open(filename, 'w+')
            print('Step, Loss, Train accuracy,  Validation accuracy, Area under ROC for output threshold, Area under ROC for differential threshold, Area under ROC for ratio threshold, Time', file=self.log_file)
    
        log_msg = 'learing rate: {:f}; steps: {:d}'.format(self.learning_rate, steps)
        if batch_size is not None:
            log_msg += '; batch size: {:d}'.format(batch_size)
        if early_stopping is not None:
            log_msg += '; early_stopping: {:d}'.format(early_stopping)
        print(log_msg)
        
    def log_step_info(self, sess, x_trn, y_trn, x_vld, y_vld, step, train_time, logging):
        loss = sess.run(self.cross_entropy, feed_dict={self.x: x_trn, self.y_: y_trn})
        trn_acc = sess.run(self.accuracy, feed_dict={self.x: x_trn, self.y_: y_trn})
        vld_acc, outputs = sess.run([self.accuracy, self.y_final], feed_dict={self.x: x_vld, self.y_: y_vld})
        #
        predictions = [np.argmax(o) for o in outputs]
        y = [np.argmax(o) for o in y_vld]
        auc = dict()
        y_true = [a==b for a,b in zip(np.array(y), np.array(predictions))]
        for i in [0,1,2]:
            y_score = rejection_score(outputs, i)
            auc[i] = metrics.roc_auc_score(y_true, y_score)
        #
        log_msg = '{:8d}, {:9f}, {:9f}, {:9f}, {:9f}, {:9f}, {:9f}, {:f}'.format(step, loss, trn_acc, vld_acc, auc[0], auc[1], auc[2], train_time)
        if logging: print(log_msg, file=self.log_file)
        print(log_msg)
        #
        return loss, trn_acc
    
            
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