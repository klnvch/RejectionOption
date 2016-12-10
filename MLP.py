'''
Created on Oct 26, 2016

@author: anton
'''

import time, os
import numpy as np
import tensorflow as tf
from sklearn.utils import shuffle

class MLP:
    
    def __init__(self, learning_rate, num_input, num_hidden, num_output, activation_function='softmax'):
        print("learning rate: %g, layers: [%d, %d, %d], activation_function: %s" % (learning_rate, num_input, num_hidden, num_output, activation_function))
        
        self.learning_rate = learning_rate
        
        if num_hidden == -1: num_hidden = int((num_input + num_output) * 0.666)
        
        self.x = tf.placeholder(tf.float32, [None, num_input])
        self.y_ = tf.placeholder(tf.float32, [None, num_output])

        w_h = tf.Variable(tf.truncated_normal([num_input, num_hidden], stddev=0.1), name="hidden_weights")
        b_h = tf.Variable(tf.constant(0.1, shape=[num_hidden]), name="hidden_biases")
        w_o = tf.Variable(tf.truncated_normal([num_hidden, num_output], stddev=0.1), name="output_weights")
        b_o = tf.Variable(tf.constant(0.1, shape=[num_output]), name="outputs_biases")

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
        
        
    def train(self, steps, trn_x, trn_y, vld_x, vld_y, early_stopping=None, logging = True):
        """
        if early stopping not None output is
            [model_file, step, loss, trn_acc, vld_acc, area] for best_vld_acc, best_area0, best_area1 and best_area2
        """
        self.clean_model_dir()
        self.log_init(steps, logging, early_stopping)
        
        #       [model_file, step, loss, trn_acc, vld_acc, area]
        if early_stopping is not None:
            result=[['saver/model_best_vld_acc.ckpt', 0, 0, 0, 0, 0],
                    ['saver/model_best_area_0.ckpt',  0, 0, 0, 0, 0],
                    ['saver/model_best_area_1.ckpt',  0, 0, 0, 0, 0],
                    ['saver/model_best_area_2.ckpt',  0, 0, 0, 0, 0]]
            counter = [0, 0, 0, 0]

        saver = tf.train.Saver()
        with tf.Session() as sess:
            tf.initialize_all_variables().run()

            for step in range(steps):
                batch_x, batch_y = shuffle(trn_x, trn_y)
                _, loss = sess.run([self.train_step, self.cross_entropy], feed_dict={self.x: batch_x, self.y_: batch_y})
                if step%1000 == 0:
                    trn_acc = sess.run(self.accuracy, feed_dict={self.x: trn_x, self.y_: trn_y})
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
                    else:
                        self.log_accuracy(step, loss, trn_acc, logging=logging)
        
            if early_stopping is None:
                saver.save(sess, 'saver/model.ckpt', write_meta_graph=False) 
        self.log_finish(logging)
    
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
    
    def log_init(self, steps, logging, early_stopping):
        
        self.log_file = None
        if logging: 
            self.log_file = open("tests/test_%g_%d_%s_%d.txt"%(self.learning_rate, steps, self.dataset_name, int(time.time())), "w+")
    
        log_msg = 'learing rate: {:f}; steps: {:d}'.format(self.learning_rate, steps)
        if early_stopping is not None:
            log_msg += '; early_stopping: {:d}'.format(early_stopping)
        if logging: print(log_msg, file=self.log_file)
        print(log_msg)
            
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
        
    
    def log_finish(self, logging):
    
        log_msg = "train finished"
        if logging: print(log_msg, file=self.log_file)
        else:       print(log_msg)
    
    
    ###########