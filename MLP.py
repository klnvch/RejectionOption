'''
Created on Oct 26, 2016

@author: anton
'''

import time, os
import numpy as np
import tensorflow as tf
from DataUtils import shuffle

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
        self.log_init(steps, logging)
        
        trn_acc = 0
        vld_acc = 0
        loss = 0
        #       [model_file, step, loss, trn_acc, vld_acc, area]
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
                if step%100 == 0:
                    trn_acc = sess.run(self.accuracy, feed_dict={self.x: trn_x, self.y_: trn_y})
                    vld_acc = sess.run(self.accuracy, feed_dict={self.x: vld_x, self.y_: vld_y})
                    r_0, c_0, _, _ = self.test_rejection_internal(sess, vld_x, vld_y, outliers=None, threshold_method=0, rejection_rate_limit=50)
                    r_1, c_1, _, _ = self.test_rejection_internal(sess, vld_x, vld_y, outliers=None, threshold_method=1, rejection_rate_limit=50)
                    r_2, c_2, _, _ = self.test_rejection_internal(sess, vld_x, vld_y, outliers=None, threshold_method=2, rejection_rate_limit=50)
                    areas = [vld_acc, np.trapz(c_0, r_0), np.trapz(c_1, r_1), np.trapz(c_2, r_2)]
                    mark = ''
                    if early_stopping is not None and step > 5000:
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
                        if counter == [None, None, None, None]:
                            break
                                    
                    self.log_accuracy(step, loss, trn_acc, vld_acc, areas, mark, logging)
        
            if early_stopping is None:
                saver.save(sess, 'saver/model.ckpt', write_meta_graph=False) 
        self.log_finish(logging)
    
        if early_stopping is None:
            return trn_acc, 0, loss
        else:
            return result
    
    def test(self, x, y, filename='saver/model.ckpt', logging = True):
        saver = tf.train.Saver()
        with tf.Session() as sess:
            saver.restore(sess, filename)
            return sess.run(self.accuracy, feed_dict={self.x: x, self.y_: y})
        
        
    def test_cer(self, x, y, outliers=None, threshold=0.0, threshold_method = 0, logging = True):
        
        saver = tf.train.Saver()
        with tf.Session() as sess:
            saver.restore(sess, "saver/model.ckpt")
            
            predictions, outputs = sess.run([self.correct_prediction, self.y_final], feed_dict={self.x:x, self.y_: y})
            
            outliers_outputs=None
            if outliers is not None:
                outliers_outputs = sess.run(self.y_final, feed_dict={self.x: outliers})
            
            c, e, r_c, r_e = self.test_cer_internal(x, predictions, outputs, outliers, outliers_outputs, threshold, threshold_method, logging)
        
        return c, e, r_c, r_e
    
    """
    Returns data for two graphics rejected vs correct and rejected errors vs rejected correct
    """
    def test_rejection(self, x, y, outliers=None, threshold_method=0, rejection_rate_limit=100, filename='saver/model.ckpt'):
        saver = tf.train.Saver()
        with tf.Session() as sess:
            saver.restore(sess, filename)
            return self.test_rejection_internal(sess, x, y, outliers, threshold_method, rejection_rate_limit)
    
    """
    Returns data for two graphics rejected vs correct and rejected errors vs rejected correct
    """
    def test_rejection_internal(self, sess, data, labels, outliers=None, threshold_method=0, rejection_rate_limit=100):
        rejected = []
        correct = []
        
        rejected_errors = []
        rejected_correct = []
            
        predictions, outputs = sess.run([self.correct_prediction, self.y_final], feed_dict={self.x:data, self.y_: labels})
            
        outliers_outputs=None
        if outliers is not None:
            outliers_outputs = sess.run(self.y_final, feed_dict={self.x: outliers})
        
        for i in range(0, 101):
            c, e, r_c, r_e = self.test_cer_internal(data, predictions, outputs, outliers, outliers_outputs, threshold=i*0.01, threshold_method=threshold_method, logging=False)
            #
            rejection_rate = 100.0 * (r_c.shape[0] + r_e.shape[0]) / (c.shape[0] + e.shape[0] + r_c.shape[0] + r_e.shape[0])
            classification_rate = 0.0
            if c.shape[0] + e.shape[0] != 0:
                classification_rate = 100.0 * c.shape[0] / (c.shape[0] + e.shape[0])
            if rejection_rate > rejection_rate_limit:
                rejected.append(rejection_rate_limit)
                correct.append(classification_rate)
                break
            else:
                rejected.append(rejection_rate)
                correct.append(classification_rate)
            #
            if e.shape[0] + r_e.shape[0] == 0:
                rejected_errors.append(100.0)
            else:
                rejected_errors.append(100.0 * r_e.shape[0] / (e.shape[0] + r_e.shape[0]))
            rejected_correct.append(100.0 * r_c.shape[0] / (c.shape[0] + r_c.shape[0]))
                
        return rejected, correct, rejected_errors, rejected_correct
    
    
    def test_cer_internal(self, data, predictions, outputs, outliers=None, outliers_outputs=None, threshold=0.0, threshold_method = 0, logging = True):
        correct = []
        error = []
        reject_correct = []
        reject_error = []
            
        for sample, is_correct, output in zip(data, predictions, outputs):
            if is_correct:
                if self.reject_method(output, threshold, threshold_method):
                    reject_correct.append(sample)
                else:
                    correct.append(sample)
            else:
                if self.reject_method(output, threshold, threshold_method):
                    reject_error.append(sample)
                else:
                    error.append(sample)
            
        if outliers is not None:            
            for sample, output in zip(outliers, outliers_outputs):
                if self.reject_method(output, threshold, threshold_method):
                    reject_error.append(sample)
                else:
                    error.append(sample)
        
        #self.log_results(threshold, correct, error, reject_correct, reject_error, logging, self.log_file)      
        return np.array(correct), np.array(error), np.array(reject_correct), np.array(reject_error)
        
    
    def reject_method(self, y, threshold, i):
        if i == 0:
            return y.max() < threshold
        elif i == 1:
            output = np.copy(y)
            max1 = output.max()
            output[output.argmax()] = 0;
            max2 = output.max()
            return max1 - max2 < threshold
        elif i == 2:
            output = np.copy(y)
            max1 = output.max()
            output[output.argmax()] = 0;
            max2 = output.max()
            return 1.0 - max2/max1 < threshold
        else:
            raise ValueError('Unknown rejection method')
            
    def clean_model_dir(self):
        dirPath = "saver/"
        fileList = os.listdir(dirPath)
        for fileName in fileList:
            os.remove(dirPath+"/"+fileName)
    
    def log_init(self, steps, logging):
        
        self.log_file = None
        if logging: 
            self.log_file = open("tests/test_%g_%d_%s_%d.txt"%(self.learning_rate, steps, self.dataset_name, int(time.time())), "w+")
    
        log_msg = "learing rate: %g; steps: %d"%(self.learning_rate, steps)
        if logging: print(log_msg, file=self.log_file)
        else:       print(log_msg)
            
    def log_accuracy(self, i, loss, trn_acc, vld_acc, areas=[0,0,0], mark='', logging=True):
        
        if vld_acc != 0:
            log_msg = 'step{:8d}, loss:{:9f}, trn_acc:{:9f}, vld_acc{:9f}, r0:{:12f}, r1:{:12f}, r2:{:12f}, {:s}'.format(i, loss, trn_acc, vld_acc, areas[1], areas[2], areas[3], mark)
            if logging: print(log_msg, file=self.log_file)
            else:       print(log_msg)
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