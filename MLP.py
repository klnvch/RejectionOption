'''
Created on Oct 26, 2016

@author: anton

This module keeps all code for Tensorflow to implement MLP
The code is imperfect and must be replaced with Keras library
'''

import time, os
import tensorflow as tf
from sklearn.utils import shuffle
#from RBF import get_kmeans_centers

class MLP:
    
    def __init__(self, learning_rate, layers, functions,
                 optimizer_name=None, beta=0.0, batch_size=None):
        """ Creates ANN tensorflow model
        Args:
            layers: [4,5,6,7]
            functions: [relu, sigmoid, softmax]
        """
        print('create neural network...')
        log_msg = 'learning rate: {:g}, function: {:s}, optimizer: {:s}'
        log_msg = log_msg.format(learning_rate, str(functions), optimizer_name)
        print(log_msg)
        log_msg = 'layers: {:s}, beta: {:g}, batch size: {:d}'
        log_msg = log_msg.format(str(layers), beta, batch_size)
        print(log_msg)
        
        functions = self.parse_functions(functions)
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.optimizer_name = optimizer_name
        self.n_input = layers[0]
        self.n_hidden = layers[1:-1]
        self.n_output = layers[-1]
        
        self.x = tf.placeholder(tf.float32, [None, self.n_input])
        self.y_ = tf.placeholder(tf.float32, [None, self.n_output])
        self.keep_prob = tf.placeholder(tf.float32)
        
        if 'rbf' in functions or 'rbf-reg' in functions:
            if 'rbf' in functions:
                h, r1 = self.add_rbf(self.x, self.n_input, self.n_hidden[0])
            elif 'rbf-reg' in functions:
                h, r1 = self.add_rbf_reg(self.x, self.n_input, self.n_hidden[0])
            
            y, _ = self.add_layer(h,
                                   [self.n_hidden[0], self.n_output],
                                   None, '2', None)
            regularizers = r1
            
        elif 'conv' in functions:
            x = self.add_conv(self.x)
            h, _ = self.add_layer(x,
                                   [7 * 7 * 64, self.n_hidden[0]],
                                   functions[1], '1', self.keep_prob, None)
            y, _ = self.add_layer(h,
                                   [self.n_hidden[0], self.n_output],
                                   None, '2', None, None)
            regularizers = None
            
        elif len(self.n_hidden) == 0:
            y, r1 = self.add_layer(self.x,
                                   [self.n_input, self.n_output],
                                   None, '', None)
            regularizers = r1
            
        elif len(self.n_hidden) == 1:
            h, r1 = self.add_layer(self.x,
                                   [self.n_input, self.n_hidden[0]],
                                   functions[0], '1', self.keep_prob)
            y, r2 = self.add_layer(h,
                                   [self.n_hidden[0], self.n_output],
                                   None, '2', None)
            regularizers = r1 + r2
            
        elif len(self.n_hidden) == 2:
            h1, r1 = self.add_layer(self.x,
                                    [self.n_input, self.n_hidden[0]],
                                    functions[0], '1', self.keep_prob)
            h2, r2 = self.add_layer(h1,
                                    [self.n_hidden[0], self.n_hidden[1]],
                                    functions[1], '2', self.keep_prob)
            y, r3 = self.add_layer(h2,
                                    [self.n_hidden[1], self.n_output],
                                    None, '3', None)
            regularizers = r1 + r2 + r3
            
        elif len(self.n_hidden) == 3:
            h1, r1 = self.add_layer(self.x,
                                    [self.n_input, self.n_hidden[0]],
                                    functions[0], '1', self.keep_prob)
            h2, r2 = self.add_layer(h1,
                                    [self.n_hidden[0], self.n_hidden[1]],
                                    functions[1], '2', self.keep_prob)
            h3, r3 = self.add_layer(h2,
                                    [self.n_hidden[1], self.n_hidden[2]],
                                    functions[2], '3', self.keep_prob)
            y, r4 = self.add_layer(h3,
                                    [self.n_hidden[2], self.n_output],
                                    None, '4', None)
            regularizers = r1 + r2 + r3 + r4
        
        self.y_final = functions[-1](y)
        
        #if 'rbf' in functions or 'rbf-reg' in functions:
        #    self.loss = tf.reduce_mean(tf.square(self.y_ - self.y_final))
        if functions[-1] == tf.nn.softmax:
            self.loss = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(labels=self.y_,
                                                        logits=y))
        elif functions[-1] == tf.nn.sigmoid:
            self.loss = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(labels=self.y_,
                                                        logits=y))
        
        if regularizers is not None:
            self.loss = tf.reduce_mean(self.loss + beta * regularizers)
        else:
            self.loss = tf.reduce_mean(self.loss)
        
        #self.global_step = tf.Variable(0, trainable=False)
        #starter_learning_rate = 0.1
        #self.learning_rate = tf.train.exponential_decay(starter_learning_rate,
        #                        self.global_step, 1000, 0.96,
        #                        staircase=True)
        
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
            optimizer = tf.train.AdamOptimizer(self.learning_rate)
        elif optimizer_name == 'Ftrl':
            optimizer = tf.train.FtrlOptimizer(learning_rate)
        elif optimizer_name == 'RMSProp':
            optimizer = tf.train.RMSPropOptimizer(learning_rate)
        
        self.train_step = optimizer.minimize(self.loss)
        #                                     global_step=self.global_step)
        
        # grads_and_vars = optimizer.compute_gradients(self.loss)
        # grad_norms = [tf.nn.l2_loss(g) for g, _ in grads_and_vars]
        # self.grad_norm = tf.add_n(grad_norms)
        
        y_true = tf.equal(tf.argmax(self.y_final, 1), tf.argmax(self.y_, 1))
        self.accuracy = tf.reduce_mean(tf.cast(y_true, tf.float32))
    
    def add_layer(self, x, shape, activation_function, postfix_name,
                  keep_prob, regularization = tf.nn.l2_loss):
        w = tf.Variable(tf.truncated_normal(shape, stddev=0.1),
                        name='weights_' + postfix_name)
        b = tf.Variable(tf.constant(0.1, shape=[shape[1]]),
                        name='biases_' + postfix_name)
        if regularization is not None:  l2_loss = regularization(w)
        else:                           l2_loss = None
        if activation_function is None:
            return tf.matmul(x, w) + b, l2_loss
        else:
            layer = activation_function(tf.matmul(x, w) + b)
            if keep_prob is not None:
                layer = tf.nn.dropout(layer, keep_prob)
            return layer, l2_loss
    
    def add_rbf(self, x, n_input, n_centers):
        self.centers = tf.Variable(tf.random_uniform([n_centers, n_input],
                                                      -1.0, 1.0),
                                   name='centers')
        self.variances = tf.Variable(tf.truncated_normal([n_centers],
                                                         15.0, 5.0),
                                     name='sigmas')
        
        l2_loss = tf.nn.l2_loss(self.variances)
        
        e_x = tf.expand_dims(x, 1)
        e_c = tf.expand_dims(self.centers, 0)
        
        a = tf.squared_difference(e_x, e_c, 'norm')
        a = tf.reduce_sum(a, axis=2, name='reduce_sum')
        b = 2.0 * tf.square(self.variances, 'sigma')
        G = - tf.divide(a, b, 'divide')
        G = tf.exp(G, 'exponent')
        
        return G, l2_loss
    
    def add_rbf_reg(self, x, n_input, n_centers):
        self.centers = tf.Variable(tf.random_uniform([n_centers, n_input],
                                                      -1.0, 1.0),
                                   name='centers')
        self.variances = tf.Variable(tf.truncated_normal([n_centers, n_input],
                                                         15.0, 5.0),
                                     name='sigmas')
        
        l2_loss = tf.nn.l2_loss(self.variances)
        
        e_x = tf.expand_dims(x, 1)
        e_c = tf.expand_dims(self.centers, 0)
        
        diff = tf.squared_difference(e_x, e_c)
        
        a = 1.0 / tf.square(self.variances)
        a = tf.multiply(diff, a)
        a = tf.reduce_sum(a, axis=2)
    
        G = tf.exp(- a / 2.0)
        
        return G, l2_loss
    
    def add_conv(self, x):
        """
        only for MNIST dataset
        """
        # The first convolutional layer
        x = tf.reshape(x, [-1, 28, 28, 1])
        W_conv1 = tf.Variable(tf.truncated_normal([5, 5, 1, 32], stddev=0.1))
        b_conv1 = tf.Variable(tf.constant(0.1, shape=[32]))
        h_conv1 = tf.nn.relu(tf.nn.conv2d(x, W_conv1, strides=[1, 1, 1, 1],
                                          padding='SAME') + b_conv1)
        h_pool1 = tf.nn.max_pool(h_conv1, ksize=[1, 2, 2, 1],
                                 strides=[1, 2, 2, 1], padding='SAME')
        # The second convolutional layer
        W_conv2 = tf.Variable(tf.truncated_normal([5, 5, 32, 64], stddev=0.1))
        b_conv2 = tf.Variable(tf.constant(0.1, shape=[64]))
        h_conv2 = tf.nn.relu(tf.nn.conv2d(h_pool1, W_conv2, strides=[1, 1, 1, 1],
                                          padding='SAME') + b_conv2)
        h_pool2 = tf.nn.max_pool(h_conv2, ksize=[1, 2, 2, 1],
                                 strides=[1, 2, 2, 1], padding='SAME')
        # make flat again and return
        h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
        return h_pool2_flat
    
    def parse_functions(self, functions):
        def find_match(f):
            if   f == 'relu'   : return tf.nn.relu
            elif f == 'sigmoid': return tf.nn.sigmoid
            elif f == 'softmax': return tf.nn.softmax
            elif f == 'rbf':     return f
            elif f == 'rbf-reg': return f
            elif f == 'conv':    return f
            else: raise ValueError('wrong function: ' + f)
        return [find_match(f) for f in functions]
    
    def train(self, n_steps, trn, vld=None, keep_prob=1.0, early_stopping=None,
              log=True):
        """
        if early stopping not None output is
            [model_file, step, loss, trn_acc, vld_acc, area] for best_vld_acc,
             best_area0, best_area1 and best_area2
        """
        log_msg = 'steps: {:d}, dropout: {:g}, early stopping: {:d}'
        log_msg = log_msg.format(n_steps, keep_prob, early_stopping)
        print(log_msg)
        self.clean_model_dir()
        self.n_batches = int(trn.size / self.batch_size)
        self.vld = vld
        #self.log_init(steps, batch_size, log)
        
        saver = tf.train.Saver()
        save_path = 'saver/model.ckpt'
        dt = 0
        counter = 0
        best_vld_loss = 100000
        step = 0
        with tf.Session() as sess:
            #if hasattr(self, 'centers') and hasattr(self, 'variances'):
            #    c, v = get_kmeans_centers(trn.x, self.n_hidden[0])
            #    self.centers = tf.assign(self.centers, c)
            #    self.variances = tf.assign(self.variances, v)
            
            tf.global_variables_initializer().run()
            
            if hasattr(self, 'centers') and hasattr(self, 'variances'):
                print(sess.run(self.centers))
                print(sess.run(self.variances))
            
            for step in range(n_steps + 1):
                #print(sess.run(self.variances, {self.x: trn.x}))
                
                # train 
                start_time = time.time()
                x, y = shuffle(trn.x, trn.y)
                if self.batch_size is None:
                    sess.run(self.train_step,
                             feed_dict={self.x: x, self.y_: y,
                                        self.keep_prob: keep_prob})
                else:
                    for i in range(0, trn.x.shape[0], self.batch_size):
                        sess.run(self.train_step,
                                 feed_dict={self.x: x[i:i + self.batch_size],
                                            self.y_: y[i:i + self.batch_size],
                                            self.keep_prob: keep_prob})
                finish_time = time.time()
                dt += (finish_time - start_time)
                if step % 1 == 0:
                    trn_loss, trn_acc, vld_loss, vld_acc = \
                        self.log_step_info(sess, step, trn, vld, dt, log)
                    dt = 0
                    # early stopping
                    if vld is not None and early_stopping > 0:
                        if vld_loss < best_vld_loss:
                            best_vld_loss = vld_loss
                            counter = 0
                        else:
                            counter += 1
                        if counter >= early_stopping:
                            break
            
            info = [save_path, step, trn_loss, trn_acc, vld_acc]
            saver.save(sess, save_path)
            
            self.log_finish(sess)
        
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
            return sess.run(self.y_final, feed_dict={self.x: x,
                                                     self.keep_prob: 1.0})
    
    def clean_model_dir(self):
        dirPath = 'saver/'
        fileList = os.listdir(dirPath)
        for fileName in fileList:
            os.remove(dirPath + '/' + fileName)
    
    def log_init(self, steps, batch_size, logging):
        self.log_file = None
        if logging:
            filename = 'tests/{:s}_{:g}_{:d}_{:d}_{:s}_{:d}.csv'
            filename.format(self.optimizer_name,
                            self.learning_rate, steps, batch_size,
                            str(self.n_hidden), int(time.time()))
            self.log_file = open(filename, 'w+')
            print('Step,Loss,Train accuracy,Validation accuracy,'\
                  'ROC AUC: output threshold,ROC AUC: differential threshold,'\
                  'ROC AUC: ratio threshold,Time', file=self.log_file)
        
        log_msg = 'learing rate: {:f}; steps: {:d}'
        log_msg = log_msg.format(self.learning_rate, steps)
        if batch_size is not None:
            log_msg += '; batch size: {:d}'.format(batch_size)
        print(log_msg)
    
    def log_step_info(self, sess, step, trn, vld, train_time, logging):
        if trn.size < 8000:
            trn_loss, trn_acc = sess.run([self.loss, self.accuracy],
                feed_dict={self.x: trn.x, self.y_: trn.y,
                           self.keep_prob: 1.0})
        else:
            trn_loss, trn_acc = sess.run([self.loss, self.accuracy],
                feed_dict={self.x: trn.x[:8000], self.y_: trn.y[:8000],
                           self.keep_prob: 1.0})
        if vld is not None:
            vld_loss, vld_acc = sess.run([self.loss, self.accuracy],
                               feed_dict={self.x: vld.x,
                                          self.y_: vld.y,
                                          self.keep_prob: 1.0})
        else:
            vld_loss = 0
            vld_acc = 0
        
        log_msg = '{:8d}, {:9f}, {:9f}, {:9f}, {:9f}, {:f}'
        log_msg = log_msg.format(step, trn_loss, vld_loss, trn_acc, vld_acc, train_time)
        if logging: print(log_msg, file=self.log_file)
        print(log_msg)
        
        return trn_loss, trn_acc, vld_loss, vld_acc
    
    def log_finish(self, sess):
        if hasattr(self, 'centers') and hasattr(self, 'variances'):
            print(sess.run(self.centers))
            print(sess.run(self.variances))
        print('train finished')

if __name__ == '__main__':
    pass