'''
Created on Dec 19, 2016

@author: anton
'''

from csv import reader
import matplotlib.pyplot as plt
import numpy as np

DIR_TEST = 'tests/'

FILE_TEST_OLD = ['sigmoid_gradient_0.1_20000_256_[24]_1481742134',
             'sigmoid_gradient_0.1_20000_256_[28]_1481744074',
             'sigmoid_adagrad_0.01_20000_256_[28]_rs_1482193004',
             'sigmoid_adagrad_0.01_20000_512_[28]_rs_1482200579',
             'sigmoid_gradient_0.1_20000_256_[32]_1481746487',
             'sigmoid_adagrad_0.01_20000_256_[32]_rs_1482194907',
             'sigmoid_adagrad_0.01_20000_512_[32]_rs_1482202214',
             'sigmoid_gradient_0.1_20000_256_[36]_1481748759',
             'sigmoid_adagrad_0.01_20000_256_[36]_rs_1482196776',
             'sigmoid_adagrad_0.01_20000_512_[36]_rs_1482203863',
             'sigmoid_adagrad_0.01_20000_256_[40]_rs_1482198668',
             'sigmoid_adagrad_0.01_20000_512_[40]_rs_1482205541',
             'softmax_gradient_0.01_10000_256_[24]_1481753606',
             'softmax_gradient_0.01_10000_256_[28]_1481754757',
             'softmax_adagrad_0.01_20000_256_[28]_rs_1482207235',
             'softmax_gradient_0.01_10000_256_[32]_1481755922',
             'softmax_adagrad_0.01_20000_256_[32]_rs_1482209491',
             'softmax_gradient_0.01_10000_256_[36]_1481757079',
             'softmax_adagrad_0.01_20000_256_[36]_rs_1482211759',
             'softmax_adagrad_0.01_20000_256_[40]_rs_1482214074',
             'softmax_gradient_0.01_10000_256_[10, 10]_1481834319',
             'sigmoid_gradient_0.1_10000_256_[16, 16]_1481835700',
             'sigmoid_adam_0.1_20000_256_[16, 16]_1481837697',
             'sigmoid_adagrad_0.1_10000_256_[16, 16]_1481842808',
             'sigmoid_adagrad_0.1_10000_256_[16, 16]_s_1481843900',
             'sigmoid_gradient_0.01_20000_256_[20, 20]_rs_1482216396',
             'sigmoid_adam_0.01_20000_256_[20, 20]_rs_1482218326',
             'sigmoid_adagrad_0.01_20000_256_[20, 20]_rs_1482220406',
             'sigmoid_adam_0.01_2000_128_[20, 20]_rs_gpu_1482258641',
             'sigmoid_adam_0.01_2000_512_[20, 20]_rs_gpu_1482259458',
             'sigmoid_adam_0.01_2000_128_[20, 20]_rs_1482259957',
             'sigmoid_adam_0.01_2000_512_[20, 20]_rs_1482260557']

FILE_TEST = ['sigmoid_adam_0.01_2000_128_[20, 20]_1482340044',
             'sigmoid_adam_0.01_2000_128_[20, 20]_1482340285']

def draw_curve(x, y, lgnd, use_max=True):
    if use_max: i = y.argmax()
    else:       i = y.argmin()
    
    label, = plt.plot(x, y, linewidth=2.0, label=lgnd+' ({:d}, {:g})'.format(int(x[i]), float(y[i])))
    plt.plot(x[i], y[i], 'ro')
    return label

def print_results(test, data):
    line = '| {:54s}'.format(test)
    line += ' | {:6d}'.format(data[-1,0].astype(np.int))
    line += ' | {:8f}'.format(data[-1,1].astype(np.float))
    line += ' | {:8f}'.format(data[-1,2].astype(np.float))
    line += ' | {:8f}'.format(data[-1,3].astype(np.float))
    line += ' | {:8f}'.format(data[-1,4].astype(np.float))
    line += ' | {:8f}'.format(data[-1,5].astype(np.float))
    line += ' | {:8f}'.format(data[-1,6].astype(np.float))
    line += ' | {:4d}'.format(int(np.sum(data[1:,7].astype(np.float)) / 60.0))
    line += ' |'
    print(line)


if __name__ == '__main__':
    for test in FILE_TEST:
        with open(DIR_TEST + test + '.csv', 'r') as f:
            data = list(reader(f))
            data = np.array(data)
            print_results(test, data)
        
            x = data[1:,0]
            
            plt.clf()
            l1 = draw_curve(x, data[1:,1], data[0,1], False)
            l2 = draw_curve(x, data[1:,2], data[0,2], True)
            l3 = draw_curve(x, data[1:,3], data[0,3], True)
            l4 = draw_curve(x, data[1:,4], data[0,4], True)
            l5 = draw_curve(x, data[1:,5], data[0,5], True)
            l6 = draw_curve(x, data[1:,6], data[0,6], True)
        
            
            plt.legend(handles=[l1,l2,l3,l4,l5,l6], numpoints=1, loc=0)
            plt.ylim([0.0, 1.0])
            plt.title(test)
            #mng = plt.get_current_fig_manager()
            #mng.full_screen_toggle()
            #plt.show()
            plt.gcf().set_size_inches(11.69, 8.27)
            plt.savefig('/home/anton/workspace2/DiplomaWork/images/' + test + '.png')
        
        