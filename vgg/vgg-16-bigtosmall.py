import tensorflow as tf
import tempfile
import time
import sys
import numpy as np
from decimal import Decimal

from scipy.misc import imread, imresize

from tensorflow.python.client import timeline
from tensorflow.python.profiler import model_analyzer
from tensorflow.python.profiler import option_builder


def conv2D(name, l_input, w, b):  
    return tf.nn.bias_add(tf.nn.conv2d(l_input, w, strides=[1, 1, 1, 1], padding='SAME'),b)
    #return tf.nn.bias_add(tf.nn.conv2d(l_input, w, strides=[1, 1, 1, 1], padding='SAME'),b)
    #return tf.nn.bias_add(tf.nn.conv2d(l_input, w, strides=[1, 1, 1, 1], padding='SAME'),b,name=name)

def maxPool2D(name, l_input, k):  
    return tf.nn.max_pool(l_input, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')  
  
def norm(name, l_input, lsize=4):  
    return tf.nn.lrn(l_input, lsize, bias=1.0, alpha=0.001 / 9.0, beta=0.75)  

weights ={  
    'wc1':tf.Variable(tf.random_normal([3,3,3,64])),  
    'wc2':tf.Variable(tf.random_normal([3,3,64,64])),  
    'wc3':tf.Variable(tf.random_normal([3,3,64,128])),  
    'wc4':tf.Variable(tf.random_normal([3,3,128,128])),  
      
    'wc5':tf.Variable(tf.random_normal([3,3,128,256])),  
    'wc6':tf.Variable(tf.random_normal([3,3,256,256])),  
    'wc7':tf.Variable(tf.random_normal([3,3,256,256])),  
    'wc8':tf.Variable(tf.random_normal([3,3,256,256])),  
      
    'wc9':tf.Variable(tf.random_normal([3,3,256,512])),  
    'wc10':tf.Variable(tf.random_normal([3,3,512,512])),  
    'wc11':tf.Variable(tf.random_normal([3,3,512,512])),  
    'wc12':tf.Variable(tf.random_normal([3,3,512,512])),  
    'wc13':tf.Variable(tf.random_normal([3,3,512,512])),  
    'wc14':tf.Variable(tf.random_normal([3,3,512,512])),  
    'wc15':tf.Variable(tf.random_normal([3,3,512,512])),  
    'wc16':tf.Variable(tf.random_normal([3,3,512,256])),  
      
    'wd1':tf.Variable(tf.random_normal([4096,4096])),  
    'wd2':tf.Variable(tf.random_normal([4096,4096])),  
    'out':tf.Variable(tf.random_normal([4096,10])),  
}  
  
biases ={  
    'bc1':tf.Variable(tf.constant(value=0.0001,shape=[64])),  
    'bc2':tf.Variable(tf.constant(value=0.0001,shape=[64])),  
    'bc3':tf.Variable(tf.constant(value=0.0001,shape=[128])),  
    'bc4':tf.Variable(tf.constant(value=0.0001,shape=[128])),  
    'bc5':tf.Variable(tf.constant(value=0.0001,shape=[256])),  
    'bc6':tf.Variable(tf.constant(value=0.0001,shape=[256])),  
    'bc7':tf.Variable(tf.constant(value=0.0001,shape=[256])),  
    'bc8':tf.Variable(tf.constant(value=0.0001,shape=[256])),  
    'bc9':tf.Variable(tf.constant(value=0.0001,shape=[512])),  
    'bc10':tf.Variable(tf.constant(value=0.0001,shape=[512])),  
    'bc11':tf.Variable(tf.constant(value=0.0001,shape=[512])),  
    'bc12':tf.Variable(tf.constant(value=0.0001,shape=[512])),  
    'bc13':tf.Variable(tf.constant(value=0.0001,shape=[512])),  
    'bc14':tf.Variable(tf.constant(value=0.0001,shape=[512])),  
    'bc15':tf.Variable(tf.constant(value=0.0001,shape=[512])),  
    'bc16':tf.Variable(tf.constant(value=0.0001,shape=[256])),  
      
      
    'bd1':tf.Variable(tf.constant(value=0.0001,shape=[4096])),  
    'bd2':tf.Variable(tf.constant(value=0.0001,shape=[4096])),  
    'out':tf.Variable(tf.constant(value=0.0001,shape=[10])),  
}

def convLevel(i,input,type):  
    num = i  
    with tf.name_scope('conv'+str(num)):
        out = conv2D('conv'+str(num),input,weights['wc'+str(num)],biases['bc'+str(num)])  
        if  type=='p':  
            out = maxPool2D('pool'+str(num),out, k=2)   
            out = norm('norm'+str(num),out, lsize=4)  
        return out
  
  
def VGG(x,weights,biases,dropout):  
    x = tf.reshape(x,shape=[-1,224,224,3])
    inputs = x
    for i in range(16):
        i += 1  
        if(i==2) or (i==4) or (i==12) : 
            inputs = convLevel(i,inputs,'p') 
        else:  
            inputs = convLevel(i,inputs,'c')

    dense1 = tf.reshape(inputs, [-1, weights['wd1'].get_shape().as_list()[0]])  
    dense1 = tf.nn.relu(tf.matmul(dense1, weights['wd1']) + biases['bd1'], name='fc1')

    dense1 = tf.nn.dropout(dense1, dropout)

    dense2 = tf.reshape(dense1, [-1, weights['wd2'].get_shape().as_list()[0]])  
    dense2 = tf.nn.relu(tf.matmul(dense1, weights['wd2']) + biases['bd2'], name='fc2')

    dense2 = tf.nn.dropout(dense2, dropout)  

    out = tf.matmul(dense2, weights['out']) + biases['out']

    return out


def main(iternum,gpu_size):

  learning_rate = 0.001
  batch_size = 64
  n_classes = 10

  #n_input = 784
  dropout = 0.75

  img =[]
  ylabel = []
  #x = tf.placeholder(tf.float32, [None, n_input])  
  x = tf.placeholder(tf.float32, [None, 224,224,3])

  img1 = imread('weasel.png',mode='RGB')
  img1 = imresize(img1,(224,224))
  #img1 = np.reshape(img1,(224,224,1))
  #img2 = imread('weasel.png',mode='L')
  #img2 = imresize(img1,(224,224))

  for i in range(1,batch_size+1):
    img.append(img1)

  ylabel.append([0,1,0,0,0,0,0,0,0,0])
  #ylabel.append([1,0])
 
  y = tf.placeholder(tf.float32, [None, n_classes])  
  z = tf.placeholder(tf.float32)  
  keep_prob = tf.placeholder(tf.float32)

  pred = VGG(x, weights, biases, keep_prob)  
	
  cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=pred, labels=y))
  #cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=pred, labels=y))

  optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)  
   
  correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))  
  accuracy_ = tf.reduce_mean(tf.cast(correct_pred,tf.float32))  

  init = tf.global_variables_initializer()

  graph_location = tempfile.mkdtemp()
  train_writer = tf.summary.FileWriter(graph_location)

  from tensorflow.core.protobuf import rewriter_config_pb2
  rewrite_options = rewriter_config_pb2.RewriterConfig(disable_model_pruning=True,batch_size_num=batch_size)
  rewrite_options.memory_optimization = rewriter_config_pb2.RewriterConfig.ITRI_SWAPPING
# RECOMPUTATION_HEURISTICS SWAPPING_HEURISTICS NO_MEM_OPT ITRI_SWAPPING
  rewrite_options.layout_optimizer = rewriter_config_pb2.RewriterConfig.OFF
  rewrite_options.function_optimization = rewriter_config_pb2.RewriterConfig.OFF
  rewrite_options.constant_folding = rewriter_config_pb2.RewriterConfig.OFF
  rewrite_options.arithmetic_optimization = rewriter_config_pb2.RewriterConfig.OFF
  rewrite_options.loop_optimization = rewriter_config_pb2.RewriterConfig.OFF
  rewrite_options.dependency_optimization = rewriter_config_pb2.RewriterConfig.OFF

  graph_options = tf.GraphOptions(rewrite_options=rewrite_options)#,infer_shapes=True)
  #config = tf.ConfigProto(graph_options=graph_options)
  #config = tf.ConfigProto()
  #config.gpu_options.allow_growth=True
  #config.allow_soft_placement = True

  gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction = gpu_size)# low 
  config = tf.ConfigProto(graph_options=graph_options,gpu_options=gpu_options)
  #config = tf.ConfigProto(gpu_options=gpu_options)

  run_metadata = tf.RunMetadata()
  with tf.Session(config=config) as sess: 
    sess.run(init)  
    step = 1  
    while step < iternum:  
      tStart = time.time()
      #sess.run(optimizer,feed_dict={x:img,y:ylabel,keep_prob:dropout})
      sess.run(optimizer,feed_dict={x:img,y:ylabel,keep_prob:dropout},options=tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE),run_metadata=run_metadata)
      tEnd = time.time()
      #summary_writer = tf.summary.FileWriter("./board1", graph=tf.get_default_graph())
      #trace = timeline.Timeline(step_stats=run_metadata.step_stats)
      step += 1
      #with open('SWAPPING_HEURISTICS.json', 'w') as trace_file:
        #trace_file.write(trace.generate_chrome_trace_format())

      print("training end! %lf " %(tEnd-tStart))
    #_ = input('Press any key to start... ')   

if __name__ == '__main__':
    #_ = input('Press any key to start... ')
    num = int(sys.argv[1])
    gpu_size = Decimal(sys.argv[2])
    main(num,gpu_size)

