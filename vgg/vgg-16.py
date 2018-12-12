import tensorflow as tf
import tempfile
import time
import sys
import numpy as np
from decimal import Decimal
import lms

from scipy.misc import imread, imresize

from tensorflow.python.client import timeline
from tensorflow.python.profiler import model_analyzer
from tensorflow.python.profiler import option_builder


def conv2D(name, l_input, w, b):  
    return tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(l_input, w, strides=[1, 1, 1, 1], padding='SAME'),b))
    #return tf.nn.bias_add(tf.nn.conv2d(l_input, w, strides=[1, 1, 1, 1], padding='SAME'),b)
    #return tf.nn.bias_add(tf.nn.conv2d(l_input, w, strides=[1, 1, 1, 1], padding='SAME'),b,name=name)

def maxPool2D(name, l_input, w, b):  
    return tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(l_input, w, strides=[1, 2, 2, 1], padding='SAME'),b))  

def maxPool(name, l_input, k):  
    return tf.nn.max_pool(l_input, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')  
  
def norm(name, l_input, lsize=4):  
    return tf.nn.lrn(l_input, lsize, bias=1.0, alpha=0.001 / 9.0, beta=0.75)  

weights ={  
    'wc1':tf.Variable(tf.random_normal([3,3,3,32],seed=2)),  
    'wc2':tf.Variable(tf.random_normal([3,3,32,128],seed=2)),  
    'wc3':tf.Variable(tf.random_normal([3,3,128,128],seed=2)),  
    'wc4':tf.Variable(tf.random_normal([3,3,128,512],seed=2)),  
      
    'wc5':tf.Variable(tf.random_normal([3,3,512,512],seed=2)),  
    'wc6':tf.Variable(tf.random_normal([3,3,512,512],seed=2)),  
    'wc7':tf.Variable(tf.random_normal([3,3,512,512],seed=2)),  
    'wc8':tf.Variable(tf.random_normal([3,3,512,512],seed=2)),  
      
    'wc9':tf.Variable(tf.random_normal([3,3,512,512],seed=2)),  
    'wc10':tf.Variable(tf.random_normal([3,3,512,512],seed=2)),  
    'wc11':tf.Variable(tf.random_normal([3,3,512,512],seed=2)),  
    'wc12':tf.Variable(tf.random_normal([3,3,512,2048],seed=2)),  
    'wc13':tf.Variable(tf.random_normal([3,3,2048,2048],seed=2)),  
    'wc14':tf.Variable(tf.random_normal([3,3,2048,2048],seed=2)),  
    'wc15':tf.Variable(tf.random_normal([3,3,2048,2048],seed=2)),  
    'wc16':tf.Variable(tf.random_normal([3,3,2048,2048],seed=2)),  
      
    'wd1':tf.Variable(tf.random_normal([4096,4096],seed=2)),  
    'wd2':tf.Variable(tf.random_normal([4096,4096],seed=2)),  
    'wd3':tf.Variable(tf.random_normal([4096,4096],seed=2)), 
    'out':tf.Variable(tf.random_normal([4096,10],seed=2)),  
}  
 
biases ={  
    'bc1':tf.Variable(tf.constant(value=0.0001,shape=[32])),  
    'bc2':tf.Variable(tf.constant(value=0.0001,shape=[128])),  
    'bc3':tf.Variable(tf.constant(value=0.0001,shape=[128])),  
    'bc4':tf.Variable(tf.constant(value=0.0001,shape=[512])),  
    'bc5':tf.Variable(tf.constant(value=0.0001,shape=[512])),  
    'bc6':tf.Variable(tf.constant(value=0.0001,shape=[512])),  
    'bc7':tf.Variable(tf.constant(value=0.0001,shape=[512])),  
    'bc8':tf.Variable(tf.constant(value=0.0001,shape=[512])),  
    'bc9':tf.Variable(tf.constant(value=0.0001,shape=[512])),  
    'bc10':tf.Variable(tf.constant(value=0.0001,shape=[512])),  
    'bc11':tf.Variable(tf.constant(value=0.0001,shape=[512])),  
    'bc12':tf.Variable(tf.constant(value=0.0001,shape=[2048])),  
    'bc13':tf.Variable(tf.constant(value=0.0001,shape=[2048])),  
    'bc14':tf.Variable(tf.constant(value=0.0001,shape=[2048])),  
    'bc15':tf.Variable(tf.constant(value=0.0001,shape=[2048])),  
    'bc16':tf.Variable(tf.constant(value=0.0001,shape=[2048])),  
      
      
    'bd1':tf.Variable(tf.constant(value=0.0001,shape=[4096])),  
    'bd2':tf.Variable(tf.constant(value=0.0001,shape=[4096])),  
    'bd3':tf.Variable(tf.constant(value=0.0001,shape=[4096])), 
    'out':tf.Variable(tf.constant(value=0.0001,shape=[10])),  
}

def convLevel(i,input,type):  
    num = i  
    with tf.name_scope('conv'+str(num)):

        out = conv2D('conv'+str(num),input,weights['wc'+str(num)],biases['bc'+str(num)])
        if  type=='p':  
            #out = maxPool2D('pool'+str(num),out,weights['wc'+str(num)], biases['bc'+str(num)])  
            out = maxPool('pool'+str(num),out, k=2)   
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
    dense1 = tf.nn.relu(tf.matmul(dense1, weights['wd1']) + biases['bd1'])

    dense1 = tf.nn.dropout(dense1, dropout)
    
    dense2 = tf.reshape(dense1, [-1, weights['wd2'].get_shape().as_list()[0]])  
    dense2 = tf.nn.relu(tf.matmul(dense1, weights['wd2']) + biases['bd2'])

    dense2 = tf.nn.dropout(dense2, dropout)  


    #dense3 = tf.reshape(dense2, [-1, weights['wd3'].get_shape().as_list()[0]])  
    #dense3 = tf.nn.relu(tf.matmul(dense2, weights['wd3']) + biases['bd3'])

    #dense3 = tf.nn.dropout(dense3, dropout)  

    out = tf.matmul(dense2, weights['out']) + biases['out']

    return out


def main(iternum,gpu_size):

  learning_rate = 0.001
  batch_size = 16
  #test gpu 			usage 0.614 - 16 
  #tests swap gpu usage 0.429 - 16

  #test gpu usage 			0.614		- 16	- 7227
  #test swap gpu usage 	0.55		- 16	- 6501
  #test swap gpu usage 	0.5			- 16	- 5943
  #test swap gpu usage 	0.45		- 16	- 5385
  #test swap gpu usage 	0.43		- 16	- 5161
  #test swap gpu usage 	0.42		- 16	- 5049

  #batch usage 0.6
  #non swap batch	- 15()
  #swap			batch -	32

  n_classes = 10
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

  for i in range(1,batch_size+1):
    if i%10==0:
      ylabel.append([1,0,0,0,0,0,0,0,0,0])
    if i%10==1:
      ylabel.append([0,1,0,0,0,0,0,0,0,0])
    if i%10==2:
      ylabel.append([0,0,1,0,0,0,0,0,0,0])
    if i%10==3:
      ylabel.append([0,0,0,1,0,0,0,0,0,0])
    if i%10==4:
      ylabel.append([0,0,0,0,1,0,0,0,0,0])
    if i%10==5:
      ylabel.append([0,0,0,0,0,1,0,0,0,0])
    if i%10==6:
      ylabel.append([0,0,0,0,0,0,1,0,0,0])
    if i%10==7:
      ylabel.append([0,0,0,0,0,0,0,1,0,0])
    if i%10==8:
      ylabel.append([0,0,0,0,0,0,0,0,1,0])
    if i%10==9:
      ylabel.append([0,0,0,0,0,0,0,0,0,1])

  y = tf.placeholder(tf.float32, [None, n_classes])  
  z = tf.placeholder(tf.float32)  
  keep_prob = tf.placeholder(tf.float32)


  pred = VGG(x, weights, biases, keep_prob)  
	
  cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=pred, labels=y))
  #cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=pred, labels=y))
  with tf.name_scope('adam_optimizer'):#AdamOptimizer
    optimizer = tf.train.AdamOptimizer(learning_rate=0.01).minimize(cost)  #0.01~0.01
   
  correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))  
  accuracy_ = tf.reduce_mean(tf.cast(correct_pred,tf.float32))  

  init = tf.global_variables_initializer()

  graph_location = tempfile.mkdtemp()
  train_writer = tf.summary.FileWriter(graph_location)

  from tensorflow.core.protobuf import rewriter_config_pb2
  rewrite_options = rewriter_config_pb2.RewriterConfig(disable_model_pruning=True,batch_size_num=batch_size)
  rewrite_options.memory_optimization = rewriter_config_pb2.RewriterConfig.NO_MEM_OPT 
# RECOMPUTATION_HEURISTICS SWAPPING_HEURISTICS NO_MEM_OPT ITRI_SWAPPING
  rewrite_options.layout_optimizer = rewriter_config_pb2.RewriterConfig.OFF
  rewrite_options.function_optimization = rewriter_config_pb2.RewriterConfig.OFF
  rewrite_options.constant_folding = rewriter_config_pb2.RewriterConfig.OFF
  rewrite_options.arithmetic_optimization = rewriter_config_pb2.RewriterConfig.OFF
  rewrite_options.loop_optimization = rewriter_config_pb2.RewriterConfig.OFF
  rewrite_options.dependency_optimization = rewriter_config_pb2.RewriterConfig.OFF

  graph_options = tf.GraphOptions(rewrite_options=rewrite_options)#,infer_shapes=True)

##### no_mem_opt 0.61 ,7081MB best 0.67 7841
##### swap layer1 2 	0.6 , lowest 0.6 
##### swap layer2 3-5 	0.57 , lowest 0.56
##### swap layer3 6 	0.56
##### swap layer4 7-9 	0.53-0.52, 0.51 increase training time, lowest 0.49
##### swap layer5 10 	0.5, 0.51 increase training time, lowest 0.48,  0.49can layer5
##### swap layer6 11 	0.51, 0.5 increase training time, lowest 0.47
##### swap layer9 	 	0.45, 
#swap layer lowest 0.43(helper)

####new case no_mem_opt only conv 0.63
##### swap layer8		0.58   , 0.54(no doing same layer)


#### case no_mem_opt has maxpool no norm 0.61
##### swap layer8		0.53

#### new  
##### swap layer1	0.65 //maxpool12 can't run if layer1 and reshape put gradients/conv14/ and gradients/conv13/  7619MB  
#swap lr1,reshape ->12 should 0.6

##### swap layer1	0.6		; lr1->10  re->9
##### swap layer1-2	0.58	; lr2->8   lr1->1  re->0   			  
##### swap layer1-3	0.57	; lr3->10  lr2->2  lr1->1  re->0        
##### swap layer1-4	0.53	; lr4->8   lr3->5  lr2->2  lr1->1  re->0
##### swap layer1-5	0.61	; lr5->10  lr4->8  lr3->3  lr2->2  lr1->1  re->0
##### swap layer1-6	0.62	; lr6->10  lr5->9  lr4->7  lr3->3  lr2->2  lr1->1  re->0
##### swap layer1-7	0.54  	; lr7->10  lr6->9  lr5->8  lr4->4  lr3->3  lr2->2  lr1->1  re->0
##### swap layer1-8	0.53  	; lr8->10  lr7->9  lr6->6  lr5->5  lr4->4  lr3->3  lr2->2  lr1->1  re->0
##### swap layer1-9	0.53	; lr9->10  lr8->8  lr7->7  lr6->6  lr5->5  lr4->4  lr3->3  lr2->2  lr1->1  re->0

  gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction = gpu_size) 
  config = tf.ConfigProto(graph_options=graph_options,gpu_options=gpu_options)#,log_device_placement=True)
  #config = tf.ConfigProto(graph_options=graph_options)
  #config.gpu_options.allow_growth=True
  #config.allow_soft_placement = True

  #config = tf.ConfigProto(gpu_options=gpu_options)
  run_options = tf.RunOptions(report_tensor_allocations_upon_oom = True)

  #lms_obj =lms.LMS({'adam_optimizer'})
  #lms_obj.run(graph=tf.get_default_graph())

  with tf.Session(config=config) as sess:  
  #with tf.Session() as sess:
    run_metadata = tf.RunMetadata()
    sess.run(init)

    step = 1
    ttStart = time.time()
    while step < iternum:  
      tStart = time.time()
      sess.run(optimizer,feed_dict={x:img,y:[ylabel[step-1]],keep_prob:1.})
      #optimizer.run(feed_dict={x:img,y:ylabel,keep_prob:dropout})
      #sess.run(optimizer,feed_dict={x:img,y:[ylabel[step-1]],keep_prob:dropout},options=run_options)
      #sess.run(optimizer,feed_dict={x:img,y:ylabel,keep_prob:dropout},options=tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE),run_metadata=run_metadata)
      tduration = time.time() - tStart
      #summary_writer = tf.summary.FileWriter("./vgg16", graph=tf.get_default_graph())
      #trace = timeline.Timeline(step_stats=run_metadata.step_stats)
      print ("examples_per_sec %.1f ,sec_per_batch %.3f " %((batch_size/tduration) ,tduration) )
      #print(sess.run(cost, feed_dict={x: img, y: [[1,0,0,0,0,0,0,0,0,0]], keep_prob: 1.}))
      step += 1

      """
      with open('case3.json', 'w') as trace_file:
        trace_file.write(trace.generate_chrome_trace_format())
      """
    ttend = time.time() - ttStart
    #print("total training time %lf " %(ttend))
    #_ = input('Press any key to start... ')   


if __name__ == '__main__':
    #_ = input('Press any key to start... ')
    num = int(sys.argv[1])
    gpu_size = Decimal(sys.argv[2])
    main(num,gpu_size)

