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

def IOU_(y_pred, y_true):
    """Returns a (approx) IOU score
    intesection = y_pred.flatten() * y_true.flatten()
    Then, IOU = 2 * intersection / (y_pred.sum() + y_true.sum() + 1e-7) + 1e-7
    Args:
        y_pred (4-D array): (N, H, W, 1)
        y_true (4-D array): (N, H, W, 1)
    Returns:
        float: IOU score
    """
    H, W, _ = y_pred.get_shape().as_list()[1:]

    pred_flat = tf.reshape(y_pred, [-1, H*W])
    true_flat = tf.reshape(y_true, [-1, H*W])

    intersection = 2 * tf.reduce_sum(pred_flat * true_flat, axis=1) + 1e-7
    denominator = tf.reduce_sum(
        pred_flat, axis=1) + tf.reduce_sum(
            true_flat, axis=1) + 1e-7

    return tf.reduce_mean(intersection / denominator)


def make_train_op(y_pred, y_true):
    """Returns a training operation
    Loss function = - IOU(y_pred, y_true)
    IOU is
        (the area of intersection)
        --------------------------
        (the area of two boxes)
    Args:
        y_pred (4-D Tensor): (N, H, W, 1)
        y_true (4-D Tensor): (N, H, W, 1)
    Returns:
        train_op: minimize operation
    """
    loss = -IOU_(y_pred, y_true)

    global_step = tf.train.get_or_create_global_step()

    optim = tf.train.AdamOptimizer()
    return optim.minimize(loss, global_step=global_step)


def upconv_2D(tensor, n_filter, name):
    """Up Convolution `tensor` by 2 times
    Args:
        tensor (4-D Tensor): (N, H, W, C)
        n_filter (int): Filter Size
        name (str): name of upsampling operations
    Returns:
        output (4-D Tensor): (N, 2 * H, 2 * W, C)
    """

    return tf.layers.conv2d_transpose(
        tensor,
        filters=n_filter,
        kernel_size=2,
        strides=2,
        kernel_initializer=tf.truncated_normal_initializer(seed=1),
        kernel_regularizer=tf.contrib.layers.l2_regularizer(0.1),
        name="upsample_{}".format(name))

def upconv_concat(inputA, input_B, n_filter, name):
    """Upsample `inputA` and concat with `input_B`
    Args:
        input_A (4-D Tensor): (N, H, W, C)
        input_B (4-D Tensor): (N, 2*H, 2*H, C2)
        name (str): name of the concat operation
    Returns:
        output (4-D Tensor): (N, 2*H, 2*W, C + C2)
    """
    up_conv = upconv_2D(inputA, n_filter, name)

    return tf.concat(
        [up_conv, input_B], axis=-1, name="concat_{}".format(name))

def conv_conv_pool(input_,
                   n_filters,
                   training,
                   name,
                   pool=True,
                   activation=tf.nn.relu):
    """{Conv -> BN -> RELU}x2 -> {Pool, optional}
    Args:
        input_ (4-D Tensor): (batch_size, H, W, C)
        n_filters (list): number of filters [int, int]
        training (1-D Tensor): Boolean Tensor
        name (str): name postfix
        pool (bool): If True, MaxPool2D
        activation: Activaion functions
    Returns:
        net: output of the Convolution operations
        pool (optional): output of the max pooling operations
    """
    net = input_

    with tf.variable_scope("layer{}".format(name)):
        for i, F in enumerate(n_filters):
            net = tf.layers.conv2d(
                net,
                F, (3, 3),
                activation=None,
                padding='same',
                kernel_regularizer=tf.contrib.layers.l2_regularizer(0.1),
                name="conv_{}".format(i + 1))
            net = tf.layers.batch_normalization(
                net, training=training, name="bn_{}".format(i + 1))
            net = activation(net, name="relu{}_{}".format(name, i + 1))

        if pool is False:
            return net

        pool = tf.layers.max_pooling2d(
            net, (2, 2), strides=(2, 2), name="pool_{}".format(name))

        return net, pool


def make_unet(X, training, flags=None):
    """Build a U-Net architecture
    Args:
        X (4-D Tensor): (N, H, W, C)
        training (1-D Tensor): Boolean Tensor is required for batchnormalization layers
    Returns:
        output (4-D Tensor): (N, H, W, C)
            Same shape as the `input` tensor
    Notes:
        U-Net: Convolutional Networks for Biomedical Image Segmentation
        https://arxiv.org/abs/1505.04597
    """
    net = X # / 127.5 - 1
    conv1, pool1 = conv_conv_pool(net, [8, 8], training, name=1)
    conv2, pool2 = conv_conv_pool(pool1, [16, 16], training, name=2)
    conv3, pool3 = conv_conv_pool(pool2, [32, 32], training, name=3)
    conv4, pool4 = conv_conv_pool(pool3, [64, 64], training, name=4)
    conv5 = conv_conv_pool(
        pool4, [128, 128], training, name=5, pool=False)

    up6 = upconv_concat(conv5, conv4, 64, name=6)
    conv6 = conv_conv_pool(up6, [64, 64], training, name=6, pool=False)

    up7 = upconv_concat(conv6, conv3, 32, name=7)
    conv7 = conv_conv_pool(up7, [32, 32], training, name=7, pool=False)

    up8 = upconv_concat(conv7, conv2, 16, name=8)
    conv8 = conv_conv_pool(up8, [16, 16], training, name=8, pool=False)

    up9 = upconv_concat(conv8, conv1,8, name=9)
    conv9 = conv_conv_pool(up9, [8, 8], training, name=9, pool=False)
    """
    dense1 = tf.reshape(conv9, [224*8,224*8])  
    dense1 = tf.nn.relu(tf.matmul(dense1, tf.Variable(tf.random_normal([224*8,224*8]))  ) + tf.Variable(tf.constant(value=0.0001,shape=[224*8])) , name='fc1')
    out = tf.matmul(dense1,tf.Variable(tf.random_normal([224*8,10]))  ) + tf.Variable(tf.constant(value=0.0001,shape=[10]) )
    return out
    """
    return tf.layers.conv2d(
        conv9,
        1, (1, 1),
        name='final',
        activation=tf.nn.sigmoid,
        padding='same')
    

def main(iternum,gpu_size):

  learning_rate = 0.001
  batch_size = 327
#batch size 128
# NO_MEM_OPT		0.56
# ITRI_SWAPPING 0.218

#mem usage 0.56(6613 MB)
# NO_MEM_OPT		128		174.53
# ITRI_SWAPPING 

  n_classes = 10
  dropout = 0.75

  img =[]
  ylabel = []
  #x = tf.placeholder(tf.float32, [None, n_input])  
  x = tf.placeholder(tf.float32, [None, 224,224,3])
  y = tf.placeholder(tf.float32, [None, 224,224,1])

  img1 = imread('weasel.png',mode='RGB')
  img1 = imresize(img1,(224,224))
  #img1 = np.reshape(img1,(224,224,1))
  #img2 = imread('weasel.png',mode='L')
  #img2 = imresize(img1,(224,224))

  for i in range(1,batch_size+1):
    img.append(img1)
    
  ylabel = np.zeros([batch_size,224,224,1])
  #ylabel.append([0,1,0,0,0,0,0,0,0,0])

  #y = tf.placeholder(tf.float32, [None, 10])  
  mode = tf.placeholder(tf.bool, name="mode")

  pred = make_unet(x, mode)  
  #cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=pred, labels=y))
  #optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost) 
  with tf.name_scope('adam_optimizer'): 
    optimizer = make_train_op(pred,y)

  """
  cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=pred, labels=y))
  #cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=pred, labels=y))
  with tf.name_scope('adam_optimizer'):
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)  
   
  correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))  
  accuracy_ = tf.reduce_mean(tf.cast(correct_pred,tf.float32))  
  """
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
      #sess.run(optimizer,feed_dict={x:img,y:ylabel,mode:True}
      sess.run(optimizer,feed_dict={x:img,y:ylabel,mode:True},options=run_options)
      #sess.run(optimizer,feed_dict={x:img,y:ylabel,keep_prob:dropout},options=tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE),run_metadata=run_metadata)
      tduration = time.time() - tStart
      #summary_writer = tf.summary.FileWriter("./test3", graph=tf.get_default_graph())
      #trace = timeline.Timeline(step_stats=run_metadata.step_stats)
      print ("examples_per_sec %.1f ,sec_per_batch %.3f " %((batch_size/tduration) ,tduration) )
      #print(sess.run(accuracy_, feed_dict={x: img, y: [[0,1,0,0,0,0,0,0,0,0]], keep_prob: dropout}))
      step += 1
      

      #with open('case3.json', 'w') as trace_file:
      #  trace_file.write(trace.generate_chrome_trace_format())

    ttend = time.time() - ttStart
    #print("total training time %lf " %(ttend))
    #_ = input('Press any key to start... ')   


if __name__ == '__main__':
    #_ = input('Press any key to start... ')
    num = int(sys.argv[1])
    gpu_size = Decimal(sys.argv[2])
    main(num,gpu_size)


#https://github.com/kkweon/UNet-in-Tensorflow/blob/master/train.py
