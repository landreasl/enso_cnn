#!/usr/bin/env python
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
import numpy as np
from netCDF4 import Dataset

tg_mn = int(target_mon - 1)
ld_mn1 = int(23 - lead_mon + tg_mn)
ld_mn2 = int(23 - lead_mon + tg_mn + 3)
batch_size = batsiz
num_convf = convfilter
num_hiddf = hiddfilter
xdim = xxx
ydim = yyy
zdim = zzz
xdim2 = int(xdim/4)
ydim2 = int(ydim/4)
sample_size = SAMSIZ
tot_size = TOTSIZ
conv_drop = CDRP
hidd_drop = HDRP

def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=1))

def init_bias(shape):
    return tf.Variable(tf.random_uniform(shape, minval=-0.01, maxval=0.01))

with tf.device('/cpu:0'): # Using CPU
  # Read Data (NetCDF4)
  inp1 = Dataset('/home/jhkim/data/SODA/samfile','r')
  inp2 = Dataset('/home/jhkim/data/SODA/labfile','r')

  inpv1 = np.zeros((sample_size,zzz,yyy,xxx))
  inpv1[:,0:3,:,:] = inp1.variables['sst'][0:sample_size,ld_mn1:ld_mn2,:,:]
  inpv1[:,3:6,:,:] = inp1.variables['t300'][0:sample_size,ld_mn1:ld_mn2,:,:]

  inpv2 = inp2.variables['pr'][0:sample_size,tg_mn,0]
  reinp1 = np.swapaxes(inpv1,1,3)  # (tdim,zdim,ydim,xdim) -> (tdim,xdim,ydim,zdim)
  
  trX = reinp1[:,:,:,:]
  trY = inpv2[:,:]    

with tf.device('/gpu:number_gpu'):

  # Define Dimension
  trX = trX.reshape(-1, xdim, ydim, zdim)  
  
  X = tf.placeholder(tf.float32, [None, xdim, ydim, zdim]) 
  Y = tf.placeholder(tf.float32, [None, 1])   
  
  w = init_weights([8, 4, zdim, num_convf])
  b = init_bias([num_convf])
  w2 = init_weights([4, 2, num_convf, num_convf]) 
  b2 = init_bias([num_convf])
  w3 = init_weights([4, 2, num_convf, num_convf])
  b3 = init_bias([num_convf])
  w4 = init_weights([num_convf * xdim2 * ydim2, num_hiddf])
  b4 = init_bias([num_hiddf])
  w_o = init_weights([num_hiddf, 1])
  b_o = init_bias([1])
  
  # Drop out
  p_keep_conv = tf.placeholder(tf.float32)
  p_keep_hidden = tf.placeholder(tf.float32)
  
  # Model
  l1a = tf.tanh(tf.nn.conv2d(X, w, strides=[1, 1, 1, 1], padding='SAME') + b)
  l1 = tf.nn.max_pool(l1a, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
  l1 = tf.nn.dropout(l1, p_keep_conv)
  
  l2a = tf.tanh(tf.nn.conv2d(l1, w2, strides=[1, 1, 1, 1], padding='SAME') + b2)
  l2 = tf.nn.max_pool(l2a, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
  l2 = tf.nn.dropout(l2, p_keep_conv)
  
  l3a = tf.tanh(tf.nn.conv2d(l2, w3, strides=[1, 1, 1, 1], padding='SAME') + b3)
  l3 = tf.reshape(l3a, [-1, w4.get_shape().as_list()[0]])
  l3 = tf.nn.dropout(l3, p_keep_conv)
  
  l4 = tf.tanh(tf.matmul(l3, w4) + b4)
  l4 = tf.nn.dropout(l4, p_keep_hidden)
  
  py_x = tf.matmul(l4, w_o) + b_o
  
  cost = tf.reduce_mean(tf.squared_difference(py_x, Y))
  batch = tf.Variable(0, dtype=tf.float32)
  train_op = tf.train.RMSPropOptimizer(0.005, 0.9).minimize(cost)
  predict_op = py_x

saver = tf.train.Saver()
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

# Launch the graph in a session
with tf.Session(config=config) as sess:
    saver.restore(sess,'home_directory/output/nino34_lead_monmonth_target_mon/opfname/ENmember/model.ckpt')

    print('-------------------------------------------------------------------------------')
    print('')
    print('Case: case')
    print('Parameter: opfname')
    print('Ensemblemember')
    print('')
    print('Training...')
    print('')

    for i in range(epoch):
        training_batch = zip(range(0, len(trX), batch_size), range(batch_size,
                                   len(trX)+1, batch_size))
        for start, end in training_batch:
            sess.run(train_op, feed_dict={X: trX[start:end], Y: trY[start:end], 
             p_keep_conv: conv_drop, p_keep_hidden: hidd_drop})

    save_path = saver.save(sess,'home_directory/output/case/opfname/ENmember/model.ckpt')


 