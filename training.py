import data_preprocessing
import numpy as np
import tensorflow as tf

#training version v7
# Parameters
WindowSize = (7, 7)
Window_width = WindowSize[0]
Window_height = WindowSize[1]
Half_width = int((WindowSize[0] - 1) / 2)
Half_height = int((WindowSize[1] - 1) / 2)
border_padValue = 0.0

#data processing
data1 = data_preprocessing.preprocess('three_samples_50x50_100samples.txt',WindowSize,Half_width,Half_height,border_padValue)
data2 = data_preprocessing.preprocess('full_samples_50x50_600samples.txt',WindowSize,Half_width,Half_height, border_padValue)
data3 = data_preprocessing.preprocess('null_samples_50x50_100samples.txt',WindowSize,Half_width,Half_height, border_padValue)
data = data1 + data2 + data3
data = np.array(data)
shuffle_indices = np.random.permutation(np.arange(len(data)))
shuffled_data = data[shuffle_indices]
train_sample_index = int( 0.92 * float(len(data)) )
training_data,test_data = shuffled_data[:train_sample_index], shuffled_data[train_sample_index:]
print('Number of training data samples: {}'.format(len(training_data)))
print('Number of test data samples: {}'.format(len(test_data)))
test_x, test_y = zip(*test_data)
test_x = np.array(test_x)
test_y = np.array(test_y)

#build graph
x = tf.placeholder(tf.float32, shape = [None, Window_height*Window_width] )
y_ = tf.placeholder(tf.float32, shape = [None, 77])

def weight_variable(shape):
    initial = tf.truncated_normal(shape,stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x,W,padding):
    """

    :param x:
    :param W:
    :param padding: if padding is 'SAME', out_height = ceil(float(in_height) / float(strides[1])), out_width  = ceil(float(in_width) / float(strides[2]))
                    if padding is 'valid', out_height = ceil(float(in_height - filter_height + 1) / float(strides[1])), out_width  = ceil(float(in_width - filter_width + 1) / float(strides[2]))
    :return:
    """
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding=padding)

def max_pool_2x2(x,padding):
    """

    :param x:
    :param padding: If padding == "SAME": output_spatial_shape[i] = ceil(input_spatial_shape[i] / strides[i])
                    If padding == "VALID": output_spatial_shape[i] = ceil((input_spatial_shape[i] - (spatial_filter_shape[i]-1) * dilation_rate[i]) / strides[i]).
    :return:
    """
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,1,1,1],padding=padding)

# first conv layer
W_conv1 = weight_variable([4,4,1,100])
b_conv1 = bias_variable([100])
x_puzzle = tf.reshape(x,[-1,Window_height,Window_width,1])

h_conv1 = tf.nn.relu( conv2d(x_puzzle,W_conv1,'SAME') + b_conv1 )
dim_height = Window_height
dim_width = Window_width

h_pool1 = max_pool_2x2( h_conv1,'VALID' ) #output_dim = ceil( (input_dim - (filter_dim -1) ) / stride ) = ceil( (7-(2-1)) / 1 ) = 6
dim_height = dim_height - 1
dim_width = dim_width - 1

# second conv layer
W_conv2 = weight_variable([4,4,100,200])
b_conv2 = bias_variable([200])

h_conv2 = tf.nn.relu( conv2d(h_pool1, W_conv2, 'SAME') + b_conv2 )
dim_height = dim_height
dim_width = dim_width
h_pool2 = max_pool_2x2(h_conv2,'VALID')
dim_height = dim_height - 1
dim_width = dim_width - 1

assert dim_height == 5, 'dimension not correct'
assert dim_width == 5, 'dimension not correct'

# densely connected layer
W_fc1 = weight_variable([dim_height*dim_width*200, 2000])
b_fc1 = bias_variable([2000])
h_pool2_flat = tf.reshape(h_pool2 , [-1, dim_height*dim_width*200])
h_fc1 = tf.nn.relu( tf.matmul(h_pool2_flat, W_fc1) +b_fc1 )

# dropout layer
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# classifier layer
W_fc2 = weight_variable([2000,77])
b_fc2 = weight_variable([77])
y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

# softmax and cross_entropy
cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y_,logits=y_conv)
)
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal( tf.argmax(y_conv,1), tf.argmax(y_,1) )
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# summary
tf.summary.scalar('accuracy',accuracy)
merged = tf.summary.merge_all()
train_writer = tf.summary.FileWriter('../TetrisPuzzle_summary/train_v7')
test_writer = tf.summary.FileWriter('../TetrisPuzzle_summary/test_v7')

# save
saver = tf.train.Saver()
best_train_acc = 0.0
best_test_acc = 0.0

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    batches = data_preprocessing.data_batch_iter(training_data,50,3)
    i = 0
    for inputdata,outputdata in batches:
        if i%100 == 0:
            # train accuracy
            summary,train_accuracy = sess.run([merged,accuracy], feed_dict={x:inputdata,y_:outputdata, keep_prob:1.0})
            train_writer.add_summary(summary,i)
            print('step %d, training accuracy %g' % (i, train_accuracy))
            # save best train model
            if train_accuracy > best_train_acc:
                best_train_acc = train_accuracy
                save_path = saver.save(sess,'../TetrisPuzzle_saver/train_v7.ckpt')
                print("best Train acc. Model saved in file: %s" % save_path)

        if i%1000 == 0:
            # test accuracy
            summary, test_accuracy = sess.run([merged,accuracy], feed_dict={x:test_x, y_:test_y, keep_prob:1.0})
            test_writer.add_summary(summary,i)
            print('step %d, test accuracy %g' % (i, test_accuracy))
            # save best test model
            if test_accuracy > best_test_acc:
                best_test_acc = test_accuracy
                save_path = saver.save(sess, '../TetrisPuzzle_saver/test_v7.ckpt')
                print("best Test acc. Model saved in file: %s" % save_path)

        train_step.run(feed_dict={x:inputdata, y_:outputdata, keep_prob:0.5})
        i += 1
