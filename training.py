import data_preprocessing
import numpy as np
import tensorflow as tf

Window_width = data_preprocessing.WindowSize[0]
Window_height = data_preprocessing.WindowSize[1]

#data processing
data1 = data_preprocessing.preprocess('10_10.txt')
data2 = data_preprocessing.preprocess('20_20.txt')
data3 = data_preprocessing.preprocess('50_50.txt')
data = data1 + data2 + data3
data = np.array(data)
shuffle_indices = np.random.permutation(np.arange(len(data)))
shuffled_data = data[shuffle_indices]
train_sample_index = int( 0.945 * float(len(data)) )
training_data,test_data = shuffled_data[:train_sample_index], shuffled_data[train_sample_index:]
print('Number of training data samples: {}'.format(len(training_data)))
print('Number of test data samples: {}'.format(len(test_data)))
test_x, test_y = zip(*test_data)
test_x = np.array(test_x)
test_y = np.array(test_y)

#build graph
x = tf.placeholder(tf.float32, shape = [None, Window_width*Window_height])
y_ = tf.placeholder(tf.float32, shape = [None, 77])

def weight_variable(shape):
    initial = tf.truncated_normal(shape,stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x,W,padding):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding=padding)
    #return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')

def max_pool_2x2(x,padding):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 1, 1, 1], padding=padding)
    #return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,1,1,1],padding='VALID')

# first conv layer
W_conv1 = weight_variable([7,7,1,100])
b_conv1 = bias_variable([100])
x_puzzle = tf.reshape(x,[-1,13,13,1])

#'SAME' output_dim = ceil( input_dim / stride )
#'VALID' output_dim = ceil( (input_dim - (filter_dim -1) ) / stride )

h_conv1 = tf.nn.relu( conv2d(x_puzzle,W_conv1, 'VALID') + b_conv1 ) # output_dim = 7

h_pool1 = max_pool_2x2( h_conv1, 'SAME' ) #output_dim = 7

# second conv layer
W_conv2 = weight_variable([4,4,100,200])
b_conv2 = bias_variable([200])

h_conv2 = tf.nn.relu( conv2d(h_pool1, W_conv2, 'SAME') + b_conv2 ) # output_dim = 7
h_pool2 = max_pool_2x2(h_conv2, 'SAME') # output_dim = 7

# densely connected layer
W_fc1 = weight_variable([7*7*200, 3000])
b_fc1 = bias_variable([3000])
h_pool2_flat = tf.reshape(h_pool2 , [-1, 7*7*200])
h_fc1 = tf.nn.relu( tf.matmul(h_pool2_flat, W_fc1) +b_fc1 )

# dropout layer
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# classifier layer
W_fc2 = weight_variable([3000,77])
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
train_writer = tf.summary.FileWriter('../TetrisPuzzle_summary/train_bigwindow')
test_writer = tf.summary.FileWriter('../TetrisPuzzle_summary/test_bigwindow')

# save
saver = tf.train.Saver()
best_train_acc = 0.0
best_test_acc = 0.0

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    batches = data_preprocessing.data_batch_iter(training_data,50,5)
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
                save_path = saver.save(sess,'../TetrisPuzzle_saver/bigwindow_train.ckpt')
                print("best Train acc. Model saved in file: %s" % save_path)

        if i%1000 == 0:
            # test accuracy
            summary, test_accuracy = sess.run([merged,accuracy], feed_dict={x:test_x, y_:test_y, keep_prob:1.0})
            test_writer.add_summary(summary,i)
            print('step %d, test accuracy %g' % (i, test_accuracy))
            # save best test model
            if test_accuracy > best_test_acc:
                best_test_acc = test_accuracy
                save_path = saver.save(sess, '../TetrisPuzzle_saver/bigwindow_test.ckpt')
                print("best Test acc. Model saved in file: %s" % save_path)

        train_step.run(feed_dict={x:inputdata, y_:outputdata, keep_prob:0.5})
        i += 1
