# try minist using our built tensorflow graph

from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import RL
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

def build_neuralnetwork(height,width):
    myGraph = tf.Graph()
    with myGraph.as_default():
        x = tf.placeholder(tf.float32,shape= [None,width*height], name='input_puzzles')
        y_ = tf.placeholder(tf.float32, shape = [None,10], name='labels')
        is_training = tf.placeholder(tf.bool, name='is_training')

        def conv_block(ipt, filter_kernalsize, channels_in, channels_out, is_training, name):
            with tf.name_scope(name):
                w = tf.Variable(tf.truncated_normal([filter_kernalsize[0], filter_kernalsize[1], channels_in,channels_out], stddev=0.1), name="w")
                conv = tf.nn.conv2d(ipt,w,strides=[1,1,1,1],padding="SAME",name="conv")
                bn = tf.layers.batch_normalization(
                    inputs=conv,
                    momentum=0.9,
                    training=is_training,
                    name=name+"_bn"
                )
                act = tf.nn.relu(bn,name='relu')
                return act

        def residual_block(ipt, channels, is_training, name):
            with tf.name_scope(name):
                w1 = tf.Variable(tf.truncated_normal([3,3,channels,channels], stddev=0.1), name="w1")
                conv1 = tf.nn.conv2d(ipt,w1, strides=[1,1,1,1], padding="SAME", name='conv1')
                bn1 = tf.layers.batch_normalization(
                    inputs=conv1,
                    momentum=0.9,
                    training=is_training,
                    name=name+"_bn1"
                )
                rl1 = tf.nn.relu(bn1, name="rl1")

                w2 =  tf.Variable(tf.truncated_normal([3,3,channels,channels], stddev=0.1), name="w2")
                conv2 = tf.nn.conv2d(rl1, w2, strides=[1,1,1,1], padding="SAME", name="conv2")
                bn2 = tf.layers.batch_normalization(
                    inputs=conv2,
                    momentum=0.9,
                    training=is_training,
                    name=name+"_bn2"
                )
                skip_conn = bn2 + ipt
                rl2 = tf.nn.relu(skip_conn, name='rl2')

                return rl2

        def linear_fullyconnect(ipt, in_dim, out_dim, name):
            with tf.name_scope(name):
                w = tf.Variable( tf.truncated_normal([in_dim, out_dim],stddev=0.1) , name="w")
                b = tf.Variable( tf.constant(0.1, shape=[out_dim]), name='bias' )
                h_linear = tf.matmul( ipt, w ) + b
                return h_linear



        x_puzzle = tf.reshape(x,[-1, height, width, 1])
        hidden = conv_block(x_puzzle, [3,3], 1, 256, is_training,"conv_block")

        for i in range(1,4):
            hidden = residual_block(hidden, 256, is_training, 'residual_block'+str(i) )

        #output conv
        hidden = conv_block(hidden, [1,1], 256, 1, is_training, 'conv_block_out')

        #fully connect to 256
        hidden_flat = tf.reshape(hidden, [-1, width*height*1])
        h_fc1 = tf.nn.relu( linear_fullyconnect(hidden_flat, width*height*1, 256, 'fully_connect_1') )

        #fully connect to scalar
        y = linear_fullyconnect(h_fc1, 256, 10, 'fully_connect_2')


        lossL2 = tf.add_n([tf.nn.l2_loss(variable) for variable in tf.trainable_variables()
                           if 'bias' not in variable.name]) * 0.0001

        cross_entropy = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
        loss = tf.add(cross_entropy, lossL2, name='total_loss')
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy')

        #train
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_step = tf.train.AdamOptimizer(1e-4).minimize(loss, name='train_mini')

    sess = tf.Session(graph=myGraph)
    assert sess.graph is myGraph
    return sess

if __name__=='__main__':
    sess=build_neuralnetwork(28,28)
    with sess:
        sess.run(tf.global_variables_initializer())
        for i in range(20000):
            batch = mnist.train.next_batch(50)
            if i % 100 == 0:
                train_accuracy = sess.run('accuracy:0', feed_dict={
                    'input_puzzles:0': batch[0], 'labels:0': batch[1], 'is_training:0': False})
                print('step %d, training accuracy %g' % (i, train_accuracy))
            sess.run('train_mini',feed_dict={'input_puzzles:0': batch[0], 'labels:0': batch[1], 'is_training:0': True})
            if i%1000==0 and i>0:
                print('test accuracy %g' % sess.run('accuracy:0', feed_dict={
                    'input_puzzles:0': mnist.test.images, 'labels:0': mnist.test.labels, 'is_training:0': False}))

        print('test accuracy %g' % sess.run('accuracy:0', feed_dict={
            'input_puzzles:0': mnist.test.images, 'labels:0': mnist.test.labels, 'is_training:0': False}))