import tensorflow as tf

def main(jobname, taskindex):
    cluster = tf.train.ClusterSpec({"ps": ["localhost:2222"],
                                    "worker" :["localhost:2223","localhost:2224"]
                                    })

    server = tf.train.Server(cluster, job_name=jobname, task_index=taskindex)

    if jobname == "ps":
        server.join()
    elif jobname == "worker":

        # Assigns ops to the local worker by default.
        with tf.device(tf.train.replica_device_setter(
                worker_device="/job:worker/task:%d" % taskindex,
                cluster=cluster)):

            # Build model...
            x=tf.placeholder(tf.float32,shape=(None,2))
            y_=tf.placeholder(tf.float32,shape=(None,))
            w=tf.Variable(tf.truncated_normal([2,1],stddev=0.1))
            b=tf.Variable(tf.constant(0.1,shape=[1]))
            y=tf.matmul(x,w)+b

            loss = (y-y_)*(y-y_)

            train_op = tf.train.AdagradOptimizer(0.01).minimize(
                loss)

        # The StopAtStepHook handles stopping after running given steps.
        hooks = [tf.train.StopAtStepHook(last_step=1000000)]

        # The MonitoredTrainingSession takes care of session initialization,
        # restoring from a checkpoint, saving to a checkpoint, and closing when done
        # or an error occurs.
        with tf.train.MonitoredTrainingSession(master=server.target,
                                               is_chief=(taskindex == 0),
                                               checkpoint_dir="/tmp/train_logs",
                                               hooks=hooks) as mon_sess:
            while not mon_sess.should_stop():
                # Run a training step asynchronously.
                # See `tf.train.SyncReplicasOptimizer` for additional details on how to
                # perform *synchronous* training.
                # mon_sess.run handles AbortedError in case of preempted PS.
                mon_sess.run(train_op)

    else:
        raise TypeError('Unknown job name')

