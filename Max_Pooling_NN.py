# These are all the modules we'll be using later. Make sure you can import them
# before proceeding further.
from __future__ import print_function
import numpy as np
import tensorflow as tf
import tensorlayer as tl
from six.moves import range
import sys

# Loading the data from MNIST dataset
train_dataset, train_labels, \
    valid_dataset, valid_labels, \
    test_dataset, test_labels = \
    tl.files.load_mnist_dataset(shape=(-1, 28, 28, 1))


image_size = 28
num_labels = 10
num_channels = 1  # grayscale


def reformat(labels):
    return (np.arange(num_labels) == labels[:, None]).astype(np.float32)


train_labels = reformat(train_labels)
valid_labels = reformat(valid_labels)
test_labels = reformat(test_labels)
print('Training set', train_dataset.shape, train_labels.shape)
print('Validation set', valid_dataset.shape, valid_labels.shape)
print('Test set', test_dataset.shape, test_labels.shape)


def accuracy(predictions, labels):
    return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
            / predictions.shape[0])


batch_size = 800
patch_size = 5
depth = 24
num_hidden = 128

graph = tf.Graph()

with graph.as_default():
    # Input data.
    tf_train_dataset = tf.placeholder(
        tf.float32, shape=(batch_size, image_size, image_size, num_channels))
    tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
    tf_valid_dataset = tf.constant(valid_dataset)
    tf_test_dataset = tf.constant(test_dataset)
    
    # Variables.
    layer1_weights = tf.Variable(tf.truncated_normal(
        [patch_size, patch_size, num_channels, depth], stddev=0.02))
    layer1_biases = tf.Variable(tf.zeros([depth]))
    
    layer2_weights = tf.Variable(tf.truncated_normal(
        [patch_size, patch_size, depth, 2*depth], stddev=0.02))
    layer2_biases = tf.Variable(tf.constant(1.0, shape=[2*depth]))
    
    layer3_weights = tf.Variable(tf.truncated_normal(
        [image_size // 4 * image_size // 4 * 2*depth, num_hidden], stddev=0.02))
    layer3_biases = tf.Variable(tf.constant(1.0, shape=[num_hidden]))
    layer4_weights = tf.Variable(tf.truncated_normal(
        [num_hidden, num_labels], stddev=0.02))
    layer4_biases = tf.Variable(tf.constant(1.0, shape=[num_labels]))

    global_step = tf.Variable(0)  # count the number of steps taken.
    learning_rate = tf.train.exponential_decay(0.08, global_step, 8000, 0.8)
    drpt_keep_rate = tf.train.exponential_decay(0.98, global_step, 8000, 0.98)


    # Model with Pooling and dropout
    def model_pool_dropout(data, if_dropout=True):
        
        conv = tf.nn.relu(tf.nn.conv2d(data, layer1_weights,
                            [1, 1, 1, 1], padding='SAME') + layer1_biases)
        
        # Max Pooling Layer 1
        pooling = tf.nn.max_pool(conv, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')
        
        if if_dropout:
            hidden = tf.nn.dropout(pooling, drpt_keep_rate)
        else:
            hidden = pooling
        
        conv = tf.nn.relu(tf.nn.conv2d(hidden, layer2_weights,
                            [1, 1, 1, 1], padding='SAME') + layer2_biases)
        
        # Max Pooling Layer 2
        pooling = tf.nn.max_pool(conv, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')
        
        if if_dropout:
            hidden = tf.nn.dropout(pooling, drpt_keep_rate)
        else:
            hidden = pooling
            
        shape = hidden.get_shape().as_list()
        reshape = tf.reshape(hidden, [shape[0], shape[1] * shape[2] * shape[3]])
        if if_dropout:
            hidden = tf.nn.dropout(tf.nn.relu(tf.matmul(reshape, 
                        layer3_weights) + layer3_biases), drpt_keep_rate)
        else:
            hidden = tf.nn.relu(tf.matmul(reshape, 
                        layer3_weights) + layer3_biases)
        return tf.matmul(hidden, layer4_weights) + layer4_biases


    # Training computation.
    logits = model_pool_dropout(tf_train_dataset)
    loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits_v2(labels=tf_train_labels, logits=logits))

    # Optimizer.
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)

    # Predictions for the training, validation, and test data.
    train_prediction = tf.nn.softmax(logits)
    valid_prediction = tf.nn.softmax(model_pool_dropout(tf_valid_dataset, False))
    test_prediction = tf.nn.softmax(model_pool_dropout(tf_test_dataset, False))
    saver = tf.train.Saver()

num_steps = 45001

with tf.Session(graph=graph) as session:
    tf.global_variables_initializer().run()
    log_file_name = 'MaxPoolNN_log.log'
    model_file_name = './MaxPoolNN_model.ckpt'
    
    log_file = open(log_file_name, 'a')
    print('Initialized',file=log_file)
    log_file.close()
    print('Initialized')
    for step in range(num_steps):
        offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
        batch_data = train_dataset[offset:(offset + batch_size), :, :, :]
        batch_labels = train_labels[offset:(offset + batch_size), :]
        feed_dict = {tf_train_dataset: batch_data, tf_train_labels: batch_labels}
        _, l, predictions = session.run(
            [optimizer, loss, train_prediction], feed_dict=feed_dict)
        if (step % 100 == 0):
            log_file = open(log_file_name, 'a')
            print('Minibatch loss at step %d: %f' % (step, l),file=log_file)
            minibatch_accuracy = accuracy(predictions, batch_labels)
            val_accuracy = accuracy(valid_prediction.eval(), valid_labels)
            print('Minibatch accuracy: %.2f%%' % minibatch_accuracy, file=log_file)
            print('Validation accuracy: %.2f%%' % val_accuracy, file=log_file)
            log_file.close()
            if (step % 200 == 0):
                print('Minibatch loss at step %d: %f' % (step, l))
                print('Minibatch accuracy: %.2f%%' % minibatch_accuracy)
                print('Validation accuracy: %.2f%%' % val_accuracy)
                sys.stdout.flush()

    try:
        save_path = saver.save(session, model_file_name)
    except:
        print("could not save the model")
    log_file = open(log_file_name, 'a')
    test_accuracy = accuracy(test_prediction.eval(), test_labels)
    try:
        print('Test accuracy: %.2f%%' % test_accuracy, file=log_file)
        print('Test accuracy: %.2f%%' % test_accuracy)
    except:
        print('Could not calculate the test accuracy')
    log_file.close()
