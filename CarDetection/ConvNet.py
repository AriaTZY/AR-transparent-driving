import tensorflow as tf
import numpy as np
# import pylab


def weight_init(shape, in_name='weight_name'):
    fan_in = shape[0] * shape[1] * shape[2]
    fan_out = (shape[3] * shape[0] * shape[1])
    W_bound = (6. / (fan_in + fan_out))
    init_val = tf.random_uniform(shape, -W_bound, W_bound, name=in_name)  # 随机分布，这两个效果都一样
    return tf.Variable(init_val)


def bias_init(shape, in_name='bias_name'):
    init_val = tf.zeros(shape, name=in_name)
    return tf.Variable(init_val)


# 卷积操作
def conv_2d(input_, out_dim, kernel=3, name='conv2d'):
    in_shape = input_.get_shape().as_list()
    with tf.name_scope(name):
        w = weight_init([kernel, kernel, in_shape[-1], out_dim], 'w')
        b = bias_init([out_dim], 'b')
        out = tf.nn.relu((tf.nn.conv2d(input_, w, [1, 1, 1, 1], 'VALID') + b))
    return out


# 最大值池化操作
def max_pooling(input_, kernel=5, name='max_pool'):
    with tf.name_scope(name):
        out = tf.nn.max_pool(input_, [1, kernel, kernel, 1], [1, kernel, kernel, 1], 'VALID')
    return out


# 做全连接操作
def fully_connect(input_, out_dim=10, name='fc_x'):
    shape = input_.get_shape().as_list()
    W_bound = (6. / (shape[1] + out_dim))
    with tf.name_scope(name):
        w_init_val = tf.random_uniform([shape[1], out_dim], -W_bound, W_bound, name='fc_w_init')
        w = tf.Variable(w_init_val)
        b = bias_init([out_dim], 'b')
        out = tf.matmul(input_, w) + b
    return out


class LeNet_5(object):
    def __init__(self, input_size=(40, 60), Train_mode='Train', learning_rate=0.0001, total_epoch=100, batch_size=100):
        self.input_width = input_size[1]
        self.input_height = input_size[0]
        self.batch_size = batch_size
        self.result = 0
        self.init_learning_rate = learning_rate
        self.epochs = total_epoch
        self.Train = Train_mode
        self.fast_data = []

    def train_model(self):
        tf.reset_default_graph()
        origin_input = tf.placeholder(tf.float32, [None, self.input_width*self.input_height])
        input_ = tf.reshape(origin_input, [-1, self.input_height, self.input_width, 1])
        true_val = tf.placeholder(tf.float32, [None, 2])
        conv1 = conv_2d(input_, out_dim=6, kernel=5, name='conv1')  # 输出36*56
        max_pool1 = max_pooling(conv1, kernel=2, name='max_pool1')  # 输出18*28
        conv2 = conv_2d(max_pool1, out_dim=32, kernel=5, name='conv2')  # 输出14*24
        max_pool2 = max_pooling(conv2, kernel=2, name='max_pool2')  # 输出7*12
        conv3 = conv_2d(max_pool2, out_dim=64, kernel=5, name='conv3')  # 输出3*8
        conv4 = conv_2d(conv3, out_dim=128, kernel=3, name='conv4')  # 输出1*6
        # 需要做一些矩阵降维操作
        shape_conv4 = conv4.get_shape().as_list()   # 得到最后一层卷积操作的shape
        in_node = shape_conv4[1]*shape_conv4[2]*shape_conv4[3]  # 除了0维度表示batch_size不需要，conv3的是[10, 1, 1, 120]
        conv_4_vector = tf.reshape(conv4, [-1, in_node])    # 转换成了[10, 768]
        fc_1 = fully_connect(conv_4_vector, out_dim=128, name='fc_1')    # 768->128
        fc_2 = fully_connect(fc_1, out_dim=2, name='fc_2')   # 84->10
        result = tf.nn.softmax(fc_2)

        # 损失函数-cross_entropy
        eps = 1e-10
        y_clip = tf.clip_by_value(result, eps, 1.0 - eps)
        loss = tf.reduce_mean(-tf.reduce_sum(true_val*tf.log(y_clip), reduction_indices=[1]))   # 点乘是内积，不是矩阵乘法
        train = tf.train.AdamOptimizer(self.init_learning_rate).minimize(loss)
        # 准确率
        correct_predition = tf.equal(tf.argmax(result, 1), tf.argmax(true_val, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_predition, tf.float32))
        # 数字形式的输出（最终输出）
        predict_num = tf.argmax(result, 1)

        if self.Train == 'Train':
            # 读入数据
            print('start Training')
            mnist_data_name = 'data/train_data.npy'
            mnist_label_name = 'data/train_label.npy'
            mnist_data = np.load(mnist_data_name)
            mnist_label = np.load(mnist_label_name)
            data_num = mnist_data.shape[0]
            print('共有', data_num, '组训练数据')
            batch_size = data_num//self.batch_size
            iter = 0
            best_loss = 100
            with tf.Session() as sess:
                saver = tf.train.Saver(max_to_keep=1)
                sess.run(tf.global_variables_initializer())
                for epoch in range(self.epochs):
                    for batch_index in range(batch_size):
                        iter = iter + 1
                        batch = self.batch_size
                        sess.run(train, feed_dict={origin_input: mnist_data[batch_index*batch:(batch_index+1)*batch],
                                                   true_val: mnist_label[batch_index*batch:(batch_index+1)*batch]})
                        loss_result = sess.run(loss, feed_dict={
                                origin_input: mnist_data[batch_index * batch:(batch_index + 1) * batch],
                                true_val: mnist_label[batch_index * batch:(batch_index + 1) * batch]})
                        # 更新参数过程
                        if loss_result < best_loss and epoch >= 1:
                            best_loss = loss_result
                            path = 'ckpt/LeNet.ckpt'
                            saver.save(sess, path, global_step=epoch + 1, write_meta_graph=False)
                        # 打印过程
                        if iter % 10 == 0:
                            acc = sess.run(accuracy, feed_dict={
                                origin_input: mnist_data[batch_index * batch:(batch_index + 1) * batch],
                                true_val: mnist_label[batch_index * batch:(batch_index + 1) * batch]})
                            output2 = sess.run(result, feed_dict={
                                origin_input: mnist_data[batch_index * batch:(batch_index + 1) * batch]})
                            print('epoch:%d, iter:%d, loss:%f, accuracy:%f' % (epoch, iter, loss_result, acc))
                            print(output2[10], mnist_label[10], '\n')
        elif self.Train == 'Test':
            # 读入数据
            print('start Test')
            mnist_data_name = 'data/test_data.npy'
            mnist_data = np.load(mnist_data_name)
            data_num = mnist_data.shape[0]
            print('共有', data_num, '组训练数据')
            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
                pick_num = 83
                saver = tf.train.Saver()
                path = 'ckpt/'
                saver.restore(sess, tf.train.latest_checkpoint(path))
                out = sess.run(predict_num, feed_dict={origin_input: mnist_data[:]})
                count = 1
                for i in np.arange(pick_num, pick_num + 60, 1):
                    pylab.subplot(6, 10, count)
                    show_pic = np.reshape(mnist_data[i], [40, 60])
                    pylab.imshow(show_pic, 'gray')
                    title = 'num=' + str(out[i])
                    pylab.axis('off')  # 关闭坐标轴
                    pylab.title(title)
                    count = count + 1
                print('估计值：', out[0:10])
                pylab.show()
                return out

        elif self.Train == 'Fast':
            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
                saver = tf.train.Saver()
                path = 'ckpt/'
                saver.restore(sess, tf.train.latest_checkpoint(path))
                out = sess.run(result, feed_dict={origin_input: self.fast_data[:]})
                return out

    def visual_check(self):
        print('start show')
        mnist_data_name = 'mnist_data_set.npy'
        mnist_label_name = 'mnist_label_list.npy'
        mnist_data = np.load(mnist_data_name)
        mnist_label = np.load(mnist_label_name)
        count = 1
        for i in np.arange(10, 20, 1):
            pylab.subplot(3, 4, count)
            show_pic = np.reshape(mnist_data[i], [28, 28])
            pylab.imshow(show_pic, 'gray')
            print(mnist_label[i])
            count = count + 1
        pylab.show()



def test_sample():
    class_mnist_test = LeNet_5(32, Train_mode='Test')
    return class_mnist_test.train_model()


if __name__ == '__main__':
    class_mnist = LeNet_5((60, 40), Train_mode='Train', learning_rate=0.001)
    class_mnist.train_model()
