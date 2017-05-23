"""
Code for deep Q-learning as described in:

Playing Atari with Deep Reinforcement Learning
NIPS Deep Learning Workshop 2013

and

Human-level control through deep reinforcement learning.
Nature, 518(7540):529-533, February 2015


Author of Lasagne port: Nissan Pow
Modifications: Nathan Sprague
"""
import tensorflow as tf
import tensorflow.contrib as tfc
import numpy as np


class DeepQLearner:
    """
    Deep Q-learning network using Lasagne.
    """

    def __init__(self, input_width, input_height, num_actions,
                 num_frames, discount, learning_rate, rho,
                 rms_epsilon, momentum, clip_delta, freeze_interval,
                 batch_size, network_type, update_rule,
                 batch_accumulator, rng, input_scale=255.0):

        self.input_width = input_width
        self.input_height = input_height
        self.num_actions = num_actions
        self.num_frames = num_frames
        self.batch_size = batch_size
        self.discount = discount
        self.rho = rho
        self.lr = learning_rate
        self.rms_epsilon = rms_epsilon
        self.momentum = momentum
        self.clip_delta = clip_delta
        self.freeze_interval = freeze_interval
        self.rng = rng
        self.input_scale = input_scale

        self.update_counter = 0
        device = '/gpu:0'
        with tf.device(device):
            self.global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)

            # inference graph-----------------------
            with tf.variable_scope('current'):
                self.images = tf.placeholder(tf.float32, [None, self.num_frames, self.input_height, self.input_width],
                                             name='images')
                self.action_values_given_state = self._inference(self.images)
            with tf.variable_scope('old'):
                self.images_old = tf.placeholder(tf.float32, [None, self.num_frames, self.input_height, self.input_width],
                                                 name='images')
                self.action_values_given_state_old = self._inference(self.images_old)
        self.opt = tf.train.RMSPropOptimizer(self.lr, decay=0.95, epsilon=0.01)
        self._construct_training_graph()
        self._construct_copy_op()

        config = tf.ConfigProto()
        config.log_device_placement = False
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        init = tf.global_variables_initializer()
        self.sess = tf.Session(config=config)
        self.sess.run(init)
        self.reset_q_hat()

    def _construct_training_graph(self):
        with tf.name_scope('training_input'):
            self.actions = tf.placeholder(tf.int32, (None,), 'actions')
            self.rewards = tf.placeholder(tf.float32, (None,), 'rewards')
            self.terminals = tf.placeholder(tf.bool, (None,), 'terminal')
        discount = tf.constant(self.discount, tf.float32, [], 'discount', True)
        with tf.name_scope('diff'):
            targets = self.rewards + (1.0 - tf.cast(self.terminals, tf.float32)) * discount * \
                                     tf.reduce_max(self.action_values_given_state_old, axis=1)
            targets = tf.stop_gradient(targets)
            actions = tf.one_hot(self.actions, self.num_actions, axis=-1, dtype=tf.float32)
            q_s_a = tf.reduce_sum(self.action_values_given_state * actions, axis=1)
            diff = q_s_a - targets
        with tf.name_scope('loss'):
            quad = tf.minimum(tf.abs(diff), 1.0)
            linear = tf.abs(diff) - quad
            loss = 0.5 * tf.square(quad) + linear
            self.loss = tf.reduce_sum(loss)
        self.grad_var_list = self.opt.compute_gradients(self.loss)
        self.apply_gradients = self.opt.apply_gradients(self.grad_var_list, self.global_step)

    def _construct_copy_op(self):
        with tf.name_scope('copy'):
            assign_ops = []
            for (cur, old) in zip(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='current'),
                                tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='old')):
                assert cur.name[7:] == old.name[3:]
                assign_ops.append(tf.assign(old, cur))
            self.copy_cur2old_op = tf.group(*assign_ops)

    def _inference(self, images):
        network_type = 'nature'
        input_height = self.input_height
        input_width = self.input_width
        output_dim = self.num_actions
        channels = self.num_frames
        """
        images batch * channels * height * width
        :param input_width: 84
        :param input_height: 84
        :param output_dim: num_actions
        :param channels: phi_length
        :return: inference layer
        """
        images = images / self.input_scale
        action_values = None
        if network_type == 'linear':
            with tf.variable_scope('linear'):
                images = tf.reshape(images, (-1, channels * input_height * input_width))
                dim = images.get_shape()[1].value
                weights = tf.get_variable('weights', shape=(dim, output_dim), dtype=tf.float32,
                                          initializer=tfc.layers.variance_scaling_initializer(1.0))
                bias = tf.get_variable('bias', shape=(output_dim,), dtype=tf.float32,
                                       initializer=tf.constant_initializer(0.1))
                action_values = tf.add(tf.matmul(images, weights), bias)
        if network_type == 'nature':
            with tf.variable_scope('conv1'):
                size = 8 ; channels = channels ; filters = 32 ; stride = 4
                kernel = tf.get_variable('weights', [size, size, channels, filters], dtype=tf.float32,
                                         initializer=tfc.layers.variance_scaling_initializer(uniform=True))
                conv = tf.nn.conv2d(images, kernel, [1, 1, stride, stride], padding='VALID', data_format='NCHW')
                bias = tf.get_variable('bias', [filters], dtype=tf.float32,
                                       initializer=tf.constant_initializer(0.1))
                pre_activations = tf.nn.bias_add(conv, bias, 'NCHW')
                conv1 = tf.nn.relu(pre_activations)
            with tf.variable_scope('conv2'):
                size = 4 ; channels = 32 ; filters = 64 ; stride = 2
                kernel = tf.get_variable('weights', [size, size, channels, filters], dtype=tf.float32,
                                         initializer=tfc.layers.variance_scaling_initializer(uniform=True))
                conv = tf.nn.conv2d(conv1, kernel, [1, 1, stride, stride], padding='VALID', data_format='NCHW')
                bias = tf.get_variable('bias', [filters], dtype=tf.float32,
                                       initializer=tf.constant_initializer(0.1))
                pre_activations = tf.nn.bias_add(conv, bias, 'NCHW')
                conv2 = tf.nn.relu(pre_activations)
            with tf.variable_scope('conv3'):
                size = 3 ; channels = 64 ; filters = 64 ; stride = 1
                kernel = tf.get_variable('weights', [size, size, channels, filters], dtype=tf.float32,
                                         initializer=tfc.layers.variance_scaling_initializer(uniform=True))
                conv = tf.nn.conv2d(conv2, kernel, [1, 1, stride, stride], padding='VALID', data_format='NCHW')
                bias = tf.get_variable('bias', [filters], dtype=tf.float32,
                                       initializer=tf.constant_initializer(0.1))
                pre_activations = tf.nn.bias_add(conv, bias, 'NCHW')
                conv3 = tf.nn.relu(pre_activations)
            conv3_shape = conv3.get_shape().as_list()
            with tf.variable_scope('linear1'):
                hiddens = 512 ; dim = conv3_shape[1] * conv3_shape[2] * conv3_shape[3]
                reshape = tf.reshape(conv3, [-1, dim])
                weights = tf.get_variable('weights', [dim, hiddens], tf.float32,
                                          initializer=tfc.layers.variance_scaling_initializer(uniform=True))
                bias = tf.get_variable('bias', [hiddens], tf.float32,
                                       initializer=tf.constant_initializer(0.1))
                pre_activations = tf.add(tf.matmul(reshape, weights), bias)
                linear1 = tf.nn.relu(pre_activations)
            with tf.variable_scope('linear2'):
                hiddens = output_dim ; dim = 512
                weights = tf.get_variable('weights', [dim, hiddens], tf.float32,
                                          initializer=tfc.layers.variance_scaling_initializer(uniform=True))
                bias = tf.get_variable('bias', [hiddens], tf.float32,
                                       initializer=tf.constant_initializer(0.1))
                action_values = tf.add(tf.matmul(linear1, weights), bias)
        assert action_values is not None
        return action_values

    def train(self, imgs, actions, rewards, terminals):
        """
        Train one batch.

        Arguments:

        imgs - b x (f + 1) x h x w numpy array, where b is batch size,
               f is num frames, h is height and w is width.
        actions - b x 1 numpy array of integers
        rewards - b x 1 numpy array
        terminals - b x 1 numpy boolean array (currently ignored)

        Returns: average loss
        """

        if (self.freeze_interval > 0 and
            self.update_counter % self.freeze_interval == 0):
            self.reset_q_hat()
        _, loss = self.sess.run([self.apply_gradients, self.loss],
                                feed_dict={self.images: imgs[:, :-1],
                                           self.images_old: imgs[:, 1:],
                                           self.actions: actions,
                                           self.rewards: rewards,
                                           self.terminals: terminals})
        self.update_counter += 1
        return np.sqrt(loss)

    def get_action_values(self, states):
        return self.sess.run(self.action_values_given_state, feed_dict={self.images: states})

    def q_vals(self, state):
        return self.get_action_values(np.expand_dims(state, 0))[0]

    def choose_action(self, state, epsilon):
        if self.rng.rand() < epsilon:
            return self.rng.randint(0, self.num_actions)
        q_vals = self.q_vals(state)
        return np.argmax(q_vals)

    def reset_q_hat(self):
        self.sess.run(self.copy_cur2old_op)
