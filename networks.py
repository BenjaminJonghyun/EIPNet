import tensorflow as tf


def mk_variable(name, shape, initializer=None, trainable=None):
    return tf.get_variable(name, shape, initializer=initializer, trainable=trainable)


def conv2d(input, h_k, w_k, h_s, w_s, c_output, name, padding='SAME', batchnorm=False, relu=True, trainable=None):
    c_input = input.get_shape()[-1]

    conv = lambda i, k: tf.nn.conv2d(i, k, [1, h_s, w_s, 1], padding=padding)
    with tf.variable_scope(name) as scope:
        init_weight = tf.contrib.layers.xavier_initializer()
        init_bias = tf.constant_initializer(0.0)

        w = mk_variable('w', [h_k, w_k, c_input, c_output], init_weight, trainable=trainable)
        b = mk_variable('b', [c_output], init_bias, trainable=trainable)

        c = conv(input, w)
        bias = tf.nn.bias_add(c, b)
        if batchnorm:
            bias = tf.layers.batch_normalization(bias, name=scope.name, training=trainable)
        if relu:
            bias = tf.nn.relu(bias, name=scope.name)
    return bias

def deconv2d(input, h_k, w_k, h_s, w_s, c_output, name, padding='SAME', batchnorm=False, relu=True, trainable=None):

    with tf.variable_scope(name) as scope:
        bias = tf.layers.conv2d_transpose(input, c_output, [h_k, w_k], strides=h_s, padding=padding, trainable=trainable)
        if batchnorm:
            bias = tf.layers.batch_normalization(bias, name=scope.name, training=trainable)
        if relu:
            bias = tf.nn.relu(bias, name=scope.name)
    return bias

def edge_conv(input, ksize, strides, name, padding='SAME', trainalbe=None):
    smooth = tf.nn.avg_pool2d(input, ksize=ksize, strides=strides, padding=padding)
    sub = tf.nn.relu(input - smooth)
    edge = conv2d(sub, 1, 1, 1, 1, 1, name, batchnorm=False, trainable=trainalbe)

    return edge, sub

def discriminator(input, reuse=False, trainable=None):
    with tf.variable_scope("discriminator") as scope:
        if reuse:
            scope.reuse_variables()

        di_conv1_1 = conv2d(input, 3, 3, 1, 1, 128, 'di_conv1_1',relu=False, trainable=trainable)
        di_conv1_1 = tf.nn.leaky_relu(di_conv1_1, alpha=0.2)
        di_conv1_2 = conv2d(di_conv1_1, 3, 3, 2, 2, 128, 'di_conv1_2', padding='VALID', relu=False, trainable=trainable)
        di_conv1_2 = tf.nn.leaky_relu(di_conv1_2, alpha=0.2)

        di_conv2_1 = conv2d(di_conv1_2, 3, 3, 1, 1, 256, 'di_conv2_1', relu=False, trainable=trainable)
        di_conv2_1 = tf.nn.leaky_relu(di_conv2_1, alpha=0.2)
        di_conv2_2 = conv2d(di_conv2_1, 3, 3, 2, 2, 256, 'di_conv2_2', padding='VALID', relu=False, trainable=trainable)
        di_conv2_2 = tf.nn.leaky_relu(di_conv2_2, alpha=0.2)

        di_conv3_1 = conv2d(di_conv2_2, 3, 3, 1, 1, 256, 'di_conv3_1', relu=False, trainable=trainable)
        di_conv3_1 = tf.nn.leaky_relu(di_conv3_1, alpha=0.2)
        di_conv3_2 = conv2d(di_conv3_1, 3, 3, 2, 2, 512, 'di_conv3_2', padding='VALID', relu=False, trainable=trainable)
        di_conv3_2 = tf.nn.leaky_relu(di_conv3_2, alpha=0.2)

        di_conv4_1 = conv2d(di_conv3_2, 3, 3, 1, 1, 512, 'di_conv4_1', relu=False, trainable=trainable)

        x = tf.contrib.layers.flatten(di_conv4_1)
        x = tf.layers.dense(x, 512)
        x = tf.nn.leaky_relu(x, alpha=0.2)
        logits = tf.layers.dense(x, 1)
        #x = tf.sigmoid(logits)

    return logits

def generator(input, training = None):
    with tf.variable_scope("generator"):
        Conv1 = conv2d(input, 3, 3, 1, 1, 512, 'Conv1', relu=False, trainable=training)

        Step1_1 = conv2d(Conv1, 3, 3, 1, 1, 512, 'Step1_1', trainable=training)
        Step1_2 = conv2d(Step1_1, 3, 3, 1, 1, 512, 'Step1_2', trainable=training)
        Step1_res = Conv1 + Step1_2
        Step1_de = deconv2d(Step1_res, 4, 4, 2, 2, 256, 'Step1_de', trainable=training)

        Step1_edge, sub1 = edge_conv(Step1_de, ksize=5, strides=1, name='Step1_edge', trainalbe=training)
        Step1_con = tf.concat([Step1_de, sub1], axis=3)

        Step2_1 = conv2d(Step1_con, 3, 3, 1, 1, 512, 'Step2_1', trainable=training)
        Step2_2 = conv2d(Step2_1, 3, 3, 1, 1, 256, 'Step2_2', trainable=training)
        Step2_res = Step1_de + Step2_2
        Step2_de = deconv2d(Step2_res, 4, 4, 2, 2, 128, 'Step2_de', trainable=training)

        Step2_edge, sub2 = edge_conv(Step2_de, ksize=7, strides=1, name='Step2_edge', trainalbe=training)
        Step2_con = tf.concat([Step2_de, sub2], axis=3)

        Step3_1 = conv2d(Step2_con, 3, 3, 1, 1, 256, 'Step3_1', trainable=training)
        Step3_2 = conv2d(Step3_1, 3, 3, 1, 1, 128, 'Step3_2', trainable=training)
        Step3_res = Step2_de + Step3_2
        Step3_de = deconv2d(Step3_res, 4, 4, 2, 2, 64, 'Step3_de', trainable=training)

        Step3_edge, sub3 = edge_conv(Step3_de, ksize=10, strides=1, name='Step3_edge', trainalbe=training)
        Step3_con = tf.concat([Step3_de, sub3], axis=3)

        RGB = conv2d(Step3_con, 1, 1, 1, 1, 3, 'RGB', relu=False, trainable=training)
    return RGB, Step1_edge, Step2_edge, Step3_edge

