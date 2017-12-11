# coding: utf8
"""
@author rgtjf

==============
@Update 171211
ADD rnn_type in BiLSTM function

==============
@Update 171211
Version 1.0
"""
import tensorflow as tf
from tensorflow.python.ops import array_ops
from tensorflow.python.framework import ops
from tensorflow.python.util import nest


def fwLSTM(cell_fw, input_x, input_x_len):
    outputs_fw, states_fw = tf.nn.dynamic_rnn(cell=cell_fw, inputs=input_x, sequence_length=input_x_len,
                                              dtype=tf.float32, scope='fw')
    return outputs_fw


def bwLSTM(cell_bw, input_x, input_x_len, time_major=False):
    """
    Ref: bidirectional dynamic rnn in tensorflow r1.0
    https://github.com/tensorflow/tensorflow/blob/r1.0/tensorflow/python/ops/rnn.py#L255-L377
    """
    if not time_major:
        time_dim = 1
        batch_dim = 0
    else:
        time_dim = 0
        batch_dim = 1

    inputs_reverse = array_ops.reverse_sequence(input=input_x, seq_lengths=input_x_len, seq_dim=time_dim,
            batch_dim=batch_dim)
    tmp, output_state_bw = tf.nn.dynamic_rnn(cell=cell_bw, inputs=inputs_reverse, sequence_length=input_x_len,
            dtype=tf.float32, scope='bw')

    outputs_bw = array_ops.reverse_sequence(input=tmp, seq_lengths=input_x_len, seq_dim=time_dim, batch_dim=batch_dim)

    return outputs_bw


def BiLSTM(input_x, input_x_len, hidden_size, num_layers=1, dropout_keep_rate=None, rnn_type='lstm', return_sequence=True):
    """
    BiLSTM Layer
    Args:
        input_x: [batch, sent_len, emb_size]
        input_x_len: [batch, ]
        hidden_size: int
        num_layers: int
        dropout_keep_rate: float
        rnn_type: str, 'lstm'/'gru'
        return_sequence: True/False
    Returns:
        if return_sequence=True:
            outputs: [batch, sent_len, hidden_size*2]
        else:
            output: [batch, hidden_size*2]
    """
    # fix a bug
    # ref: https://stackoverflow.com/questions/44615147/valueerror-trying-to-share-variable-rnn-multi-rnn-cell-cell-0-basic-lstm-cell-k

    def lstm_cell():
        return tf.contrib.rnn.BasicLSTMCell(hidden_size)

    def gru_cell():
        return tf.contrib.rnn.GRUCell(hidden_size)

    if rnn_type == 'lstm':
        rnn_cell = lstm_cell
    elif rnn_type == 'gru':
        rnn_cell = gru_cell
    else:
        raise NotImplementedError

    cell_fw = lstm_cell()
    cell_bw = lstm_cell()

    if num_layers > 1:
        cell_fw = tf.contrib.rnn.MultiRNNCell([rnn_cell() for _ in range(num_layers)])
        cell_bw = tf.contrib.rnn.MultiRNNCell([rnn_cell() for _ in range(num_layers)])

    if dropout_keep_rate:
        cell_fw = tf.contrib.rnn.DropoutWrapper(cell_fw, output_keep_prob=dropout_keep_rate)
        cell_bw = tf.contrib.rnn.DropoutWrapper(cell_bw, output_keep_prob=dropout_keep_rate)

    b_outputs, b_states = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, input_x,
                                                          sequence_length=input_x_len,
                                                          dtype=tf.float32)
    if return_sequence:
        outputs = tf.concat(b_outputs, axis=2)
    else:
        # states: [c, h]
        outputs = tf.concat([b_states[0][1], b_states[1][1]], axis=-1)
    return outputs


def AvgPooling(input_x, input_len):
    """
    Avg_Pooling
    Args:
        input_x: [batch, sent, embedding]
        input_len: [batch, sent]
    Returns:
        [batch, sent_embedding]
    """
    max_input_len = tf.shape(input_x)[1]
    mask = tf.sequence_mask(input_len, max_input_len, dtype=tf.float32)
    norm = mask / (tf.reduce_sum(mask, -1, keep_dims=True) + 1e-6)
    output = tf.reduce_sum(input_x * tf.expand_dims(norm, -1), axis=1)
    return output


def MaxPooling(input_x, input_len):
    """
    Max pooling.
    Args:
        input_x: [batch, max_sent_len, embedding]
        input_len: [batch]
    Returns:
        [batch, sent_embedding]
    """
    max_input_len = tf.shape(input_x)[1]
    mask = tf.sequence_mask(input_len, max_input_len, dtype=tf.float32)
    mask = tf.expand_dims((1 - mask) * -1e30, -1)
    output = tf.reduce_max(input_x + mask, axis=1)

    return output


def CNN_Pooling(inputs, filter_sizes=(1, 2, 3, 5), num_filters=100):
    """
    CNN-MaxPooling
    Args:
        inputs: [batch_size, sequence_length, hidden_size]
        filter_sizes: list, [1, 2, 3, 5]
        num_filters: int, 100
    Returns:
        pool_rep: [batch_size, feature_size]
    """
    sequence_length = inputs.get_shape()[1]
    input_size = inputs.get_shape()[2]
    inputs = tf.expand_dims(inputs, axis=-1)
    pooled_outputs = []
    for i, filter_size in enumerate(filter_sizes):
        with tf.variable_scope("conv-maxpool-%s" % filter_size):
            filter_shape = [filter_size, input_size, 1, num_filters]
            W = tf.get_variable("W", filter_shape, initializer=tf.random_normal_initializer())
            b = tf.get_variable("b", [num_filters], initializer=tf.constant_initializer(0.0))
            conv = tf.nn.conv2d(inputs, W, strides=[1, 1, 1, 1], padding='VALID', name="conv-1")
            h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu-1")
            pooled = tf.nn.max_pool(h, ksize=[1, sequence_length - filter_size + 1, 1, 1], strides=[1, 1, 1, 1],
                    padding='VALID', name="poll-1")
            pooled_outputs.append(pooled)
    num_filters_total = num_filters * len(filter_sizes)
    pooled_reshape = tf.reshape(tf.concat(pooled_outputs, axis=3), [-1, num_filters_total])
    return pooled_reshape


def dot_product_attention(att_vector, sent_vectors, sent_mask):
    """
    Attention dot_product
    Args:
        att_vector: [batch_size, hidden_size]
        sent_vectors: [batch_size, sequence_length, hidden_size]
        sent_mask: [batch_size, sequence_length]
    Returns:
        sent_rep: [batch_size, hidden_size]
    """
    att_vector = tf.expand_dims(att_vector, 1)
    att_prob = softmask(tf.reduce_sum(att_vector * sent_vectors, axis=2), sent_mask)
    sent_rep = tf.reduce_sum(sent_vectors * tf.expand_dims(att_prob, axis=-1), axis=1)
    return sent_rep


def bilinear_attention(att_vector, sent_vectors, sent_mask):
    """
    Attention bilinear
    ref [danqi]:  https://github.com/danqi/rc-cnn-dailymail/blob/master/code/nn_layers.py
    Args:
        att_vector: [batch_size, hidden_size]
        sent_vectors: [batch_size, sequence_length, hidden_size]
        sent_mask: [batch_size, sequence_length]
    Returns:
        passage_rep: [batch_size, hidden_size]
    """
    hidden_size = att_vector.get_shape()[1]
    W_bilinear = tf.get_variable("W_bilinear", shape=[hidden_size, hidden_size], dtype=tf.float32)

    att_vector = tf.matmul(att_vector, W_bilinear)
    att_vector = tf.expand_dims(att_vector, 1)
    alpha = tf.nn.softmax(tf.reduce_sum(att_vector * sent_vectors, axis=2))
    alpha = alpha * sent_mask
    alpha = alpha / tf.reduce_sum(alpha, axis=-1, keep_dims=True)
    sent_rep = tf.reduce_sum(sent_vectors * tf.expand_dims(alpha, axis=-1), axis=1)
    return sent_rep


def softmask(input_prob, input_mask, eps=1e-6):
    """
    normarlize the probability
    Args:
        input_prob: [batch_size, sequence_length]
        input_mask: [batch_size, sequence_length]
        eps:

    Returns:
        [batch_size, sequence]
    """
    input_prob = tf.exp(input_prob, name='exp')
    input_prob = input_prob * input_mask
    input_sum = tf.reduce_sum(input_prob, axis=1, keep_dims=True)
    input_prob /= (input_sum + eps)
    return input_prob


def length(data):
    """
    calculate length, according to zero
    Args:
        data:
    Returns:

    """
    used = tf.sign(tf.reduce_max(tf.abs(data), reduction_indices=2))
    length = tf.reduce_sum(used, reduction_indices=1)
    length = tf.cast(length, tf.int32)
    length_one = tf.ones(tf.shape(length), dtype=tf.int32)
    length = tf.maximum(length, length_one)
    return length


def last_relevant(output, length):
    """
    fetch the last relevant
    """
    batch_size = tf.shape(output)[0]
    max_length = int(output.get_shape()[1])
    output_size = int(output.get_shape()[2])
    index = tf.range(0, batch_size) * max_length + (length - 1)
    flat = tf.reshape(output, [-1, output_size])
    relevant = tf.gather(flat, index)
    return relevant


def gate_mechanism(word_repres, lstm_repres, output_size):
    """
    Gate Mechanism of two representations, such as word and lstm
    Args:
        word_repres: [batch_size, sent_len, dim_1]
        lstm_repres: [batch_size, sent_len, dim_2]
        output_size:
    Returns:
        outputs: [batch_size, sent_len, output_size]
    """
    batch_size = tf.shape(word_repres)[0]
    passage_len = tf.shape(word_repres)[1]
    word_size = tf.shape(word_repres)[2]
    lstm_size = tf.shape(lstm_repres)[2]

    gate_word = tf.get_variable("gate_word", [word_size, output_size], dtype=tf.float32)
    gate_lstm = tf.get_variable("gate_lstm", [lstm_size, output_size], dtype=tf.float32)
    gate_bias = tf.get_variable("gate_bias", [output_size, ], dtype=tf.float32)

    word_repres = tf.reshape(word_repres, [batch_size * passage_len, output_size])
    lstm_repres = tf.reshape(lstm_repres, [batch_size * passage_len, output_size])
    gate = tf.nn.sigmoid(tf.matmul(word_repres, gate_word) + tf.matmul(lstm_repres, gate_lstm) + gate_bias)
    outputs = word_repres * gate + lstm_repres * (1.0 - gate)
    outputs = tf.reshape(outputs, [batch_size, passage_len, output_size])

    return outputs


def kl_loss(preds, golds):
    """
    KL-divergence Loss
    Args:
        preds:
        golds:
    Returns:
        loss
    """
    golds = tf.maximum(1e-6, golds)
    preds = tf.maximum(1e-6, preds)
    loss = golds * (tf.log(golds) - tf.log(preds))
    loss = tf.reduce_sum(loss, axis=-1)
    return loss


def predict_to_score(predicts, num_class):
    """
    Checked: the last is for 0
    ===
    Example:    score=1.2, num_class=3 (for 0-2)
                (0.8, 0.2, 0.0) * (1, 2, 0)
    :param predicts:
    :param num_class:
    :return:
    """
    scores = 0.
    i = 0
    while i < num_class:
        scores += i * predicts[:, i - 1]
        i += 1
    return scores


def optimize(loss, optimize_type, lambda_l2, learning_rate, clipper=50):
    if optimize_type == 'adadelta':
        optimizer = tf.train.AdadeltaOptimizer(learning_rate=learning_rate)
        tvars = tf.trainable_variables()
        l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in tvars if v.get_shape().ndims > 1])
        loss = loss + lambda_l2 * l2_loss
        grads, _ = tf.clip_by_global_norm(tf.gradients(loss, tvars), clipper)
        train_op = optimizer.apply_gradients(list(zip(grads, tvars)))
    elif optimize_type == 'sgd':
        # Create a variable to track the global step.
        global_step = tf.Variable(0, name='global_step', trainable=False)
        min_lr = 0.000001
        _lr_rate = tf.maximum(min_lr, tf.train.exponential_decay(learning_rate, global_step, 30000, 0.98))
        train_op = tf.train.GradientDescentOptimizer(learning_rate=_lr_rate).minimize(loss)
    elif optimize_type == 'ema':
        tvars = tf.trainable_variables()
        train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
        # Create an ExponentialMovingAverage object
        ema = tf.train.ExponentialMovingAverage(decay=0.9999)
        # Create the shadow variables, and add ops to maintain moving averages # of var0 and var1.
        maintain_averages_op = ema.apply(tvars)
        # Create an op that will update the moving averages after each training
        # step.  This is what we will use in place of the usual training op.
        with tf.control_dependencies([train_op]):
            train_op = tf.group(maintain_averages_op)
    elif optimize_type == 'adam':
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        tvars = tf.trainable_variables()
        l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in tvars if v.get_shape().ndims > 1])
        loss = loss + lambda_l2 * l2_loss
        grads, _ = tf.clip_by_global_norm(tf.gradients(loss, tvars), clipper)
        train_op = optimizer.apply_gradients(list(zip(grads, tvars)))

    extra_train_ops = []
    train_ops = [train_op] + extra_train_ops
    train_op = tf.group(*train_ops)
    return train_op


def bilinear_attention_layer(question_repres, question_mask, question_rep,
                             passage_repres, passage_mask, passage_rep,
                             out_size=300, scope=None, reuse=None):
    """
    question_repres: [batch_size, sent_length, dim]
    question_mask  : [batch_size, sent_length]
    question_rep   : [batch_size, dim]
    """
    question_mask = tf.cast(question_mask, tf.float32)
    passage_mask = tf.cast(passage_mask, tf.float32)

    with tf.variable_scope(scope or "attention", reuse=reuse):
        W_bilinear = tf.get_variable("W_bilinear", [out_size, out_size], dtype=tf.float32)
        # W_bilinear_2 = tf.get_variable("W_bilinear_2", [out_size, out_size], dtype=tf.float32)

        question_rep = tf.matmul(question_rep, W_bilinear)
        question_rep = tf.expand_dims(question_rep, 1)

        passage_prob = tf.nn.softmax(tf.reduce_sum(passage_repres * question_rep, 2))
        passage_prob = passage_prob * passage_mask / tf.reduce_sum(passage_mask, -1, keep_dims=True)
        passage_outputs = passage_repres * tf.expand_dims(passage_prob, -1)

        passage_rep = tf.matmul(passage_rep, W_bilinear)
        passage_rep = tf.expand_dims(passage_rep, 1)

        question_prob = tf.nn.softmax(tf.reduce_sum(question_repres * passage_rep, 2))
        question_prob = question_prob * question_mask / tf.reduce_sum(question_mask, -1, keep_dims=True)
        question_outputs = question_repres * tf.expand_dims(question_prob, -1)
    return question_outputs, passage_outputs


def alignment_attention(question_repres, passage_repres, passage_align_mask, output_size, scope=None, reuse=None):
    """
    adopt from Chen, Qin.
    Args:
        question_repres: [batch_size, sent_len_1, hidden_size]
        passage_repres: [batch_size, sent_len_2, hidden_size]
        passage_align_mask: [batch_size, sent_len_2, sent_len_1]
        output_size: for the variable in gate
    Returns:
        [batch_size, sent_len_2, hidden_size]
    """
    with tf.variable_scope(scope or "align_att", reuse=reuse):
        question_repres = tf.expand_dims(question_repres, 1)
        passage_align_mask = tf.expand_dims(passage_align_mask, -1)
        question_repres = tf.reduce_sum(question_repres * passage_align_mask, axis=1)
        passage_repres = gate_mechanism(question_repres, passage_repres, output_size)
    return passage_repres


def linear(args, output_size, bias, bias_start=0.0, scope=None):
    """Linear map: sum_i(args[i] * W[i]), where W[i] is a variable.
    Args:
      args: a 2D Tensor or a list of 2D, batch x n, Tensors.
      output_size: int, second dimension of W[i].
      bias: boolean, whether to add a bias term or not.
      bias_start: starting value to initialize the bias; 0 by default.
      scope: (optional) Variable scope to create parameters in.
    Returns:
      A 2D Tensor with shape [batch x output_size] equal to
      sum_i(args[i] * W[i]), where W[i]s are newly created matrices.
    Raises:
      ValueError: if some of the arguments has unspecified or wrong shape.
    """
    if args is None or (nest.is_sequence(args) and not args):
        raise ValueError("`args` must be specified")
    if not nest.is_sequence(args):
        args = [args]

    # Calculate the total size of arguments on dimension 1.
    total_arg_size = 0
    shapes = [a.get_shape() for a in args]
    for shape in shapes:
        if shape.ndims != 2:
            raise ValueError("linear is expecting 2D arguments: %s" % shapes)
        if shape[1].value is None:
            raise ValueError("linear expects shape[1] to be provided for shape %s, "
                             "but saw %s" % (shape, shape[1]))
        else:
            total_arg_size += shape[1].value

    dtype = [a.dtype for a in args][0]

    # Now the computation.
    with tf.variable_scope(scope) as outer_scope:
        weights = tf.get_variable("weights", [total_arg_size, output_size], dtype=dtype)
        if len(args) == 1:
            res = tf.matmul(args[0], weights)
        else:
            res = tf.matmul(tf.concat(args, 1), weights)
        if not bias:
            return res
        with tf.variable_scope(outer_scope) as inner_scope:
            inner_scope.set_partitioner(None)
            biases = tf.get_variable("biases", [output_size], dtype=dtype,
                    initializer=tf.constant_initializer(bias_start, dtype=dtype))
    return tf.nn.bias_add(res, biases)


class FlipGradientBuilder(object):
    def __init__(self):
        self.num_calls = 0

    def __call__(self, x, l=1.0):
        grad_name = "FlipGradient%d" % self.num_calls

        @ops.RegisterGradient(grad_name)
        def _flip_gradients(op, grad):
            return [tf.negative(grad) * l]

        g = tf.get_default_graph()
        with g.gradient_override_map({"Identity": grad_name}):
            y = tf.identity(x)

        self.num_calls += 1
        return y


flip_gradient = FlipGradientBuilder()