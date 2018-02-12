# coding: utf8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import math

from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import partitioned_variables
from tensorflow.python.ops import variable_scope as vs

from tensorflow.python.ops.math_ops import sigmoid
from tensorflow.python.ops.math_ops import tanh
from tensorflow.python.ops.rnn_cell_impl import _RNNCell as RNNCell
from tensorflow.contrib.rnn.python.ops.core_rnn_cell_impl import BasicLSTMCell
from tensorflow.contrib.rnn.python.ops.core_rnn_cell_impl import GRUCell

from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import nest

import tensorflow as tf

_LSTMStateTuple = collections.namedtuple("LSTMStateTuple", ("c", "h"))


class LSTMStateTuple(_LSTMStateTuple):
    """Tuple used by LSTM Cells for `state_size`, `zero_state`, and output state.
    Stores two elements: `(c, h)`, in that order.
    Only used when `state_is_tuple=True`.
    """
    __slots__ = ()

    @property
    def dtype(self):
        (c, h) = self
        if not c.dtype == h.dtype:
            raise TypeError("Inconsistent internal state: %s vs %s" % (str(c.dtype), str(h.dtype)))
        return c.dtype


class AttBasicLSTMCell(BasicLSTMCell):
    """Basic Alignment-based LSTM recurrent network cell.

    """
    def __init__(self, attention_vec, attn_activation=tf.nn.tanh, **kwargs):
        super(AttBasicLSTMCell, self).__init__(**kwargs)
        self.attention_vec = attention_vec
        self.attn_activation = attn_activation


    @property
    def state_size(self):
        return (LSTMStateTuple(self._num_units, self._num_units) if self._state_is_tuple
                else 2 * self._num_units)

    @property
    def output_size(self):
        return self._num_units

    def __call__(self, inputs, state, scope=None):
        """Long short-term memory cell (LSTM)."""
        attn_dim = self.attention_vec.get_shape()[-1]
        self.U_a = tf.get_variable("U_a", [self._num_units, self._num_units])
        self.b_a = tf.get_variable("b_a", [self._num_units, ], initializer=tf.constant_initializer(0.0))

        self.U_m = tf.get_variable("U_m", [attn_dim, self._num_units])
        self.b_m = tf.get_variable("b_m", [self._num_units, ], initializer=tf.constant_initializer(0.0))

        self.U_s = tf.get_variable("U_s", [self._num_units, self._num_units])
        self.b_s = tf.get_variable("b_s", [self._num_units, ], initializer=tf.constant_initializer(0.0))

        # matmul
        attention_vec = tf.matmul(self.attention_vec, self.U_m) + self.b_m

        with vs.variable_scope(scope or "basic_lstm_cell"):
            # Parameters of gates are concatenated into one multiply for efficiency.
            if self._state_is_tuple:
                c, h = state
            else:
                c, h = array_ops.split(value=state, num_or_size_splits=2, axis=1)
            concat = _linear([inputs, h], 4 * self._num_units, True, scope="lstm")

            # i = input_gate, j = new_input, f = forget_gate, o = output_gate
            i, j, f, o = array_ops.split(value=concat, num_or_size_splits=4, axis=1)

            new_c = (c * sigmoid(f + self._forget_bias) + sigmoid(i) * self._activation(j))
            new_h = self._activation(new_c) * sigmoid(o)

            m = tanh(tf.matmul(new_h, self.U_a) * attention_vec + self.b_a)
            s = sigmoid(tf.matmul(m, self.U_s) + self.b_s)
            new_h = new_h * s

            if self._state_is_tuple:
                new_state = LSTMStateTuple(new_c, new_h)
            else:
                new_state = array_ops.concat([new_c, new_h], 1)
            return new_h, new_state


class AttGRUCell(GRUCell):

    def __init__(self, attn_vec, attn_activation=tanh, **kwargs):
        super(AttGRUCell, self).__init__(**kwargs)
        self.attn_vec = attn_vec
        self.attn_activation = attn_activation

    @property
    def state_size(self):
        return self._num_units

    @property
    def output_size(self):
        return self._num_units

    def __call__(self, inputs, state, scope=None):
        """Gated recurrent unit (GRU) with nunits cells."""

        attn_dim = self.attn_vec.get_shape()[-1]
        self.U_a = tf.get_variable("U_a", [self._num_units, self._num_units])
        self.b_a = tf.get_variable("b_a", [self._num_units, ], initializer=tf.constant_initializer(0.0))

        self.U_m = tf.get_variable("U_m", [attn_dim, self._num_units])
        self.b_m = tf.get_variable("b_m", [self._num_units, ], initializer=tf.constant_initializer(0.0))

        self.U_s = tf.get_variable("U_s", [self._num_units, self._num_units])
        self.b_s = tf.get_variable("b_s", [self._num_units, ], initializer=tf.constant_initializer(0.0))

        with vs.variable_scope(scope or "gru_cell"):
            with vs.variable_scope("gates"):  # Reset gate and update gate.
                # We start with bias of 1.0 to not reset and not update.
                r, u = array_ops.split(value=_linear([inputs, state], 2 * self._num_units, True, 1.0, scope="ru"),
                        num_or_size_splits=2, axis=1)
                r, u = sigmoid(r), sigmoid(u)
            with vs.variable_scope("candidate"):
                c = self._activation(_linear([inputs, r * state], self._num_units, True, scope="c"))
            new_h = u * state + (1 - u) * c

            # attn_vec
            attn_vec = tf.matmul(self.attn_vec, self.U_m) + self.b_m
            m = tanh(tf.matmul(new_h, self.U_a) * attn_vec + self.b_a)
            s = sigmoid(tf.matmul(m, self.U_s) + self.b_s)

            new_h = new_h * s
        return new_h, new_h


class SRUCell(RNNCell):
    """Simple Recurrent Unit (SRU).
       This implementation is based on: Tao Lei and Yu Zhang, "Training RNNs as Fast as CNNs,"
       https://arxiv.org/abs/1709.02755
    """
    def __init__(self, num_units, activation=None, reuse=None):
        self._num_units = num_units
        self._activation = activation or tf.tanh

    @property
    def output_size(self):
        return self._num_units

    @property
    def state_size(self):
        return self._num_units

    def __call__(self, inputs, state, scope=None):
        """Run one step of SRU."""

        with tf.variable_scope(scope or type(self).__name__):  # "SRUCell"
            with tf.variable_scope("Inputs"):
                x = _linear([inputs], self._num_units, False)
            with tf.variable_scope("Gate"):
                concat = tf.sigmoid(
                    _linear([inputs], 2 * self._num_units, True))
                f, r = tf.split(axis=1, num_or_size_splits=2, value=concat)

            c = f * state + (1 - f) * x

            # highway connection
            h = r * self._activation(c) + (1 - r) * inputs

        return h, c


class BasicAlignLSTMCell(RNNCell):

    def __init__(self, align_repres, align_rep, split_size, num_units, var_dicts, forget_bias=1.0, input_size=None,
                 state_is_tuple=True, activation=tanh):
        """Initialize the basic LSTM cell.
        Args:
          align_repres: tensor3d, [batch_size, answer_sent_size, hidden_size]
          split_size: [emb_size, sent2_size]
          num_units: int, The number of units in the LSTM cell.
          forget_bias: float, The bias added to forget gates (see above).
          input_size: Deprecated and unused.
          state_is_tuple: If True, accepted and returned states are 2-tuples of
            the `c_state` and `m_state`.  If False, they are concatenated
            along the column axis.  The latter behavior will soon be deprecated.
          activation: Activation function of the inner states.
        """
        if not state_is_tuple:
            logging.warn("%s: Using a concatenated state is slower and will soon be "
                         "deprecated.  Use state_is_tuple=True.", self)
        if input_size is not None:
            logging.warn("%s: The input_size parameter is deprecated.", self)
        self._align_repres = align_repres
        self._align_rep = align_rep
        self._split_size = split_size
        self._var_dicts = var_dicts

        self._num_units = num_units
        self._forget_bias = forget_bias
        self._state_is_tuple = state_is_tuple
        self._activation = activation

    @property
    def state_size(self):
        return (LSTMStateTuple(self._num_units, self._num_units) if self._state_is_tuple else 2 * self._num_units)

    @property
    def output_size(self):
        return self._num_units

    def __call__(self, inputs, state, scope=None):
        """Long short-term memory cell (LSTM)."""
        with vs.variable_scope(scope or "basic_lstm_cell"):
            # Parameters of gates are concatenated into one multiply for efficiency.
            if self._state_is_tuple:
                c, h = state
            else:
                c, h = array_ops.split(value=state, num_or_size_splits=2, axis=1)

            # inputs分为两个部分[embedding_size, sent_1_size]
            inputs, align = tf.split(inputs, self._split_size, axis=1)

            concat = _linear([inputs, h], 4 * self._num_units, True, scope=scope)

            # i = input_gate, j = new_input, f = forget_gate, o = output_gate
            i, j, f, o = array_ops.split(value=concat, num_or_size_splits=4, axis=1)

            # align_repres: [batch_size, question_size, hidden_size]
            # align: [batch_size, question_size]
            align_rep = tf.reduce_sum(self._align_repres * tf.expand_dims(align, -1),
                                      axis=1)  # [batch_size, hidden_size]

            new_c = (c * sigmoid(f + self._forget_bias) + sigmoid(i) * self._activation(j))
            new_h = self._activation(new_c) * sigmoid(o)

            g = _gate(new_h, self._align_rep, self._var_dicts)
            new_h = (1 - g) * new_h + g * align_rep

            if self._state_is_tuple:
                new_state = LSTMStateTuple(new_c, new_h)
            else:
                new_state = array_ops.concat([new_c, new_h], 1)
            return new_h, new_state


def _gate(y1, y2, var_dicts):
    w_1 = var_dicts["gate_weights_1"]
    w_2 = var_dicts["gate_weights_2"]
    b = var_dicts["gate_weights_bias"]
    w_y1 = tf.matmul(y1, w_1)
    w_y2 = tf.matmul(y2, w_2)
    g = sigmoid(w_y1 + w_y2 + b)
    return g


def _linear(args, output_size, bias, bias_start=0.0, scope=None):
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
    # scope = vs.get_variable_scope()
    with vs.variable_scope(scope) as outer_scope:
        weights = vs.get_variable("weights", [total_arg_size, output_size], dtype=dtype)
        if len(args) == 1:
            res = math_ops.matmul(args[0], weights)
        else:
            res = math_ops.matmul(array_ops.concat(args, 1), weights)
        if not bias:
            return res
        with vs.variable_scope(outer_scope) as inner_scope:
            inner_scope.set_partitioner(None)
            biases = vs.get_variable("biases", [output_size], dtype=dtype,
                    initializer=init_ops.constant_initializer(bias_start, dtype=dtype))
    return nn_ops.bias_add(res, biases)


def softmask(x, mask, eps=1e-6):
    """

    x: [batch_size, sequence_length]
    mask: [batch_size, sequence_length]
    """
    y = tf.exp(x)
    y = y * mask
    sumx = tf.reduce_sum(y, axis=1, keep_dims=True)
    x = y / (sumx + eps)
    return x


def simple_attention(question_repres):
    pass


def dot_product_attention(question_rep, answer_repres, answer_mask, scope=None):
    pass


def bilinear_attention(question_rep, answer_repres, answer_mask, scope=None):
    pass


def mlp_attention(question_rep, answer_repres, answer_mask, scope=None):
    pass
