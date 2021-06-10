from tensorflow.python.keras.engine.base_preprocessing_layer import PreprocessingLayer
from tensorflow.python.keras.engine import base_preprocessing_layer
from tensorflow.python.keras.utils import control_flow_util
from tensorflow.python.keras import backend
from tensorflow.python.keras.engine.input_spec import InputSpec
import tensorflow as tf
import tensorflow_io as tfio

def make_generator(seed=None):
  """Creates a random generator.
  Args:
    seed: the seed to initialize the generator. If None, the generator will be
      initialized non-deterministically.
  Returns:
    A generator object.
  """
  if seed:
    return tf.random.Generator.from_seed(seed)
  else:
    return tf.random.Generator.from_non_deterministic_state()

class RandomFreqMask(PreprocessingLayer):

  def __init__(self,
               batch_size,
               percentage=0.1,
               seed=None,
               **kwargs):
    super(RandomFreqMask, self).__init__(**kwargs)

    self.seed = seed
    self.percentage = percentage
    self.batch_size = batch_size
    self._rng = make_generator(self.seed)
    self.input_spec = InputSpec(ndim=4)

  def call(self, inputs, training=True):
    if training is None:
      training = backend.learning_phase()

    def freq_mask_vect():
        """
        Apply masking to a spectrogram in the time domain.
        Args:
        input: An audio spectogram.
        param: Parameter of time masking.
        name: A name for the operation (optional).
        Returns:
        A tensor of spectrogram.
        """
        input = inputs
        batch_size = self.batch_size
        freq_max = tf.shape(input)[2]

        param = self.percentage * freq_max

        # input = tf.convert_to_tensor(input)
        f = tf.random.uniform(shape=(), minval=0, maxval=param, dtype=tf.dtypes.int32)
        f0 = tf.random.uniform(
            shape=(batch_size, 1, 1, 1), minval=0, maxval=freq_max - f, dtype=tf.dtypes.int32
        )
        indices = tf.reshape(tf.concat([tf.range(freq_max)] * batch_size, axis=-1), (batch_size, -1, 1, 1))
        condition = tf.math.logical_and(
            tf.math.greater_equal(indices, f0), tf.math.less(indices, f0 + f)
        )
        return tf.where(condition, tf.cast(0, input.dtype), input)

    output = control_flow_util.smart_cond(training, freq_mask_vect,
                                          lambda: inputs)

    output.set_shape(inputs.shape)
    return output

  def compute_output_shape(self, input_shape):
    return input_shape

  def get_config(self):
    config = {
        'batch_size': self.batch_size,
        'seed': self.seed,
        'percentage': self.percentage
    }
    base_config = super(RandomFreqMask, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

class RandomTimeMask(PreprocessingLayer):

  def __init__(self,
               batch_size,
               percentage=0.1,
               seed=None,
               **kwargs):
    super(RandomTimeMask, self).__init__(**kwargs)

    self.seed = seed
    self.batch_size = batch_size
    self.percentage = percentage
    self._rng = make_generator(self.seed)
    self.input_spec = InputSpec(ndim=4)

  def call(self, inputs, training=True):
    if training is None:
      training = backend.learning_phase()

    def time_mask_vect():
        """
        Apply masking to a spectrogram in the time domain.
        Args:
        input: An audio spectogram.
        param: Parameter of time masking.
        name: A name for the operation (optional).
        Returns:
        A tensor of spectrogram.
        """
        input = inputs
        batch_size = self.batch_size
        time_max = tf.shape(input)[1]

        param = self.percentage * time_max

        # input = tf.convert_to_tensor(input)
        t = tf.random.uniform(shape=(), minval=0, maxval=param, dtype=tf.dtypes.int32)
        t0 = tf.random.uniform(
            shape=(batch_size, 1, 1, 1), minval=0, maxval=time_max - t, dtype=tf.dtypes.int32
        )
        indices = tf.reshape(tf.concat([tf.range(time_max)] * batch_size, axis=-1), (batch_size, -1, 1, 1))
        condition = tf.math.logical_and(
            tf.math.greater_equal(indices, t0), tf.math.less(indices, t0 + t)
        )
        return tf.where(condition, tf.cast(0, input.dtype), input)

    output = control_flow_util.smart_cond(training, time_mask_vect,
                                          lambda: inputs)

    output.set_shape(inputs.shape)
    return output

  def compute_output_shape(self, input_shape):
    return input_shape

  def get_config(self):
    config = {
        'batch_size': self.batch_size,
        'seed': self.seed,
        'percentage': self.percentage
    }
    base_config = super(RandomFreqMask, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))
