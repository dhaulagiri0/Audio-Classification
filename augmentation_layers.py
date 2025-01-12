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

class RandomNoise(PreprocessingLayer):

  def __init__(self,
               percentage=0.8,
               factor=0.5,
               seed=None,
               **kwargs):
    super(RandomNoise, self).__init__(**kwargs)

    self.seed = seed
    self.percentage = percentage
    self.factor = factor
    self._rng = make_generator(self.seed)
    self.input_spec = InputSpec(ndim=3)

  def call(self, inputs, training=True):
    if training is None:
      training = backend.learning_phase()

    if self._rng.uniform(shape=()) > self.percentage:
      return inputs  
  
    def noise_fn():
        input = inputs
        noise = tf.random.normal((input.shape[-2], 1))
        # noise = np.random.randn(len(data))
        input = input + self.factor * noise
        return input

    output = control_flow_util.smart_cond(training, noise_fn, lambda: inputs)

    output.set_shape(inputs.shape)
    return output

  def compute_output_shape(self, input_shape):
    return input_shape

  def get_config(self):
    config = {
        'seed': self.seed,
        'percentage': self.percentage,
        'factor': self.factor
    }
    base_config = super().get_config()
    return dict(list(base_config.items()) + list(config.items()))

class RandomTimeShift(PreprocessingLayer):

  def __init__(self,
               percentage=0.8,
               max_shift=22050,
               seed=None,
               **kwargs):
    super(RandomTimeShift, self).__init__(**kwargs)

    self.seed = seed
    self.percentage = percentage
    self.max_shift = max_shift
    self._rng = make_generator(self.seed)
    self.input_spec = InputSpec(ndim=3)

  def call(self, inputs, training=True):
    if training is None:
      training = backend.learning_phase()

    if self._rng.uniform(shape=()) > self.percentage:
      return inputs  

    def time_shift_fn():
        input = inputs
        shift = tf.random.uniform(shape=(), minval=0, maxval=self.max_shift, dtype=tf.int32)
        input = tf.roll(input, shift, axis=-2)
        return input

    output = control_flow_util.smart_cond(training, time_shift_fn, lambda: inputs)

    output.set_shape(inputs.shape)
    return output

  def compute_output_shape(self, input_shape):
    return input_shape

  def get_config(self):
    config = {
        'seed': self.seed,
        'percentage': self.percentage,
        'max_shift': self.max_shift
    }
    base_config = super().get_config()
    return dict(list(base_config.items()) + list(config.items()))

class RandomFreqMask(PreprocessingLayer):

  def __init__(self,
               batch_size,
               percentage=0.1,
               thresh=0.5,
               seed=None,
               **kwargs):
    super(RandomFreqMask, self).__init__(**kwargs)

    self.seed = seed
    self.percentage = percentage
    self.batch_size = batch_size
    self.thresh = thresh
    self._rng = make_generator(self.seed)
    self.input_spec = InputSpec(ndim=4)

  def call(self, inputs, training=True):
    if training is None:
      training = backend.learning_phase()

    if self._rng.uniform(shape=()) > self.thresh:
      return inputs  

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
        freq_max = input.shape[2]

        param = int(self.percentage * freq_max)

        # input = tf.convert_to_tensor(input)
        f = tf.random.uniform(shape=(), minval=0, maxval=param, dtype=tf.dtypes.int32)
        f0 = tf.random.uniform(
            shape=(batch_size, 1, 1, 1), minval=0, maxval=freq_max - f, dtype=tf.dtypes.int32
        )
        indices = tf.reshape(tf.concat([tf.range(freq_max)] * batch_size, axis=-1), (batch_size, 1, -1, 1))
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
        'percentage': self.percentage,
        'thresh': self.thresh
    }
    base_config = super().get_config()
    return dict(list(base_config.items()) + list(config.items()))

class RandomTimeMask(PreprocessingLayer):

  def __init__(self,
               batch_size,
               percentage=0.1,
               thresh=0.5,
               seed=None,
               **kwargs):
    super(RandomTimeMask, self).__init__(**kwargs)

    self.seed = seed
    self.batch_size = batch_size
    self.percentage = percentage
    self.thresh = thresh
    self._rng = make_generator(self.seed)
    self.input_spec = InputSpec(ndim=4)

  def call(self, inputs, training=True):
    if training is None:
      training = backend.learning_phase()

    if self._rng.uniform(shape=()) > self.thresh:
        return inputs  

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
        time_max = input.shape[1]

        param = int(self.percentage * time_max)

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
        'percentage': self.percentage,
        'thresh': self.thresh
    }
    base_config = super().get_config()
    return dict(list(base_config.items()) + list(config.items()))
