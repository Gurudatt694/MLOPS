import tensorflow as tf
import tensorflow_transform as tft

# Imported files such as taxi_constants are normally cached, so changes are
# not honored after the first import.  Normally this is good for efficiency, but
# during development when we may be iterating code it can be a problem. To
# avoid this problem during development, reload the file.
import taxi_constants
import sys
if 'google.colab' in sys.modules:  # Testing to see if we're doing development
  import importlib
  importlib.reload(taxi_constants)

_NUMERICAL_FEATURES = taxi_constants.NUMERICAL_FEATURES
_BUCKET_FEATURES = taxi_constants.BUCKET_FEATURES
_FEATURE_BUCKET_COUNT = taxi_constants.FEATURE_BUCKET_COUNT
_CATEGORICAL_NUMERICAL_FEATURES = taxi_constants.CATEGORICAL_NUMERICAL_FEATURES
_CATEGORICAL_STRING_FEATURES = taxi_constants.CATEGORICAL_STRING_FEATURES
_VOCAB_SIZE = taxi_constants.VOCAB_SIZE
_OOV_SIZE = taxi_constants.OOV_SIZE
_FARE_KEY = taxi_constants.FARE_KEY
_LABEL_KEY = taxi_constants.LABEL_KEY


def _make_one_hot(x, key):
  """Make a one-hot tensor to encode categorical features.
  Args:
    X: A dense tensor
    key: A string key for the feature in the input
  Returns:
    A dense one-hot tensor as a float list
  """
  integerized = tft.compute_and_apply_vocabulary(x,
          top_k=_VOCAB_SIZE,
          num_oov_buckets=_OOV_SIZE,
          vocab_filename=key, name=key)
  depth = (
      tft.experimental.get_vocabulary_size_by_name(key) + _OOV_SIZE)
  one_hot_encoded = tf.one_hot(
      integerized,
      depth=tf.cast(depth, tf.int32),
      on_value=1.0,
      off_value=0.0)
  return tf.reshape(one_hot_encoded, [-1, depth])

def _fill_in_missing(x):
  """Replace missing values in a SparseTensor.
  Fills in missing values of `x` with '' or 0, and converts to a dense tensor.
  Args:
    x: A `SparseTensor` of rank 2.  Its dense shape should have size at most 1
      in the second dimension.
  Returns:
    A rank 1 tensor where missing values of `x` have been filled in.
  """
  if not isinstance(x, tf.sparse.SparseTensor):
    return x

  default_value = '' if x.dtype == tf.string else 0
  return tf.squeeze(
      tf.sparse.to_dense(
          tf.SparseTensor(x.indices, x.values, [x.dense_shape[0], 1]),
          default_value),
      axis=1)
def preprocessing_fn(inputs):
  """tf.transform's callback function for preprocessing inputs.
  Args:
    inputs: map from feature keys to raw not-yet-transformed features.
  Returns:
    Map from string feature key to transformed feature operations.
  """
  outputs = {}
  for key in _NUMERICAL_FEATURES:
    # If sparse make it dense, setting nan's to 0 or '', and apply zscore.
    outputs[taxi_constants.t_name(key)] = tft.scale_to_z_score(
        _fill_in_missing(inputs[key]), name=key)

  for key in _BUCKET_FEATURES:
    outputs[taxi_constants.t_name(key)] = tf.cast(tft.bucketize(
            _fill_in_missing(inputs[key]), _FEATURE_BUCKET_COUNT, name=key),
            dtype=tf.float32)

  for key in _CATEGORICAL_STRING_FEATURES:
    outputs[taxi_constants.t_name(key)] = _make_one_hot(_fill_in_missing(inputs[key]), key)

  for key in _CATEGORICAL_NUMERICAL_FEATURES:
    outputs[taxi_constants.t_name(key)] = _make_one_hot(tf.strings.strip(
        tf.strings.as_string(_fill_in_missing(inputs[key]))), key)

  # Was this passenger a big tipper?
  taxi_fare = _fill_in_missing(inputs[_FARE_KEY])
  tips = _fill_in_missing(inputs[_LABEL_KEY])
  outputs[_LABEL_KEY] = tf.where(
      tf.math.is_nan(taxi_fare),
      tf.cast(tf.zeros_like(taxi_fare), tf.int64),
      # Test if the tip was > 20% of the fare.
      tf.cast(
          tf.greater(tips, tf.multiply(taxi_fare, tf.constant(0.2))), tf.int64))

  return outputs
