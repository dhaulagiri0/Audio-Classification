from unicodedata import name
from tensorflow.keras import layers
from tensorflow.keras.layers import TimeDistributed, LayerNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from tensorflow.python.keras.backend_config import epsilon
from kapre.composed import get_melspectrogram_layer
import tensorflow as tf

# todo: FIX -VE VALS IN MELSPEC
def log_fn(x):
    epsilon = 1e-6
    return tf.math.log(tf.math.abs(x) + epsilon)


def Conv1D(n_classes=10, sr=16000, dt=1.0, **kwargs):
    input_shape = (int(sr*dt), 1)
    i = get_melspectrogram_layer(input_shape=input_shape,
                                 n_mels=128,
                                 pad_end=True,
                                 n_fft=512,
                                 win_length=400,
                                 hop_length=160,
                                 sample_rate=sr,
                                 return_decibel=True,
                                 input_data_format='channels_last',
                                 output_data_format='channels_last')
    x = LayerNormalization(axis=2, name='batch_norm')(i.output)
    x = TimeDistributed(layers.Conv1D(8, kernel_size=(4), activation='tanh'), name='td_conv_1d_tanh')(x)
    x = layers.MaxPooling2D(pool_size=(2,2), name='max_pool_2d_1')(x)
    x = TimeDistributed(layers.Conv1D(16, kernel_size=(4), activation='relu'), name='td_conv_1d_relu_1')(x)
    x = layers.MaxPooling2D(pool_size=(2,2), name='max_pool_2d_2')(x)
    x = TimeDistributed(layers.Conv1D(32, kernel_size=(4), activation='relu'), name='td_conv_1d_relu_2')(x)
    x = layers.MaxPooling2D(pool_size=(2,2), name='max_pool_2d_3')(x)
    x = TimeDistributed(layers.Conv1D(64, kernel_size=(4), activation='relu'), name='td_conv_1d_relu_3')(x)
    x = layers.MaxPooling2D(pool_size=(2,2), name='max_pool_2d_4')(x)
    x = TimeDistributed(layers.Conv1D(128, kernel_size=(4), activation='relu'), name='td_conv_1d_relu_4')(x)
    x = layers.GlobalMaxPooling2D(name='global_max_pooling_2d')(x)
    x = layers.Dropout(rate=0.1, name='dropout')(x)
    x = layers.Dense(64, activation='relu', activity_regularizer=l2(0.001), name='dense')(x)
    o = layers.Dense(n_classes, activation='softmax', name='softmax')(x)
    model = Model(inputs=i.input, outputs=o, name='1d_convolution')
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def ConvDense(n_classes=10, sr=16000, dt=1.0, n_mels=128, spectrogram_width=250, spectrogram_height=128, n_fft=2048, dropout_1=0.2, dropout_2=0.2, n_neurons=1024, l2_lambda=0.001, **kwargs):
    input_shape = (int(sr*dt), 1)
    input_layer = layers.Input(input_shape)

    i1 = get_melspectrogram_layer(input_shape=input_shape,
                                 n_mels=n_mels,
                                 pad_end=True,
                                 n_fft=n_fft,
                                 win_length=int(25 * sr / 1000),
                                 hop_length=int(10 * sr / 1000),
                                 sample_rate=sr,
                                 return_decibel=True,
                                 input_data_format='channels_last',
                                 output_data_format='channels_last',
                                 name='mel1')(input_layer)

    spec1 = layers.Lambda(log_fn)(i1)
    spec1 = layers.experimental.preprocessing.Resizing(spectrogram_width, spectrogram_height)(spec1)
    

    i2 = get_melspectrogram_layer(input_shape=input_shape,
                                n_mels=n_mels,
                                pad_end=True,
                                n_fft=n_fft,
                                win_length=int(50 * sr / 1000),
                                hop_length=int(25 * sr / 1000),
                                sample_rate=sr,
                                return_decibel=True,
                                input_data_format='channels_last',
                                output_data_format='channels_last',
                                name='mel2')(input_layer)

    spec2 = layers.Lambda(log_fn)(i2)
    spec2 = layers.experimental.preprocessing.Resizing(spectrogram_width, spectrogram_height)(spec2)

    i3 = get_melspectrogram_layer(input_shape=input_shape,
                                n_mels=n_mels,
                                pad_end=True,
                                n_fft=n_fft,
                                win_length=int(100 * sr / 1000),
                                hop_length=int(50 * sr / 1000),
                                sample_rate=sr,
                                return_decibel=True,
                                input_data_format='channels_last',
                                output_data_format='channels_last',
                                name='mel3')(input_layer)

    spec3 = layers.Lambda(log_fn)(i3)
    spec3 = layers.experimental.preprocessing.Resizing(spectrogram_width, spectrogram_height)(spec3)

    x = layers.concatenate([spec1, spec2, spec3])
    x = LayerNormalization(axis=2, name='batch_norm')(x)

    densenet = tf.keras.applications.DenseNet201(
        include_top=False, weights='imagenet', input_shape=(spectrogram_width, spectrogram_height, 3), pooling=None
    )

    denseout = densenet(x)
    x = layers.GlobalAveragePooling2D(name='avgpool')(denseout)
    x = layers.Dropout(rate=dropout_1, name='dropout1')(x)
    x = layers.Dense(n_neurons, activation='relu', activity_regularizer=l2(l2_lambda), name='dense')(x)
    x = layers.Dropout(rate=dropout_2, name='dropout2')(x)
    o = layers.Dense(n_classes, activation='softmax', name='softmax')(x)

    model = Model(inputs=input_layer, outputs=o, name='densenet')
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    model.summary()
    return model

def Conv2D(n_classes=10, sr=16000, dt=1.0, **kwargs):
    input_shape = (int(sr*dt), 1)
    i = get_melspectrogram_layer(input_shape=input_shape,
                                 n_mels=128,
                                 pad_end=True,
                                 n_fft=512,
                                 win_length=400,
                                 hop_length=160,
                                 sample_rate=sr,
                                 return_decibel=True,
                                 input_data_format='channels_last',
                                 output_data_format='channels_last')
    x = LayerNormalization(axis=2, name='batch_norm')(i.output)
    x = layers.Conv2D(8, kernel_size=(7,7), activation='tanh', padding='same', name='conv2d_tanh')(x)
    x = layers.MaxPooling2D(pool_size=(2,2), padding='same', name='max_pool_2d_1')(x)
    x = layers.Conv2D(16, kernel_size=(5,5), activation='relu', padding='same', name='conv2d_relu_1')(x)
    x = layers.MaxPooling2D(pool_size=(2,2), padding='same', name='max_pool_2d_2')(x)
    x = layers.Conv2D(16, kernel_size=(3,3), activation='relu', padding='same', name='conv2d_relu_2')(x)
    x = layers.MaxPooling2D(pool_size=(2,2), padding='same', name='max_pool_2d_3')(x)
    x = layers.Conv2D(32, kernel_size=(3,3), activation='relu', padding='same', name='conv2d_relu_3')(x)
    x = layers.MaxPooling2D(pool_size=(2,2), padding='same', name='max_pool_2d_4')(x)
    x = layers.Conv2D(32, kernel_size=(3,3), activation='relu', padding='same', name='conv2d_relu_4')(x)
    x = layers.Flatten(name='flatten')(x)
    x = layers.Dropout(rate=0.2, name='dropout')(x)
    x = layers.Dense(64, activation='relu', activity_regularizer=l2(0.001), name='dense')(x)
    o = layers.Dense(n_classes, activation='softmax', name='softmax')(x)
    model = Model(inputs=i.input, outputs=o, name='2d_convolution')
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def LSTM(n_classes=10, sr=16000, dt=1.0, **kwargs):
    input_shape = (int(sr*dt), 1)
    i = get_melspectrogram_layer(input_shape=input_shape,
                                     n_mels=128,
                                     pad_end=True,
                                     n_fft=512,
                                     win_length=400,
                                     hop_length=160,
                                     sample_rate=sr,
                                     return_decibel=True,
                                     input_data_format='channels_last',
                                     output_data_format='channels_last',
                                     name='2d_convolution')
    x = LayerNormalization(axis=2, name='batch_norm')(i.output)
    x = TimeDistributed(layers.Reshape((-1,)), name='reshape')(x)
    s = TimeDistributed(layers.Dense(64, activation='tanh'),
                        name='td_dense_tanh')(x)
    x = layers.Bidirectional(layers.LSTM(32, return_sequences=True),
                             name='bidirectional_lstm')(s)
    x = layers.concatenate([s, x], axis=2, name='skip_connection')
    x = layers.Dense(64, activation='relu', name='dense_1_relu')(x)
    x = layers.MaxPooling1D(name='max_pool_1d')(x)
    x = layers.Dense(32, activation='relu', name='dense_2_relu')(x)
    x = layers.Flatten(name='flatten')(x)
    x = layers.Dropout(rate=0.2, name='dropout')(x)
    x = layers.Dense(32, activation='relu',
                         activity_regularizer=l2(0.001),
                         name='dense_3_relu')(x)
    o = layers.Dense(n_classes, activation='softmax', name='softmax')(x)
    model = Model(inputs=i.input, outputs=o, name='long_short_term_memory')
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model

