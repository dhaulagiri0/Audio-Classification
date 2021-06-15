from tensorflow.keras import layers
from tensorflow.keras.layers import TimeDistributed, LayerNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from kapre.composed import get_melspectrogram_layer
from augmentation_layers import RandomFreqMask, RandomTimeMask
import tensorflow as tf
from tensorflow.keras.models import Sequential

def norm_fn(x):
    x = tf.cast(x, tf.float32)
    mins = tf.reduce_min(x, axis=[0, 1])
    maxes = tf.reduce_max(x, axis=[0, 1])
    # return 2 * (x - mins) / (maxes - mins) - 1
    return x

def mish(inputs):
    return inputs * tf.math.tanh(tf.math.softplus(inputs))

def TriMelspecModel(n_classes=10, sr=16000, dt=1.0, backbone='densenet201', n_mels=128, spectrogram_width=250, n_fft=2048, dropout_1=0.2, dropout_2=0.2, dropout_3=0.15, dropout_4=0.1, dense_1=1024, dense_2=512, dense_3=256, l2_lambda=0.001, learning_rate=0.001, batch_size=15, mask_pct=0.2, mask_thresh=0.3, activation='relu', **kwargs):
    input_shape = (int(sr*dt), 1)
    input_layer = layers.Input(input_shape)
    # normalized_input = layers.Lambda(norm_fn)(input_layer)
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
    i1 = LayerNormalization(axis=2)(i1)
    i1_aug = RandomTimeMask(batch_size, mask_pct, mask_thresh)(i1)
    i1_aug = RandomFreqMask(batch_size, mask_pct, mask_thresh)(i1_aug) 
    spec1 = layers.experimental.preprocessing.Resizing(spectrogram_width, n_mels)(i1_aug)
    
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
    i2 = LayerNormalization(axis=2)(i2)
    i2_aug = RandomTimeMask(batch_size, mask_pct, mask_thresh)(i2)
    i2_aug = RandomFreqMask(batch_size, mask_pct, mask_thresh)(i2_aug) 
    spec2 = layers.experimental.preprocessing.Resizing(spectrogram_width, n_mels)(i2_aug)

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
    i3 = LayerNormalization(axis=2)(i3)
    i3_aug = RandomTimeMask(batch_size, mask_pct, mask_thresh)(i3)
    i3_aug = RandomFreqMask(batch_size, mask_pct, mask_thresh)(i3_aug) 
    spec3 = layers.experimental.preprocessing.Resizing(spectrogram_width, n_mels)(i3_aug)

    x = layers.concatenate([spec1, spec2, spec3])

    if backbone == 'densenet201':
        bb = tf.keras.applications.DenseNet201(
            include_top=False, weights='imagenet', input_shape=(spectrogram_width, n_mels, 3), pooling=None
        )
    if backbone == 'resnet152':
        bb = tf.keras.applications.ResNet152(
            include_top=False, weights='imagenet', input_shape=(spectrogram_width, n_mels, 3), pooling=None
        )
    if backbone == 'densenet169':
        bb = tf.keras.applications.DenseNet169(
            include_top=False, weights='imagenet', input_shape=(spectrogram_width, n_mels, 3), pooling=None
        )
    if backbone == 'densenet121':
        bb = tf.keras.applications.DenseNet121(
            include_top=False, weights='imagenet', input_shape=(spectrogram_width, n_mels, 3), pooling=None
        )

    bb_out = bb(x)
    x = layers.GlobalAveragePooling2D(name='avgpool')(bb_out)
    x = layers.Dropout(rate=dropout_1, name='dropout_1')(x)
    if activation == 'mish':
        x = layers.Dense(dense_1, activation=mish, activity_regularizer=l2(l2_lambda), name='dense_1')(x)
    else:
        x = layers.Dense(dense_1, activation=activation, activity_regularizer=l2(l2_lambda), name='dense_1')(x)
    x = layers.Dropout(rate=dropout_2, name='dropout_2')(x)
    if activation == 'mish':
        x = layers.Dense(dense_2, activation=mish, activity_regularizer=l2(l2_lambda), name='dense_2')(x)
    else:
        x = layers.Dense(dense_2, activation=activation, activity_regularizer=l2(l2_lambda), name='dense_12')(x)
    x = layers.Dropout(rate=dropout_3, name='dropout_3')(x)
    # x = layers.Dense(dense_3, activation=mish, activity_regularizer=l2(l2_lambda), name='dense_3')(x)
    # x = layers.Dropout(rate=dropout_4, name='dropout_4')(x)
    o = layers.Dense(n_classes, activation='softmax', name='softmax')(x)

    model = Model(inputs=input_layer, outputs=o, name='model')
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


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

