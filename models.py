from tensorflow.keras import layers
from tensorflow.keras.layers import TimeDistributed, LayerNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from kapre.composed import get_melspectrogram_layer
from augmentation_layers import RandomFreqMask, RandomTimeMask
import tensorflow as tf
from tensorflow.keras.models import Sequential
import tensorflow_hub as hub

def norm_fn(x):
    x = tf.cast(x, tf.float32)
    mins = tf.reduce_min(x, axis=[0, 1])
    maxes = tf.reduce_max(x, axis=[0, 1])
    return 2 * (x - mins) / (maxes - mins) - 1

def mish(inputs):
    return inputs * tf.math.tanh(tf.math.softplus(inputs))

def TriMelspecModel(
    n_classes=13, 
    sr=22050, 
    dt=1.0, 
    backbone='densenet201', 
    n_mels=128, 
    spectrogram_width=250, 
    n_fft=2048, 
    dropout_1=0.2, 
    dropout_2=0.2, 
    dropout_3=0, 
    dropout_4=0, 
    dense_1=1024, 
    dense_2=0, 
    dense_3=0, 
    l2_lambda=0.001, 
    learning_rate=0.001, 
    batch_size=26, 
    mask_pct=0.2, 
    mask_thresh=0.3, 
    activation='relu'):
    
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
        bb = tf.keras.applications.DenseNet201(include_top=False, weights='imagenet', input_shape=(spectrogram_width, n_mels, 3), pooling=None)
    if backbone == 'resnet152':
        bb = tf.keras.applications.ResNet152(include_top=False, weights='imagenet', input_shape=(spectrogram_width, n_mels, 3), pooling=None)
    if backbone == 'densenet169':
        bb = tf.keras.applications.DenseNet169(include_top=False, weights='imagenet', input_shape=(spectrogram_width, n_mels, 3), pooling=None)
    if backbone == 'densenet121':
        bb = tf.keras.applications.DenseNet121(include_top=False, weights='imagenet', input_shape=(spectrogram_width, n_mels, 3), pooling=None)
    if backbone == 'efficientnetb7':
        bb = tf.keras.applications.EfficientNetB7(include_top=False, weights='imagenet', input_shape=(spectrogram_width, n_mels, 3), pooling=None)
    if backbone == 'efficientnetv2-l':
        bb = hub.KerasLayer('gs://cloud-tpu-checkpoints/efficientnet/v2/hub/efficientnetv2-l/feature-vector', trainable=True)

    if backbone == 'efficientnet-gru':
        bb = tf.keras.applications.EfficientNetB7(include_top=False, weights='imagenet', input_shape=(spectrogram_width, n_mels, 3), pooling=None)
        bb = layers.Reshape((-1, bb.output_shape[-1]))(bb(x))
        bb = layers.Bidirectional(layers.GRU(256, return_sequences=True))(bb)
        x = layers.Bidirectional(layers.GRU(256))(bb)
    elif backbone == 'densenet-gru':
        bb = tf.keras.applications.DenseNet201(include_top=False, weights='imagenet', input_shape=(spectrogram_width, n_mels, 3), pooling=None)
        bb = layers.Reshape((-1, bb.output_shape[-1]))(bb(x))
        bb = layers.GRU(256, return_sequences=True)(bb)
        x = layers.GRU(256)(bb)
    else:
        x = bb(x)
        if not backbone == 'efficientnetv2-l':
            # efficientnetv2-l does not need globalaveragepooling as they already do the flatten for us
            x = layers.GlobalAveragePooling2D(name='avgpool')(x)

    if dropout_1 > 0:
        x = layers.Dropout(rate=dropout_1, name='dropout_1')(x)

    if dense_1 > 0:
        if activation == 'mish':
            x = layers.Dense(dense_1, activation=mish, activity_regularizer=l2(l2_lambda), name='dense_1')(x)
        else:
            x = layers.Dense(dense_1, activation=activation, activity_regularizer=l2(l2_lambda), name='dense_1')(x)

    if dropout_2 > 0:
        x = layers.Dropout(rate=dropout_2, name='dropout_2')(x)

    if dense_2 > 0:
        if activation == 'mish':
            x = layers.Dense(dense_2, activation=mish, activity_regularizer=l2(l2_lambda), name='dense_2')(x)
        else:
            x = layers.Dense(dense_2, activation=activation, activity_regularizer=l2(l2_lambda), name='dense_2')(x)

    if dropout_3 > 0:
        x = layers.Dropout(rate=dropout_3, name='dropout_3')(x)
    
    if dense_3 > 0:
        if activation == 'mish':
            x = layers.Dense(dense_3, activation=mish, activity_regularizer=l2(l2_lambda), name='dense_3')(x)
        else:
            x = layers.Dense(dense_3, activation=activation, activity_regularizer=l2(l2_lambda), name='dense_3')(x)
    
    if dropout_4 > 0:
        x = layers.Dropout(rate=dropout_4, name='dropout_4')(x)

    o = layers.Dense(n_classes, activation='softmax', name='softmax')(x)

    model = Model(inputs=input_layer, outputs=o, name='model')
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    model.summary()
    return model