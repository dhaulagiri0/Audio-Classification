from tensorflow.keras import layers
from tensorflow.keras.layers import TimeDistributed, LayerNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from kapre.composed import get_melspectrogram_layer, get_stft_magnitude_layer
from kapre.time_frequency import STFT, Magnitude, ApplyFilterbank, MagnitudeToDecibel
from tensorflow.python.keras.backend import dropout
from augmentation_layers import RandomFreqMask, RandomTimeMask
from tensorflow.keras.models import load_model
import tensorflow as tf
from tensorflow.keras.models import Sequential
import tensorflow_hub as hub
from tensorflow.keras import mixed_precision


def norm_fn(x):
    x = tf.cast(x, tf.float32)
    mins = tf.reduce_min(x, axis=[0, 1])
    maxes = tf.reduce_max(x, axis=[0, 1])
    return 2 * (x - mins) / (maxes - mins) - 1

def mish(inputs):
    return inputs * tf.math.tanh(tf.math.softplus(inputs))

def HeadModule(
    input_tensor,
    dropout_1=0.2, 
    dropout_2=0.2, 
    dropout_3=0, 
    dropout_4=0, 
    dense_1=1024, 
    dense_2=0, 
    dense_3=0,
    l2_lambda=0.001,
    activation='relu'):

    x = input_tensor

    if dropout_1 > 0:
        x = layers.Dropout(rate=dropout_1, name='dropout_1_head')(x)

    if dense_1 > 0:
        if activation == 'mish':
            x = layers.Dense(dense_1, activation=mish, activity_regularizer=l2(l2_lambda), name='dense_1_head')(x)
        else:
            x = layers.Dense(dense_1, activation=activation, activity_regularizer=l2(l2_lambda), name='dense_1_head')(x)

    if dropout_2 > 0:
        x = layers.Dropout(rate=dropout_2, name='dropout_2_head')(x)

    if dense_2 > 0:
        if activation == 'mish':
            x = layers.Dense(dense_2, activation=mish, activity_regularizer=l2(l2_lambda), name='dense_2_head')(x)
        else:
            x = layers.Dense(dense_2, activation=activation, activity_regularizer=l2(l2_lambda), name='dense_2_head')(x)

    if dropout_3 > 0:
        x = layers.Dropout(rate=dropout_3, name='dropout_3_head')(x)
    
    if dense_3 > 0:
        if activation == 'mish':
            x = layers.Dense(dense_3, activation=mish, activity_regularizer=l2(l2_lambda), name='dense_3_head')(x)
        else:
            x = layers.Dense(dense_3, activation=activation, activity_regularizer=l2(l2_lambda), name='dense_3_head')(x)
    
    if dropout_4 > 0:
        x = layers.Dropout(rate=dropout_4, name='dropout_4_head')(x)

    return x

def getMelSpecs(input_shape, input, n_fft=2048, sr=22050, spectrogram_width=250, n_mels=128, batch_size=26, mask_pct=0.3, mask_thresh=0.3):
    melspec_head_outputs = []
    win_lengths = [25, 50, 100]
    hop_lengths = [10, 25, 50]
    for i in range(3):
        i = get_melspectrogram_layer(input_shape=input_shape,
                            n_mels=n_mels,
                            pad_end=True,
                            n_fft=n_fft,
                            win_length=int(win_lengths[i] * sr / 1000),
                            hop_length=int(hop_lengths[i] * sr / 1000),
                            sample_rate=sr,
                            return_decibel=True,
                            input_data_format='channels_last',
                            output_data_format='channels_last',
                            name=f'mel{i + 1}')(input)
        i = LayerNormalization(axis=2)(i)
        i_aug = RandomTimeMask(batch_size, mask_pct, mask_thresh)(i)
        i_aug = RandomFreqMask(batch_size, mask_pct, mask_thresh)(i_aug) 
        spec = layers.experimental.preprocessing.Resizing(spectrogram_width, n_mels)(i_aug)
        melspec_head_outputs.append(spec)

    return melspec_head_outputs

def EnsembleModel(
    model_paths, 
    n_classes=13, 
    sr=22050, 
    dt=1.0,         
    l2_lambda=0.001, 
    learning_rate=0.001,
    dropout_1=0.2,
    dropout_2=0.2,
    dropout_3=0.0,
    dropout_4=0.0,
    connector_dense=128,
    dense_1=1024,
    dense_2=0,
    dense_3=0,
    activation='relu'):

    policy = mixed_precision.Policy('mixed_float16')
    mixed_precision.set_global_policy(policy)

    input_shape = (int(sr*dt), 1)
    input_layer = layers.Input(input_shape)
    output_list = []
    KerasLayer = hub.KerasLayer('gs://cloud-tpu-checkpoints/efficientnet/v2/hub/efficientnetv2-l/feature-vector', trainable=True)
    for i, path in enumerate(model_paths):
        model = load_model(path,
        custom_objects={'STFT':STFT,
                        'Magnitude':Magnitude,
                        'ApplyFilterbank':ApplyFilterbank,
                        'MagnitudeToDecibel':MagnitudeToDecibel,
                        'RandomTimeMask': RandomTimeMask,
                        'RandomFreqMask': RandomFreqMask,
                        'mish':mish,
                        'KerasLayer': KerasLayer})
        model._name = f'model{i}'
        model.trainable = False
        output_list.append(layers.Dense(connector_dense, activation=activation)(model(input_layer)))

    x = layers.concatenate(output_list)
    x = HeadModule(x, dropout_1=dropout_1, dropout_2=dropout_2, dropout_3=dropout_3, dropout_4=dropout_4, dense_1=dense_1, dense_2=dense_2, dense_3=dense_3, l2_lambda=l2_lambda, activation=activation)

    o = layers.Dense(n_classes, activation='softmax', activity_regularizer=l2(l2_lambda), name='logits')(x)

    model = Model(inputs=input_layer, outputs=o, name='ensemble_model')
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    model.summary()
    return model

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

    normalized_input = layers.Lambda(norm_fn)(input_layer)

    melspec_head_outputs = getMelSpecs(
                                input_shape, 
                                normalized_input, 
                                n_fft=n_fft, 
                                sr=sr, 
                                spectrogram_width=spectrogram_width, 
                                n_mels=n_mels, 
                                batch_size=batch_size, 
                                mask_pct=mask_pct, 
                                mask_thresh=mask_thresh)

    policy = mixed_precision.Policy('mixed_float16')
    mixed_precision.set_global_policy(policy)

    if backbone == 'trigru':
        gru_outputs = []
        for melspec in melspec_head_outputs:
            reshape1 = layers.Reshape((-1, n_mels))(melspec)
            gru = layers.Bidirectional(layers.GRU(512, return_sequences=True))(reshape1)
            gru = layers.Bidirectional(layers.GRU(256, return_sequences=False))(gru)
            gru_outputs.append(gru)
        
        x = layers.concatenate(gru_outputs)
        x = layers.Flatten()(x)
    else:
        x = layers.concatenate(melspec_head_outputs)

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
            bb = tf.keras.applications.EfficientNetB7(include_top=False, weights='imagenet', input_shape=(spectrogram_width, n_mels, 3), pooling=None)(x)
            bb = layers.Reshape((-1, bb.output_shape[-1]))(bb)
            bb = layers.Bidirectional(layers.GRU(256, return_sequences=True))(bb)
            x = layers.Bidirectional(layers.GRU(256))(bb)
        elif backbone == 'densenet-gru':
            bb = tf.keras.applications.DenseNet201(include_top=False, weights='imagenet', input_shape=(spectrogram_width, n_mels, 3), pooling=None)(x)
            bb = layers.Reshape((-1, bb.output_shape[-1]))(bb)
            bb = layers.GRU(256, return_sequences=True)(bb)
            x = layers.GRU(256)(bb)
        else:
            x = bb(x)
            if not backbone == 'efficientnetv2-l':
                # efficientnetv2-l does not need globalaveragepooling as they already do the flatten for us
                x = layers.GlobalAveragePooling2D(name='avgpool')(x)

    x = HeadModule(x, dropout_1=dropout_1, dropout_2=dropout_2, dropout_3=dropout_3, dropout_4=dropout_4, dense_1=dense_1, dense_2=dense_2, dense_3=dense_3, l2_lambda=l2_lambda, activation=activation)
    o = layers.Dense(n_classes, activation='softmax', name='softmax')(x)

    model = Model(inputs=input_layer, outputs=o, name='model')
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    model.summary()
    return model

def TriSpecModel(
    n_classes=13, 
    sr=22050, 
    dt=1.0,
    backbone='densenet201', 
    spectrogram_width=750, 
    spectrogram_height=512,
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
    activation='relu',
    return_decibel=True):

    input_shape = (int(sr*dt), 1)
    input_layer = layers.Input(input_shape)
    normalized_input = layers.Lambda(norm_fn)(input_layer) # normalize the input audio to -1 to 1 range
    
    i1 = get_stft_magnitude_layer(input_shape=input_shape,
                                n_fft=n_fft,
                                win_length=int(25 * sr / 1000),
                                hop_length=int(10 * sr / 1000),
                                pad_end=True,
                                return_decibel=True if return_decibel == True else False,
                                input_data_format='channels_last',
                                output_data_format='channels_last',
                                name='stftm1')(normalized_input)
    i1 = LayerNormalization(axis=(1,2,3))(i1) # normalize the entire spectrogram to about -1 to 1 range
    i1_aug = RandomTimeMask(batch_size, mask_pct, mask_thresh)(i1)
    i1_aug = RandomFreqMask(batch_size, mask_pct, mask_thresh)(i1_aug) 

    i2 = get_stft_magnitude_layer(input_shape=input_shape,
                                n_fft=n_fft,
                                win_length=int(50 * sr / 1000),
                                hop_length=int(25 * sr / 1000),
                                pad_end=True,
                                return_decibel=True if return_decibel == True else False,
                                input_data_format='channels_last',
                                output_data_format='channels_last',
                                name='stftm2')(normalized_input)
    i2 = LayerNormalization(axis=(1,2,3))(i2) # normalize the entire spectrogram to about -1 to 1 range
    i2_aug = RandomTimeMask(batch_size, mask_pct, mask_thresh)(i2)
    i2_aug = RandomFreqMask(batch_size, mask_pct, mask_thresh)(i2_aug) 

    i3 = get_stft_magnitude_layer(input_shape=input_shape,
                                n_fft=n_fft,
                                win_length=int(100 * sr / 1000),
                                hop_length=int(50 * sr / 1000),
                                pad_end=True,
                                return_decibel=True if return_decibel == True else False,
                                input_data_format='channels_last',
                                output_data_format='channels_last',
                                name='stftm3')(normalized_input)
    i3 = LayerNormalization(axis=(1,2,3))(i3) # normalize the entire spectrogram to about -1 to 1 range
    i3_aug = RandomTimeMask(batch_size, mask_pct, mask_thresh)(i3)
    i3_aug = RandomFreqMask(batch_size, mask_pct, mask_thresh)(i3_aug) 

    policy = mixed_precision.Policy('mixed_float16')
    mixed_precision.set_global_policy(policy)

    if backbone == 'trigru':
        reshape1 = layers.Reshape((-1, int(n_fft/2+1)))(i1_aug)
        gru1 = layers.Bidirectional(layers.GRU(512, return_sequences=True))(reshape1)
        gru1 = layers.Bidirectional(layers.GRU(256, return_sequences=False))(gru1)
        
        reshape2 = layers.Reshape((-1, int(n_fft/2+1)))(i2_aug)
        gru2 = layers.Bidirectional(layers.GRU(512, return_sequences=True))(reshape2)
        gru2 = layers.Bidirectional(layers.GRU(256, return_sequences=False))(gru2)
        
        reshape3 = layers.Reshape((-1, int(n_fft/2+1)))(i3_aug)
        gru3 = layers.Bidirectional(layers.GRU(512, return_sequences=True))(reshape3)
        gru3 = layers.Bidirectional(layers.GRU(256, return_sequences=False))(gru3)
        
        x = layers.concatenate([gru1, gru2, gru3])
        x = layers.Flatten()(x)

    else:
        spec1 = layers.experimental.preprocessing.Resizing(spectrogram_width, spectrogram_height)(i1_aug)
        spec2 = layers.experimental.preprocessing.Resizing(spectrogram_width, spectrogram_height)(i2_aug)
        spec3 = layers.experimental.preprocessing.Resizing(spectrogram_width, spectrogram_height)(i3_aug)

        x = layers.concatenate([spec1, spec2, spec3])

        if backbone == 'densenet201':
            bb = tf.keras.applications.DenseNet201(include_top=False, weights='imagenet', input_shape=(spectrogram_width, spectrogram_height, 3), pooling=None)
        if backbone == 'resnet152':
            bb = tf.keras.applications.ResNet152(include_top=False, weights='imagenet', input_shape=(spectrogram_width, spectrogram_height, 3), pooling=None)
        if backbone == 'densenet169':
            bb = tf.keras.applications.DenseNet169(include_top=False, weights='imagenet', input_shape=(spectrogram_width, spectrogram_height, 3), pooling=None)
        if backbone == 'densenet121':
            bb = tf.keras.applications.DenseNet121(include_top=False, weights='imagenet', input_shape=(spectrogram_width, spectrogram_height, 3), pooling=None)
        if backbone == 'efficientnetb7':
            bb = tf.keras.applications.EfficientNetB7(include_top=False, weights='imagenet', input_shape=(spectrogram_width, spectrogram_height, 3), pooling=None)
        if backbone == 'efficientnetv2-l':
            bb = hub.KerasLayer('gs://cloud-tpu-checkpoints/efficientnet/v2/hub/efficientnetv2-l/feature-vector', trainable=True)
        if backbone == 'efficientnet-gru':
            bb = tf.keras.applications.EfficientNetB7(include_top=False, weights='imagenet', input_shape=(spectrogram_width, spectrogram_height, 3), pooling=None)(x)
            bb = layers.Reshape((-1, bb.output_shape[-1]))(bb)
            bb = layers.Bidirectional(layers.GRU(256, return_sequences=True))(bb)
            x = layers.Bidirectional(layers.GRU(256))(bb)
        elif backbone == 'densenet-gru':
            bb = tf.keras.applications.DenseNet201(include_top=False, weights='imagenet', input_shape=(spectrogram_width, spectrogram_height, 3), pooling=None)(x)
            bb = layers.Reshape((-1, bb.output_shape[-1]))(bb)
            bb = layers.GRU(256, return_sequences=True)(bb)
            x = layers.GRU(256)(bb)
        else:
            x = bb(x)
            if not backbone == 'efficientnetv2-l':
                # efficientnetv2-l does not need globalaveragepooling as they already do the flatten for us
                x = layers.GlobalAveragePooling2D(name='avgpool')(x)

    x = HeadModule(x, dropout_1=dropout_1, dropout_2=dropout_2, dropout_3=dropout_3, dropout_4=dropout_4, dense_1=dense_1, dense_2=dense_2, dense_3=dense_3, l2_lambda=l2_lambda, activation=activation)

    o = layers.Dense(n_classes, activation='softmax', name='softmax')(x)

    model = Model(inputs=input_layer, outputs=o, name='model')
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    model.summary()
    return model

def SingleMelspecUpscaleModel(
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
    normalized_input = layers.Lambda(norm_fn)(input_layer) # normalize the input audio to -1 to 1 range

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
                                name='mel1')(normalized_input)

    policy = mixed_precision.Policy('mixed_float16')
    mixed_precision.set_global_policy(policy)

    i1 = LayerNormalization(axis=2)(i1)
    i1_aug = RandomTimeMask(batch_size, mask_pct, mask_thresh)(i1)
    i1_aug = RandomFreqMask(batch_size, mask_pct, mask_thresh)(i1_aug) 
    spec1 = layers.experimental.preprocessing.Resizing(spectrogram_width, n_mels)(i1_aug)

    x = layers.Conv2D(3, 3, padding='same')(spec1)

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
        bb = tf.keras.applications.EfficientNetB7(include_top=False, weights='imagenet', input_shape=(spectrogram_width, n_mels, 3), pooling=None)(x)
        bb = layers.Reshape((-1, bb.output_shape[-1]))(bb)
        bb = layers.Bidirectional(layers.GRU(256, return_sequences=True))(bb)
        x = layers.Bidirectional(layers.GRU(256))(bb)
    elif backbone == 'densenet-gru':
        bb = tf.keras.applications.DenseNet201(include_top=False, weights='imagenet', input_shape=(spectrogram_width, n_mels, 3), pooling=None)(x)
        bb = layers.Reshape((-1, bb.output_shape[-1]))(bb)
        bb = layers.GRU(256, return_sequences=True)(bb)
        x = layers.GRU(256)(bb)
    else:
        x = bb(x)
        if not backbone == 'efficientnetv2-l':
            # efficientnetv2-l does not need globalaveragepooling as they already do the flatten for us
            x = layers.GlobalAveragePooling2D(name='avgpool')(x)
    
    x = HeadModule(x, dropout_1=dropout_1, dropout_2=dropout_2, dropout_3=dropout_3, dropout_4=dropout_4, dense_1=dense_1, dense_2=dense_2, dense_3=dense_3, l2_lambda=l2_lambda, activation=activation)

    o = layers.Dense(n_classes, activation='softmax', activity_regularizer=l2(l2_lambda), name='logits')(x)

    model = Model(inputs=input_layer, outputs=o, name='SingleMelspecUpscaleModel')
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    model.summary()
    return model
    

def ConvPreWavBlock(x, outchannels):
    x = layers.Conv1D(outchannels, (3, 3), 1, padding='same', use_bias=False, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv1D(outchannels, (3, 3), 1, padding='same', dilation_rate=2, use_bias=False, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPool1D(strides=4)(x)
    return x

def ConvBlock(x, outchannels, pool_size, pool_type):
    x = layers.Conv2D(outchannels, (3, 3), 1, padding='same', use_bias=False, activation='relu')(x)
    x = layers.Conv2D(outchannels, (3, 3), 1, padding='same', use_bias=False, activation='relu')(x)
    if pool_type == 'max':
        x = layers.MaxPool2D(pool_size)(x)
    else:
        x = layers.AveragePooling2D(pool_size)(x)
    return x

def WavegramCNN(
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

    # logmel head    
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

    policy = mixed_precision.Policy('mixed_float16')
    mixed_precision.set_global_policy(policy)

    x = layers.concatenate([spec1, spec2, spec3])
    logmel_out = ConvBlock(x, 64, (2, 2), pool_type='avg') 


    # wavegram head
    x = layers.Conv1D(64, 11, 5, padding='same', use_bias=False, activation='relu')(input_layer)
    x = layers.BatchNormalization()(x)
    # conv block
    x = ConvPreWavBlock(x, 64)
    x = ConvPreWavBlock(x, 128)
    x = ConvPreWavBlock(x, 128)
    x = layers.Reshape((-1, 128, 1))(x)
    x = layers.experimental.preprocessing.Resizing((spectrogram_width, 128, 1))(x)
    wavegram_out = ConvBlock(x, 64, (2, 1), pool_type='max') 

    x = layers.concatenate([wavegram_out, logmel_out])

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

    x = HeadModule(x, dropout_1=dropout_1, dropout_2=dropout_2, dropout_3=dropout_3, dropout_4=dropout_4, dense_1=dense_1, dense_2=dense_2, dense_3=dense_3, l2_lambda=l2_lambda, activation=activation)

    o = layers.Dense(n_classes, activation='softmax', name='softmax')(x)

    model = Model(inputs=input_layer, outputs=o, name='model')
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    model.summary()
    return model