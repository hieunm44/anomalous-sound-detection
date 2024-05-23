import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Dense, Activation, Input, Lambda, Conv2D, Conv2DTranspose, Reshape, TimeDistributed, BatchNormalization
from keras.models import Sequential, Model, load_model
from keras.callbacks import EarlyStopping
from keras.losses import mse
from sklearn.metrics import mean_squared_error, roc_auc_score
import utils
import keras.backend as K
K.set_image_data_format("channels_first")
from tensorflow.python.framework.ops import disable_eager_execution
disable_eager_execution()


def visualize(target_dir):
    machine_type = target_dir.split('/')[-2]
    machine_id = target_dir.split('/')[-1]
    print(f'{machine_type}_{machine_id}')
    train_data = np.load(f'feat/train_files_{machine_type}_{machine_id}.npz')['arr_0']
    train_labels = np.load(f'feat/train_labels_{machine_type}_{machine_id}.npz')['arr_0']
    eval_data = np.load(f'feat/eval_files_{machine_type}_{machine_id}.npz')['arr_0']
    eval_labels = np.load(f'feat/eval_labels_{machine_type}_{machine_id}.npz')['arr_0']
    len_audio = int(train_data.shape[0]/train_labels.shape[0])

    plt.figure(figsize=(300, 3))
    plt.imshow(eval_data.T, cmap=plt.cm.jet, aspect='auto')
    ax = plt.gca()
    ax.invert_yaxis()
    xticks_value = np.arange(0, eval_data.shape[0], len_audio)
    xticks_label = np.arange(0, eval_labels.shape[0])
    plt.xticks(xticks_value, xticks_label)

    plt.show()


def plot_losses(train_loss, val_loss):
    plt.plot(train_loss, label='Training loss')
    plt.plot(val_loss, label='Validation loss')
    plt.legend()
    plt.title('Training process')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.xticks(np.arange(len(train_loss)), [str(i) for i in np.arange(len(train_loss))])
    plt.show()


def plot_re(re):
    plt.figure(figsize=(45, 9))
    plt.plot(re, marker='o')
    plt.title('Reconstruction loss')
    plt.xlabel('sample')
    plt.ylabel('loss')
    plt.show()


def AE_model(data_in):
    model = Sequential()
    model.add(Dense(32, input_shape=(data_in.shape[1],)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dense(16))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dense(8))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dense(4))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dense(2))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dense(4))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dense(8))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dense(16))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dense(32))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dense(data_in.shape[1]))
    model.add(BatchNormalization())
    model.add(Activation('sigmoid'))

    model.compile(loss='mse', optimizer='adam')
    model.summary()
    return model


def CAE_model(data_in):
    model = Sequential()
    input_shape = (data_in.shape[-3], data_in.shape[-2], data_in.shape[-1])
    # Encoder
    model.add(Conv2D(32, kernel_size=(1, 3), input_shape=input_shape))
    model.add(BatchNormalization(axis=1))
    model.add(Activation('relu'))
    model.add(Conv2D(32, kernel_size=(1, 3)))
    model.add(BatchNormalization(axis=1))
    model.add(Activation('relu'))
    model.add(Conv2D(32, kernel_size=(1, 3)))
    model.add(BatchNormalization(axis=1))
    model.add(Activation('relu'))

    # Decoder
    model.add(Conv2DTranspose(32, kernel_size=(1, 3)))
    model.add(BatchNormalization(axis=1))
    model.add(Activation('relu'))
    model.add(Conv2DTranspose(32, kernel_size=(1, 3)))
    model.add(BatchNormalization(axis=1))
    model.add(Activation('relu'))
    model.add(Conv2DTranspose(32, kernel_size=(1, 3)))
    model.add(BatchNormalization(axis=1))
    model.add(Activation('relu'))
    model.add(Conv2D(1, kernel_size=(1, 3), padding='same'))
    model.add(BatchNormalization(axis=1))
    model.add(Activation('sigmoid'))

    model.compile(loss='mse', optimizer='adam')
    model.summary()
    return model


def train_AE(target_dir):
    machine_type = target_dir.split('/')[-2]
    machine_id = target_dir.split('/')[-1]
    print(f'{machine_type}_{machine_id}')
    train_data = np.load(f'feat/train_files_{machine_type}_{machine_id}.npz')['arr_0']
    model = AE_model(train_data)
    callback = EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=10, mode='min')
    hist = model.fit(train_data,
                     train_data,
                     epochs=5,
                     batch_size=128,
                     shuffle=True,
                     validation_split=0.1,
                     verbose=1,
                     callbacks=[callback])
    train_loss = hist.history['loss']
    val_loss = hist.history['val_loss']
    plot_losses(train_loss, val_loss)
    model.save(f'models/AE_{machine_type}_{machine_id}.h5')


def train_VAE(target_dir):
    machine_type = target_dir.split('/')[-2]
    machine_id = target_dir.split('/')[-1]
    print(f'{machine_type}_{machine_id}')
    train_data = np.load(f'feat/train_files_{machine_type}_{machine_id}.npz')['arr_0']

    def sampling(args):
        z_mean, z_log_sigma = args
        epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim), mean=0., stddev=1.0)
        return z_mean + K.exp(z_log_sigma) * epsilon

    x = Input(shape=(train_data.shape[1],))
    h = Dense(32)(x)
    h = BatchNormalization()(h)
    h = Activation('relu')(h)
    h = Dense(16)(h)
    h = BatchNormalization()(h)
    h = Activation('relu')(h)
    h = Dense(8)(h)
    h = BatchNormalization()(h)
    h = Activation('relu')(h)
    h = Dense(4)(h)
    h = BatchNormalization()(h)
    h = Activation('relu')(h)

    latent_dim = 2
    z_mean = Dense(latent_dim)(h)
    z_log_var = Dense(latent_dim)(h)
    z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])

    h_decoded = Dense(4)(z)
    h_decoded = BatchNormalization()(h_decoded)
    h_decoded = Activation('relu')(h_decoded)
    h_decoded = Dense(8)(h_decoded)
    h_decoded = BatchNormalization()(h_decoded)
    h_decoded = Activation('relu')(h_decoded)
    h_decoded = Dense(16)(h_decoded)
    h_decoded = BatchNormalization()(h_decoded)
    h_decoded = Activation('relu')(h_decoded)
    h_decoded = Dense(32)(h_decoded)
    h_decoded = BatchNormalization()(h_decoded)
    h_decoded = Activation('relu')(h_decoded)
    x_decoded = Dense(train_data.shape[1])(h_decoded)

    vae = Model(x, x_decoded)

    def vae_loss(x, x_decoded):
        xent_loss = mse(x, x_decoded)
        kl_loss = -0.0005*K.mean(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
        return xent_loss + kl_loss

    vae.compile(optimizer='adam', loss=vae_loss)
    vae.summary()

    callback = EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=10, mode='min')
    hist = vae.fit(train_data,
                   train_data,
                   epochs=2,
                   batch_size=128,
                   shuffle=True,
                   validation_split=0.1,
                   verbose=1,
                   callbacks=[callback])
    train_loss = hist.history['loss']
    val_loss = hist.history['val_loss']
    plot_losses(train_loss, val_loss)
    vae.save(f'models/VAE_{machine_type}_{machine_id}.h5')


def train_CAE(target_dir):
    machine_type = target_dir.split('/')[-2]
    machine_id = target_dir.split('/')[-1]
    print(f'{machine_type}_{machine_id}')
    train_data = np.load(f'feat/train_files_{machine_type}_{machine_id}.npz')['arr_0']
    train_labels = np.load(f'feat/train_labels_{machine_type}_{machine_id}.npz')['arr_0']
    len_audio = int(train_data.shape[0]/train_labels.shape[0])
    nb_ch = 1
    train_data = utils.preprocess_data(train_data, len_audio, nb_ch)
    model = CAE_model(train_data)
    callback = EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=10, mode='min')
    hist = model.fit(train_data,
                     train_data,
                     epochs=2,
                     batch_size=128,
                     shuffle=True,
                     validation_split=0.1,
                     verbose=1,
                     callbacks=[callback])
    train_loss = hist.history['loss']
    val_loss = hist.history['val_loss']
    plot_losses(train_loss, val_loss)
    model.save(f'models/CAE_{machine_type}_{machine_id}.h5')


def train_CVAE(target_dir):
    machine_type = target_dir.split('/')[-2]
    machine_id = target_dir.split('/')[-1]
    print(f'{machine_type}_{machine_id}')
    train_data = np.load(f'feat/train_files_{machine_type}_{machine_id}.npz')['arr_0']
    train_labels = np.load(f'feat/train_labels_{machine_type}_{machine_id}.npz')['arr_0']
    len_audio = int(train_data.shape[0]/train_labels.shape[0])
    nb_ch = 1
    train_data = utils.preprocess_data(train_data, len_audio, nb_ch)

    def sampling(args):
        z_mean, z_log_sigma = args
        epsilon = K.random_normal(shape=(K.shape(z_mean)[1], latent_dim), mean=0., stddev=1.0)
        return z_mean + K.exp(z_log_sigma) * epsilon

    input_shape = (train_data.shape[-3],
                   train_data.shape[-2], train_data.shape[-1])
    x = Input(shape=input_shape)
    h = Conv2D(32, kernel_size=(3, 3), padding='same')(x)
    h = BatchNormalization(axis=1)(h)
    h = Activation('relu')(h)
    h = Conv2D(32, kernel_size=(3, 3), padding='same')(h)
    h = BatchNormalization(axis=1)(h)
    h = Activation('relu')(h)
    h = Conv2D(32, kernel_size=(3, 3), padding='same')(h)
    h = BatchNormalization(axis=1)(h)
    h = Activation('relu')(h)
    h = Conv2D(1, kernel_size=(3, 3), padding='same')(h)
    h = Reshape((313, 64))(h)

    h = TimeDistributed(Dense(32))(h)
    h = BatchNormalization()(h)
    h = Activation('relu')(h)
    h = TimeDistributed(Dense(16))(h)
    h = BatchNormalization()(h)
    h = Activation('relu')(h)
    # h = TimeDistributed(Dense(8))(h)
    # h = BatchNormalization()(h)
    # h = Activation('relu')(h)
    # h = TimeDistributed(Dense(4))(h)
    # h = BatchNormalization()(h)
    # h = Activation('relu')(h)
    # h = TimeDistributed(Dense(2))(h)
    # h = BatchNormalization()(h)
    # h = Activation('relu')(h)

    latent_dim = 2
    z_mean = TimeDistributed(Dense(latent_dim))(h)
    z_log_var = TimeDistributed(Dense(latent_dim))(h)
    z = Lambda(sampling, output_shape=(K.shape(z_mean)[1], latent_dim))([z_mean, z_log_var])

    # h = TimeDistributed(Dense(4))(z)
    # h = BatchNormalization()(h)
    # h = Activation('relu')(h)
    # h = TimeDistributed(Dense(8))(h)
    # h = BatchNormalization()(h)
    # h = Activation('relu')(h)
    h = TimeDistributed(Dense(16))(z)
    h = BatchNormalization()(h)
    h = Activation('relu')(h)
    h = TimeDistributed(Dense(32))(h)
    h = BatchNormalization()(h)
    h = Activation('relu')(h)
    h = TimeDistributed(Dense(64))(h)
    h_decoded = Reshape((1, 313, 64))(h)
    h_decoded = Conv2DTranspose(32, kernel_size=(3, 3), padding='same')(h_decoded)
    h_decoded = BatchNormalization(axis=1)(h_decoded)
    h_decoded = Activation('relu')(h_decoded)
    h_decoded = Conv2DTranspose(32, kernel_size=(3, 3), padding='same')(h_decoded)
    h_decoded = BatchNormalization(axis=1)(h_decoded)
    h_decoded = Activation('relu')(h_decoded)
    h_decoded = Conv2DTranspose(32, kernel_size=(3, 3), padding='same')(h_decoded)
    h_decoded = BatchNormalization(axis=1)(h_decoded)
    h_decoded = Activation('relu')(h_decoded)
    x_decoded = Conv2DTranspose(1, kernel_size=(3, 3), padding='same')(h_decoded)

    cvae = Model(x, x_decoded)

    def cvae_loss(x, x_decoded):
        xent_loss = mse(x, x_decoded)
        kl_loss = -0.0005*K.mean(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
        return xent_loss + kl_loss

    cvae.compile(optimizer='adam', loss=cvae_loss)
    cvae.summary()

    callback = EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=20, mode='min')
    hist = cvae.fit(train_data,
                    train_data,
                    epochs=2,
                    batch_size=128,
                    shuffle=True,
                    validation_split=0.1,
                    verbose=1,
                    callbacks=[callback])
    train_loss = hist.history['loss']
    val_loss = hist.history['val_loss']
    plot_losses(train_loss, val_loss)

    eval_data = np.load(f'feat/eval_files_{machine_type}_{machine_id}.npz')['arr_0']
    eval_labels = np.load(f'feat/eval_labels_{machine_type}_{machine_id}.npz')['arr_0']
    eval_data = utils.preprocess_data(eval_data, len_audio, nb_ch)
    pred = cvae.predict(eval_data)
    re = np.zeros(eval_labels.shape[0])
    for i in range(eval_labels.shape[0]):
        re[i] = mean_squared_error(eval_data[i].reshape(-1), pred[i].reshape(-1))
    plot_re(re)

    print(f'RE of normal sounds: {str(np.mean(re[:int(re.shape[0]/2)]))}')
    print(f'RE of abnormal sounds: {str(np.mean(re[int(re.shape[0]/2):]))}')
    print(f'AUC: {str(roc_auc_score(eval_labels, re))}')
    print(f'pAUC: {str(roc_auc_score(eval_labels, re, max_fpr=0.1))}')

    # visualize(target_dir)
    cvae.save(f'models/CVAE_{machine_type}_{machine_id}.h5')


def test_AE(target_dir):
    machine_type = target_dir.split('/')[-2]
    machine_id = target_dir.split('/')[-1]
    print(f'{machine_type}_{machine_id}')
    eval_data = np.load(f'feat/eval_files_{machine_type}_{machine_id}.npz')['arr_0']
    eval_labels = np.load(f'feat/eval_labels_{machine_type}_{machine_id}.npz')['arr_0']
    len_audio = int(eval_data.shape[0]/eval_labels.shape[0])
    model = load_model(f'models/AE_{machine_type}_{machine_id}.h5')
    pred = model.predict(eval_data)
    re = np.zeros(eval_labels.shape[0])
    for i in range(eval_labels.shape[0]):
        re[i] = mean_squared_error(eval_data[i*len_audio:(i+1)*len_audio], pred[i*len_audio:(i+1)*len_audio])
    plot_re(re)

    print(f'RE of normal sounds: {str(np.mean(re[:int(re.shape[0]/2)]))}')
    print(f'RE of abnormal sounds: {str(np.mean(re[int(re.shape[0]/2):]))}')
    print(f'AUC: {str(roc_auc_score(eval_labels, re))}')
    print(f'pAUC: {str(roc_auc_score(eval_labels, re, max_fpr=0.1))}')


def test_VAE(target_dir):
    machine_type = target_dir.split('/')[-2]
    machine_id = target_dir.split('/')[-1]
    print(f'{machine_type}_{machine_id}')
    eval_data = np.load(f'feat/eval_files_{machine_type}_{machine_id}.npz')['arr_0']
    eval_labels = np.load(f'feat/eval_labels_{machine_type}_{machine_id}.npz')['arr_0']
    len_audio = int(eval_data.shape[0]/eval_labels.shape[0])

    train_data = np.load(f'feat/train_files_{machine_type}_{machine_id}.npz')['arr_0']

    def sampling(args):
        z_mean, z_log_sigma = args
        epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim), mean=0., stddev=1.0)
        return z_mean + K.exp(z_log_sigma) * epsilon

    x = Input(shape=(train_data.shape[1],))
    h = Dense(32)(x)
    h = BatchNormalization()(h)
    h = Activation('relu')(h)
    h = Dense(16)(h)
    h = BatchNormalization()(h)
    h = Activation('relu')(h)
    h = Dense(8)(h)
    h = BatchNormalization()(h)
    h = Activation('relu')(h)
    h = Dense(4)(h)
    h = BatchNormalization()(h)
    h = Activation('relu')(h)

    latent_dim = 2
    z_mean = Dense(latent_dim)(h)
    z_log_var = Dense(latent_dim)(h)
    z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])

    h_decoded = Dense(4)(z)
    h_decoded = BatchNormalization()(h_decoded)
    h_decoded = Activation('relu')(h_decoded)
    h_decoded = Dense(8)(h_decoded)
    h_decoded = BatchNormalization()(h_decoded)
    h_decoded = Activation('relu')(h_decoded)
    h_decoded = Dense(16)(h_decoded)
    h_decoded = BatchNormalization()(h_decoded)
    h_decoded = Activation('relu')(h_decoded)
    h_decoded = Dense(32)(h_decoded)
    h_decoded = BatchNormalization()(h_decoded)
    h_decoded = Activation('relu')(h_decoded)
    h_decoded = Dense(train_data.shape[1])(h_decoded)
    h_decoded = BatchNormalization()(h_decoded)
    x_decoded = Activation('sigmoid')(h_decoded)

    vae = Model(x, x_decoded)

    def vae_loss(x, x_decoded):
        xent_loss = mse(x, x_decoded)
        kl_loss = -0.5*K.mean(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
        return xent_loss + kl_loss

    model = load_model(f'models/VAE_{machine_type}_{machine_id}.h5', custom_objects={'vae_loss': vae_loss})

    pred = model.predict(eval_data)
    re = np.zeros(eval_labels.shape[0])
    for i in range(eval_labels.shape[0]):
        re[i] = mean_squared_error(eval_data[i*len_audio:(i+1)*len_audio], pred[i*len_audio:(i+1)*len_audio])
    plot_re(re)

    print(f'RE of normal sounds: {str(np.mean(re[:int(re.shape[0]/2)]))}')
    print(f'RE of abnormal sounds: {str(np.mean(re[int(re.shape[0]/2):]))}')
    print(f'AUC: {str(roc_auc_score(eval_labels, re))}')
    print(f'pAUC: {str(roc_auc_score(eval_labels, re, max_fpr=0.1))}')

    # visualize(target_dir)


def test_CAE(target_dir):
    machine_type = target_dir.split('/')[-2]
    machine_id = target_dir.split('/')[-1]
    print(f'{machine_type}_{machine_id}')
    eval_data = np.load(f'feat/eval_files_{machine_type}_{machine_id}.npz')['arr_0']
    eval_labels = np.load(f'feat/eval_labels_{machine_type}_{machine_id}.npz')['arr_0']
    len_audio = int(eval_data.shape[0]/eval_labels.shape[0])
    nb_ch = 1
    eval_data = utils.preprocess_data(eval_data, len_audio, nb_ch)
    model = load_model(f'models/CAE_{machine_type}_{machine_id}.h5')
    pred = model.predict(eval_data)
    re = np.zeros(eval_labels.shape[0])
    for i in range(eval_labels.shape[0]):
        re[i] = mean_squared_error(eval_data[i].reshape(-1), pred[i].reshape(-1))
    plt.figure(figsize=(45, 9))
    plt.plot(re, marker='o')
    plt.title('Reconstruction loss')
    plt.xlabel('sample')
    plt.ylabel('loss')
    plt.show()
    print(f'RE of normal sounds: {str(np.mean(re[:int(re.shape[0]/2)]))}')
    print(f'RE of abnormal sounds: {str(np.mean(re[int(re.shape[0]/2):]))}')
    print(f'AUC: {str(roc_auc_score(eval_labels, re))}')
    print(f'pAUC: {str(roc_auc_score(eval_labels, re, max_fpr=0.1))}')

    # visualize(target_dir)


if __name__ == '__main__':
    audio_folder = 'MIMII/'
    models_folder = 'models/'
    dirs = sorted(glob.glob(os.path.abspath(f'{audio_folder}/*/*')))
    for d in dirs:
        train_AE(d)
        test_AE(d)
        # train_VAE(d)
        # test_VAE(d)
        # train_CAE(d)
        # test_CAE(d)
        # train_CVAE(d)
