import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Dense, Activation, Input, Lambda, Conv2D, Conv2DTranspose, MaxPooling2D, Flatten, Reshape, TimeDistributed
from keras.models import Sequential, Model, load_model
from keras.callbacks import EarlyStopping
from keras.layers.normalization import BatchNormalization
from keras.losses import mse
from keras.optimizers import Adam
from keras import backend as K
K.set_image_data_format('channels_first')
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, mean_squared_error, f1_score, roc_curve, roc_auc_score


# AE
def AE_model(data_in):
    model = Sequential()
    model.add(Dense(32, input_shape = (data_in.shape[1],)))
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

    model.compile(loss = 'mse', optimizer = 'adam')
    model.summary()
    return model


# CAE
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
    print(machine_type + '_' + machine_id)
    train_data = np.load('feat/train_files_' + machine_type + '_' + machine_id + '.npz')['arr_0']
    model = AE_model(train_data)
    callback = EarlyStopping(monitor = 'val_loss', min_delta = 0.0001, patience = 10, mode = 'min')
    hist = model.fit(train_data,
                    train_data,
                    epochs=1000,
                    batch_size=128,
                    shuffle=True,
                    validation_split=0.1,
                    verbose=1,
                    callbacks=[callback])
    plt.plot(hist.history['loss'], label = 'Training loss')  
    plt.plot(hist.history['val_loss'], label = 'Validation loss')
    plt.legend()
    plt.show()
    model.save('models/AE_' + machine_type + '_' + machine_id + '.h5')


def train_VAE(target_dir):
    machine_type = target_dir.split('/')[-2]
    machine_id = target_dir.split('/')[-1]
    print(machine_type + '_' + machine_id)
    train_data = np.load('feat/train_files_' + machine_type + '_' + machine_id + '.npz')['arr_0']

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
        kl_loss = - 0.0005 * K.mean(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
        return xent_loss + kl_loss

    vae.compile(optimizer = 'adam', loss = vae_loss)
    vae.summary()

    callback = EarlyStopping(monitor = 'val_loss', min_delta = 0.0001, patience = 10, mode = 'min')
    hist = vae.fit(train_data,
                    train_data,
                    epochs=1000,
                    batch_size=128,
                    shuffle=True,
                    validation_split=0.1,
                    verbose=1,
                    callbacks=[callback])
    plt.plot(hist.history['loss'], label = 'Training loss')  
    plt.plot(hist.history['val_loss'], label = 'Validation loss')
    plt.legend()
    plt.show()
    vae.save('/models/VAE_' + machine_type + '_' + machine_id + '.h5')


def split_multi_channels(data, num_channels):
    in_shape = data.shape
    if len(in_shape) == 3:
        hop = in_shape[2] // num_channels
        tmp = np.zeros((in_shape[0], num_channels, in_shape[1], hop))
        for i in range(num_channels):
            tmp[:, i, :, :] = data[:, :, i * hop:(i + 1) * hop]
    else:
        print("ERROR: The input should be a 3D matrix but it seems to have dimensions ", in_shape)
        exit()
    return tmp


def split_in_seqs(data, subdivs):
    if len(data.shape) == 1:
        if data.shape[0] % subdivs:
            data = data[:-(data.shape[0] % subdivs), :]
        data = data.reshape((data.shape[0] // subdivs, subdivs, 1))
    elif len(data.shape) == 2:
        if data.shape[0] % subdivs:
            data = data[:-(data.shape[0] % subdivs), :]
        data = data.reshape((data.shape[0] // subdivs, subdivs, data.shape[1]))
    elif len(data.shape) == 3:
        if data.shape[0] % subdivs:
            data = data[:-(data.shape[0] % subdivs), :, :]
        data = data.reshape((data.shape[0] // subdivs, subdivs, data.shape[1], data.shape[2]))
    return data


def preprocess_data(_X, _seq_len, _nb_ch):
    _X = split_in_seqs(_X, _seq_len) 
    _X = split_multi_channels(_X, _nb_ch) 
    return _X


def train_CAE(target_dir):
    machine_type = target_dir.split('/')[-2]
    machine_id = target_dir.split('/')[-1]
    print(machine_type + '_' + machine_id)
    train_data = np.load('feat/train_files_' + machine_type + '_' + machine_id + '.npz')['arr_0']
    train_labels = np.load('feat/train_labels_' + machine_type + '_' + machine_id + '.npz')['arr_0']
    len_audio = int(train_data.shape[0]/train_labels.shape[0])
    nb_ch = 1
    train_data = preprocess_data(train_data, len_audio, nb_ch)
    model = CAE_model(train_data)
    callback = EarlyStopping(monitor = 'val_loss', min_delta = 0.0001, patience = 10, mode = 'min')
    hist = model.fit(train_data,
                    train_data,
                    epochs=1000,
                    batch_size=128,
                    shuffle=True,
                    validation_split=0.1,
                    verbose=1,
                    callbacks=[callback])
    plt.plot(hist.history['loss'], label = 'Training loss')  
    plt.plot(hist.history['val_loss'], label = 'Validation loss')
    plt.legend()
    plt.show()
    model.save('models/CAE_' + machine_type + '_' + machine_id + '.h5')


def test_AE(target_dir):
    machine_type = target_dir.split('/')[-2]
    machine_id = target_dir.split('/')[-1]
    print(machine_type + '_' + machine_id)
    eval_data = np.load('feat/eval_files_' + machine_type + '_' + machine_id + '.npz')['arr_0']
    eval_labels = np.load('feat/eval_labels_' + machine_type + '_' + machine_id + '.npz')['arr_0']
    len_audio = int(eval_data.shape[0]/eval_labels.shape[0])
    model = load_model('models/AE_' + machine_type + '_' + machine_id + '.h5')
    pred = model.predict(eval_data)
    re = np.zeros(eval_labels.shape[0])
    for i in range(eval_labels.shape[0]):
        re[i] = mean_squared_error(eval_data[i*len_audio:(i+1)*len_audio], pred[i*len_audio:(i+1)*len_audio])
    plt.figure(figsize=(45,9))
    plt.plot(re, marker = 'o')
    plt.show()
    print('RE of normal sounds: '+ str(np.mean(re[:int(re.shape[0]/2)])))
    print('RE of abnormal sounds: '+ str(np.mean(re[int(re.shape[0]/2):])))
    print('AUC: ' + str(roc_auc_score(eval_labels, re)))
    print('pAUC: ' + str(roc_auc_score(eval_labels, re, max_fpr=0.1)))


def test_VAE(target_dir):
    machine_type = target_dir.split('/')[-2]
    machine_id = target_dir.split('/')[-1]
    print(machine_type + '_' + machine_id)
    eval_data = np.load('feat/eval_files_' + machine_type + '_' + machine_id + '.npz')['arr_0']
    eval_labels = np.load('feat/eval_labels_' + machine_type + '_' + machine_id + '.npz')['arr_0']
    len_audio = int(eval_data.shape[0]/eval_labels.shape[0])

    train_data = np.load('feat/train_files_' + machine_type + '_' + machine_id + '.npz')['arr_0']

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
        kl_loss = - 0.5 * K.mean(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
        return xent_loss + sys.float_info.epsilon

    model = load_model('models/VAE_' + machine_type + '_' + machine_id + '.h5', custom_objects={'vae_loss':vae_loss})

    pred = model.predict(eval_data)
    re = np.zeros(eval_labels.shape[0])
    for i in range(eval_labels.shape[0]):
        re[i] = mean_squared_error(eval_data[i*len_audio:(i+1)*len_audio], pred[i*len_audio:(i+1)*len_audio])
    plt.figure(figsize=(45,9))
    plt.plot(re, marker = 'o')
    plt.show()
    print('RE of normal sounds: '+ str(np.mean(re[:int(re.shape[0]/2)])))
    print('RE of abnormal sounds: '+ str(np.mean(re[int(re.shape[0]/2):])))
    print('AUC: ' + str(roc_auc_score(eval_labels, re)))
    print('pAUC: ' + str(roc_auc_score(eval_labels, re, max_fpr=0.1)))

    visualize(target_dir)
    fig=plt.figure(figsize = (300, 3))
    plt.imshow(pred.T, cmap=plt.cm.jet, aspect = 'auto')
    ax = plt.gca()
    ax.invert_yaxis()
    xticks_value = np.arange(0, pred.shape[0], len_audio)
    xticks_label = np.arange(0, eval_labels.shape[0])
    plt.xticks(xticks_value, xticks_label)
    plt.show()


def test_CAE(target_dir):
    machine_type = target_dir.split('/')[-2]
    machine_id = target_dir.split('/')[-1]
    print(machine_type + '_' + machine_id)
    eval_data = np.load(path + 'feat/eval_files_' + machine_type + '_' + machine_id + '.npz')['arr_0']
    eval_labels = np.load(path + 'feat/eval_labels_' + machine_type + '_' + machine_id + '.npz')['arr_0']
    len_audio = int(eval_data.shape[0]/eval_labels.shape[0])
    nb_ch = 1
    eval_data = preprocess_data(eval_data, len_audio, nb_ch)
    model = load_model(path + 'models/CAE_' + machine_type + '_' + machine_id + '.h5')
    pred = model.predict(eval_data)
    re = np.zeros(eval_labels.shape[0])
    for i in range(eval_labels.shape[0]):
        re[i] = mean_squared_error(eval_data[i].reshape(-1), pred[i].reshape(-1))
    plt.figure(figsize=(45,9))
    plt.plot(re, marker = 'o')
    plt.show()
    print('RE of normal sounds: '+ str(np.mean(re[:int(re.shape[0]/2)])))
    print('RE of abnormal sounds: '+ str(np.mean(re[int(re.shape[0]/2):])))
    print('AUC: ' + str(roc_auc_score(eval_labels, re)))
    print('pAUC: ' + str(roc_auc_score(eval_labels, re, max_fpr=0.1)))

    pred = pred.reshape(-1, 64)
    visualize(target_dir)
    fig=plt.figure(figsize = (300, 3))
    plt.imshow(pred.T, cmap=plt.cm.jet, aspect = 'auto')
    ax = plt.gca()
    ax.invert_yaxis()
    xticks_value = np.arange(0, pred.shape[0], len_audio)
    xticks_label = np.arange(0, eval_labels.shape[0])
    plt.xticks(xticks_value, xticks_label)
    plt.show()