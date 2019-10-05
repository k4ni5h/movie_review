In [1]:

    # This Python 3 environment comes with many helpful analytics libraries installed
    # It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
    # For example, here's several helpful packages to load in 

    import numpy as np # linear algebra
    import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

    # Input data files are available in the "../input/" directory.
    # For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

    import os
    print(os.listdir("../input"))

    # Any results you write to the current directory are saved as output.

    ['movie-review-sentiment-analysis-kernels-only', 'fatsttext-common-crawl']

In [2]:

    train = pd.read_csv('../input/movie-review-sentiment-analysis-kernels-only/train.tsv', sep="\t")
    test = pd.read_csv('../input/movie-review-sentiment-analysis-kernels-only/test.tsv', sep="\t")
    sub = pd.read_csv('../input/movie-review-sentiment-analysis-kernels-only/sampleSubmission.csv', sep=",")

In [3]:

    train.head(10)

Out[3]:



In [4]:

    train.loc[train.SentenceId == 2]

Out[4]:



In [5]:

    from keras.preprocessing.text import Tokenizer
    from keras.preprocessing.sequence import pad_sequences
    from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation, Conv1D, GRU, CuDNNGRU, CuDNNLSTM, BatchNormalization
    from keras.layers import Bidirectional, GlobalMaxPool1D, MaxPooling1D, Add, Flatten
    from keras.layers import GlobalAveragePooling1D, GlobalMaxPooling1D, concatenate, SpatialDropout1D
    from keras.models import Model, load_model
    from keras import initializers, regularizers, constraints, optimizers, layers, callbacks
    from keras import backend as K
    from keras.engine import InputSpec, Layer
    from keras.optimizers import Adam

    from keras.callbacks import ModelCheckpoint, TensorBoard, Callback, EarlyStopping

    Using TensorFlow backend.

In [6]:

    full_text = list(train['Phrase'].values) + list(test['Phrase'].values)
    tk = Tokenizer(lower = True, filters='')
    tk.fit_on_texts(full_text)

In [7]:

    train_tokenized = tk.texts_to_sequences(train['Phrase'])
    test_tokenized = tk.texts_to_sequences(test['Phrase'])

In [8]:

    max_len = 50
    X_train = pad_sequences(train_tokenized, maxlen = max_len)
    X_test = pad_sequences(test_tokenized, maxlen = max_len)

In [9]:

    embedding_path = "../input/fatsttext-common-crawl/crawl-300d-2M/crawl-300d-2M.vec"

In [10]:

    embed_size = 300
    max_features = 30000

In [11]:

    def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')
    embedding_index = dict(get_coefs(*o.strip().split(" ")) for o in open(embedding_path))

    word_index = tk.word_index
    nb_words = min(max_features, len(word_index))
    embedding_matrix = np.zeros((nb_words + 1, embed_size))
    for word, i in word_index.items():
        if i >= max_features: continue
        embedding_vector = embedding_index.get(word)
        if embedding_vector is not None: embedding_matrix[i] = embedding_vector

In [12]:

    y = train['Sentiment']
    from sklearn.preprocessing import OneHotEncoder
    ohe = OneHotEncoder(sparse=False)
    y_ohe = ohe.fit_transform(y.values.reshape(-1, 1))

    /opt/conda/lib/python3.6/site-packages/sklearn/preprocessing/_encoders.py:363: FutureWarning: The handling of integer data will change in version 0.22. Currently, the categories are determined based on the range [0, max(values)], while in the future they will be determined based on the unique values.
    If you want the future behaviour and silence this warning, you can specify "categories='auto'".
    In case you used a LabelEncoder before this OneHotEncoder to convert the categories to integers, then you can now use the OneHotEncoder directly.
      warnings.warn(msg, FutureWarning)

In [13]:

    def build_model1(lr=0.0, lr_d=0.0, units=0, spatial_dr=0.0, kernel_size1=3, kernel_size2=2, dense_units=128, dr=0.1, conv_size=32):
        file_path = "best_model.hdf5"
        check_point = ModelCheckpoint(file_path, monitor = "val_loss", verbose = 1,
                                      save_best_only = True, mode = "min")
        early_stop = EarlyStopping(monitor = "val_loss", mode = "min", patience = 3)
        
        inp = Input(shape = (max_len,))
        x = Embedding(19479, embed_size, weights = [embedding_matrix], trainable = False)(inp)
        x1 = SpatialDropout1D(spatial_dr)(x)

        x_gru = Bidirectional(CuDNNGRU(units, return_sequences = True))(x1)
        x1 = Conv1D(conv_size, kernel_size=kernel_size1, padding='valid', kernel_initializer='he_uniform')(x_gru)
        avg_pool1_gru = GlobalAveragePooling1D()(x1)
        max_pool1_gru = GlobalMaxPooling1D()(x1)
        
        x3 = Conv1D(conv_size, kernel_size=kernel_size2, padding='valid', kernel_initializer='he_uniform')(x_gru)
        avg_pool3_gru = GlobalAveragePooling1D()(x3)
        max_pool3_gru = GlobalMaxPooling1D()(x3)
        
        x_lstm = Bidirectional(CuDNNLSTM(units, return_sequences = True))(x1)
        x1 = Conv1D(conv_size, kernel_size=kernel_size1, padding='valid', kernel_initializer='he_uniform')(x_lstm)
        avg_pool1_lstm = GlobalAveragePooling1D()(x1)
        max_pool1_lstm = GlobalMaxPooling1D()(x1)
        
        x3 = Conv1D(conv_size, kernel_size=kernel_size2, padding='valid', kernel_initializer='he_uniform')(x_lstm)
        avg_pool3_lstm = GlobalAveragePooling1D()(x3)
        max_pool3_lstm = GlobalMaxPooling1D()(x3)
        
        
        x = concatenate([avg_pool1_gru, max_pool1_gru, avg_pool3_gru, max_pool3_gru,
                        avg_pool1_lstm, max_pool1_lstm, avg_pool3_lstm, max_pool3_lstm])
        x = BatchNormalization()(x)
        x = Dropout(dr)(Dense(dense_units, activation='relu') (x))
        x = BatchNormalization()(x)
        x = Dropout(dr)(Dense(int(dense_units / 2), activation='relu') (x))
        x = Dense(5, activation = "sigmoid")(x)
        model = Model(inputs = inp, outputs = x)
        model.compile(loss = "binary_crossentropy", optimizer = Adam(lr = lr, decay = lr_d), metrics = ["accuracy"])
        history = model.fit(X_train, y_ohe, batch_size = 128, epochs = 20, validation_split=0.1, 
                            verbose = 1, callbacks = [check_point, early_stop])
        model = load_model(file_path)
        return model

In [14]:

    model1 = build_model1(lr = 1e-3, lr_d = 1e-10, units = 64, spatial_dr = 0.3, kernel_size1=3, kernel_size2=2, dense_units=32, dr=0.1, conv_size=32)

    Train on 140454 samples, validate on 15606 samples
    Epoch 1/20
    140454/140454 [==============================] - 86s 615us/step - loss: 0.3572 - acc: 0.8358 - val_loss: 0.3238 - val_acc: 0.8524

    Epoch 00001: val_loss improved from inf to 0.32377, saving model to best_model.hdf5
    Epoch 2/20
    140454/140454 [==============================] - 79s 564us/step - loss: 0.3114 - acc: 0.8579 - val_loss: 0.3114 - val_acc: 0.8559

    Epoch 00002: val_loss improved from 0.32377 to 0.31138, saving model to best_model.hdf5
    Epoch 3/20
    140454/140454 [==============================] - 79s 564us/step - loss: 0.3004 - acc: 0.8630 - val_loss: 0.3030 - val_acc: 0.8588

    Epoch 00003: val_loss improved from 0.31138 to 0.30304, saving model to best_model.hdf5
    Epoch 4/20
    140454/140454 [==============================] - 76s 538us/step - loss: 0.2917 - acc: 0.8664 - val_loss: 0.3018 - val_acc: 0.8603

    Epoch 00004: val_loss improved from 0.30304 to 0.30177, saving model to best_model.hdf5
    Epoch 5/20
    140454/140454 [==============================] - 70s 502us/step - loss: 0.2850 - acc: 0.8698 - val_loss: 0.3021 - val_acc: 0.8598

    Epoch 00005: val_loss did not improve from 0.30177
    Epoch 6/20
    122112/140454 [=========================>....] - ETA: 9s - loss: 0.2789 - acc: 0.8730

In [15]:

    model2 = build_model1(lr = 1e-3, lr_d = 1e-10, units = 128, spatial_dr = 0.5, kernel_size1=3, kernel_size2=2, dense_units=64, dr=0.2, conv_size=32)

    Train on 140454 samples, validate on 15606 samples
    Epoch 1/20
    140454/140454 [==============================] - 83s 593us/step - loss: 0.3567 - acc: 0.8410 - val_loss: 0.4010 - val_acc: 0.8177

    Epoch 00001: val_loss improved from inf to 0.40097, saving model to best_model.hdf5
    Epoch 2/20
    140454/140454 [==============================] - 79s 559us/step - loss: 0.3214 - acc: 0.8540 - val_loss: 0.3152 - val_acc: 0.8537

    Epoch 00002: val_loss improved from 0.40097 to 0.31520, saving model to best_model.hdf5
    Epoch 3/20
    140454/140454 [==============================] - 78s 559us/step - loss: 0.3113 - acc: 0.8582 - val_loss: 0.3070 - val_acc: 0.8580

    Epoch 00003: val_loss improved from 0.31520 to 0.30700, saving model to best_model.hdf5
    Epoch 4/20
    140454/140454 [==============================] - 79s 560us/step - loss: 0.3041 - acc: 0.8612 - val_loss: 0.3025 - val_acc: 0.8589

    Epoch 00004: val_loss improved from 0.30700 to 0.30246, saving model to best_model.hdf5
    Epoch 5/20
    140454/140454 [==============================] - 79s 560us/step - loss: 0.2967 - acc: 0.8645 - val_loss: 0.3074 - val_acc: 0.8574

    Epoch 00005: val_loss did not improve from 0.30246
    Epoch 6/20
    140454/140454 [==============================] - 79s 559us/step - loss: 0.2920 - acc: 0.8666 - val_loss: 0.2976 - val_acc: 0.8608

    Epoch 00006: val_loss improved from 0.30246 to 0.29765, saving model to best_model.hdf5
    Epoch 7/20
    140454/140454 [==============================] - 79s 560us/step - loss: 0.2873 - acc: 0.8690 - val_loss: 0.2996 - val_acc: 0.8618

    Epoch 00007: val_loss did not improve from 0.29765
    Epoch 8/20
    140454/140454 [==============================] - 79s 559us/step - loss: 0.2823 - acc: 0.8713 - val_loss: 0.2971 - val_acc: 0.8623

    Epoch 00008: val_loss improved from 0.29765 to 0.29713, saving model to best_model.hdf5
    Epoch 9/20
    140454/140454 [==============================] - 79s 562us/step - loss: 0.2792 - acc: 0.8730 - val_loss: 0.2980 - val_acc: 0.8620

    Epoch 00009: val_loss did not improve from 0.29713
    Epoch 10/20
    140454/140454 [==============================] - 79s 561us/step - loss: 0.2766 - acc: 0.8746 - val_loss: 0.3000 - val_acc: 0.8633

    Epoch 00010: val_loss did not improve from 0.29713
    Epoch 11/20
    140454/140454 [==============================] - 79s 560us/step - loss: 0.2732 - acc: 0.8761 - val_loss: 0.2968 - val_acc: 0.8613

    Epoch 00011: val_loss improved from 0.29713 to 0.29681, saving model to best_model.hdf5
    Epoch 12/20
     52480/140454 [==========>...................] - ETA: 47s - loss: 0.2706 - acc: 0.8774

In [16]:

    def build_model2(lr=0.0, lr_d=0.0, units=0, spatial_dr=0.0, kernel_size1=3, kernel_size2=2, dense_units=128, dr=0.1, conv_size=32):
        file_path = "best_model.hdf5"
        check_point = ModelCheckpoint(file_path, monitor = "val_loss", verbose = 1,
                                      save_best_only = True, mode = "min")
        early_stop = EarlyStopping(monitor = "val_loss", mode = "min", patience = 3)

        inp = Input(shape = (max_len,))
        x = Embedding(19479, embed_size, weights = [embedding_matrix], trainable = False)(inp)
        x1 = SpatialDropout1D(spatial_dr)(x)

        x_gru = Bidirectional(CuDNNGRU(units, return_sequences = True))(x1)
        x_lstm = Bidirectional(CuDNNLSTM(units, return_sequences = True))(x1)
        
        x_conv1 = Conv1D(conv_size, kernel_size=kernel_size1, padding='valid', kernel_initializer='he_uniform')(x_gru)
        avg_pool1_gru = GlobalAveragePooling1D()(x_conv1)
        max_pool1_gru = GlobalMaxPooling1D()(x_conv1)
        
        x_conv2 = Conv1D(conv_size, kernel_size=kernel_size2, padding='valid', kernel_initializer='he_uniform')(x_gru)
        avg_pool2_gru = GlobalAveragePooling1D()(x_conv2)
        max_pool2_gru = GlobalMaxPooling1D()(x_conv2)
        
        
        x_conv3 = Conv1D(conv_size, kernel_size=kernel_size1, padding='valid', kernel_initializer='he_uniform')(x_lstm)
        avg_pool1_lstm = GlobalAveragePooling1D()(x_conv3)
        max_pool1_lstm = GlobalMaxPooling1D()(x_conv3)
        
        x_conv4 = Conv1D(conv_size, kernel_size=kernel_size2, padding='valid', kernel_initializer='he_uniform')(x_lstm)
        avg_pool2_lstm = GlobalAveragePooling1D()(x_conv4)
        max_pool2_lstm = GlobalMaxPooling1D()(x_conv4)
        
        
        x = concatenate([avg_pool1_gru, max_pool1_gru, avg_pool2_gru, max_pool2_gru,
                        avg_pool1_lstm, max_pool1_lstm, avg_pool2_lstm, max_pool2_lstm])
        x = BatchNormalization()(x)
        x = Dropout(dr)(Dense(dense_units, activation='relu') (x))
        x = BatchNormalization()(x)
        x = Dropout(dr)(Dense(int(dense_units / 2), activation='relu') (x))
        x = Dense(5, activation = "sigmoid")(x)
        model = Model(inputs = inp, outputs = x)
        model.compile(loss = "binary_crossentropy", optimizer = Adam(lr = lr, decay = lr_d), metrics = ["accuracy"])
        history = model.fit(X_train, y_ohe, batch_size = 128, epochs = 20, validation_split=0.1, 
                            verbose = 1, callbacks = [check_point, early_stop])
        model = load_model(file_path)
        return model

In [17]:

    model3 = build_model2(lr = 1e-4, lr_d = 0, units = 64, spatial_dr = 0.5, kernel_size1=4, kernel_size2=3, dense_units=32, dr=0.1, conv_size=32)

    Train on 140454 samples, validate on 15606 samples
    Epoch 1/20
    140454/140454 [==============================] - 76s 541us/step - loss: 0.4545 - acc: 0.7941 - val_loss: 0.3679 - val_acc: 0.8422

    Epoch 00001: val_loss improved from inf to 0.36793, saving model to best_model.hdf5
    Epoch 2/20
    140454/140454 [==============================] - 73s 521us/step - loss: 0.3651 - acc: 0.8413 - val_loss: 0.3353 - val_acc: 0.8482

    Epoch 00002: val_loss improved from 0.36793 to 0.33532, saving model to best_model.hdf5
    Epoch 3/20
    140454/140454 [==============================] - 62s 439us/step - loss: 0.3466 - acc: 0.8467 - val_loss: 0.3261 - val_acc: 0.8502

    Epoch 00003: val_loss improved from 0.33532 to 0.32608, saving model to best_model.hdf5
    Epoch 4/20
    140454/140454 [==============================] - 62s 439us/step - loss: 0.3377 - acc: 0.8487 - val_loss: 0.3219 - val_acc: 0.8514

    Epoch 00004: val_loss improved from 0.32608 to 0.32188, saving model to best_model.hdf5
    Epoch 5/20
    140454/140454 [==============================] - 62s 439us/step - loss: 0.3314 - acc: 0.8511 - val_loss: 0.3176 - val_acc: 0.8518

    Epoch 00005: val_loss improved from 0.32188 to 0.31759, saving model to best_model.hdf5
    Epoch 6/20
    140454/140454 [==============================] - 62s 439us/step - loss: 0.3275 - acc: 0.8519 - val_loss: 0.3167 - val_acc: 0.8530

    Epoch 00006: val_loss improved from 0.31759 to 0.31669, saving model to best_model.hdf5
    Epoch 7/20
    140454/140454 [==============================] - 62s 440us/step - loss: 0.3238 - acc: 0.8537 - val_loss: 0.3149 - val_acc: 0.8537

    Epoch 00007: val_loss improved from 0.31669 to 0.31492, saving model to best_model.hdf5
    Epoch 8/20
    140454/140454 [==============================] - 62s 440us/step - loss: 0.3211 - acc: 0.8542 - val_loss: 0.3134 - val_acc: 0.8545

    Epoch 00008: val_loss improved from 0.31492 to 0.31341, saving model to best_model.hdf5
    Epoch 9/20
    140454/140454 [==============================] - 62s 439us/step - loss: 0.3181 - acc: 0.8559 - val_loss: 0.3121 - val_acc: 0.8552

    Epoch 00009: val_loss improved from 0.31341 to 0.31213, saving model to best_model.hdf5
    Epoch 10/20
    140454/140454 [==============================] - 61s 437us/step - loss: 0.3156 - acc: 0.8564 - val_loss: 0.3115 - val_acc: 0.8548

    Epoch 00010: val_loss improved from 0.31213 to 0.31148, saving model to best_model.hdf5
    Epoch 11/20
    140454/140454 [==============================] - 62s 440us/step - loss: 0.3136 - acc: 0.8571 - val_loss: 0.3117 - val_acc: 0.8557

    Epoch 00011: val_loss did not improve from 0.31148
    Epoch 12/20
     54784/140454 [==========>...................] - ETA: 36s - loss: 0.3122 - acc: 0.8580

In [18]:

    model4 = build_model2(lr = 1e-3, lr_d = 0, units = 64, spatial_dr = 0.5, kernel_size1=3, kernel_size2=3, dense_units=64, dr=0.3, conv_size=32)

    Train on 140454 samples, validate on 15606 samples
    Epoch 1/20
    140454/140454 [==============================] - 77s 545us/step - loss: 0.3753 - acc: 0.8303 - val_loss: 0.3191 - val_acc: 0.8508

    Epoch 00001: val_loss improved from inf to 0.31906, saving model to best_model.hdf5
    Epoch 2/20
    140454/140454 [==============================] - 71s 508us/step - loss: 0.3267 - acc: 0.8526 - val_loss: 0.3137 - val_acc: 0.8518

    Epoch 00002: val_loss improved from 0.31906 to 0.31375, saving model to best_model.hdf5
    Epoch 3/20
    140454/140454 [==============================] - 61s 438us/step - loss: 0.3162 - acc: 0.8565 - val_loss: 0.3160 - val_acc: 0.8544

    Epoch 00003: val_loss did not improve from 0.31375
    Epoch 4/20
    140454/140454 [==============================] - 61s 436us/step - loss: 0.3081 - acc: 0.8594 - val_loss: 0.3093 - val_acc: 0.8569

    Epoch 00004: val_loss improved from 0.31375 to 0.30933, saving model to best_model.hdf5
    Epoch 5/20
    140454/140454 [==============================] - 61s 436us/step - loss: 0.3018 - acc: 0.8621 - val_loss: 0.3046 - val_acc: 0.8589

    Epoch 00005: val_loss improved from 0.30933 to 0.30463, saving model to best_model.hdf5
    Epoch 6/20
    140454/140454 [==============================] - 61s 437us/step - loss: 0.2972 - acc: 0.8639 - val_loss: 0.3059 - val_acc: 0.8586

    Epoch 00006: val_loss did not improve from 0.30463
    Epoch 7/20
    140454/140454 [==============================] - 61s 437us/step - loss: 0.2932 - acc: 0.8660 - val_loss: 0.3048 - val_acc: 0.8606

    Epoch 00007: val_loss did not improve from 0.30463
    Epoch 8/20
    140454/140454 [==============================] - 61s 436us/step - loss: 0.2887 - acc: 0.8678 - val_loss: 0.3018 - val_acc: 0.8603

    Epoch 00008: val_loss improved from 0.30463 to 0.30181, saving model to best_model.hdf5
    Epoch 9/20
    140454/140454 [==============================] - 61s 436us/step - loss: 0.2855 - acc: 0.8695 - val_loss: 0.3023 - val_acc: 0.8609

    Epoch 00009: val_loss did not improve from 0.30181
    Epoch 10/20
    140454/140454 [==============================] - 61s 437us/step - loss: 0.2829 - acc: 0.8707 - val_loss: 0.3037 - val_acc: 0.8596

    Epoch 00010: val_loss did not improve from 0.30181
    Epoch 11/20
    140454/140454 [==============================] - 61s 436us/step - loss: 0.2811 - acc: 0.8720 - val_loss: 0.3018 - val_acc: 0.8603

    Epoch 00011: val_loss improved from 0.30181 to 0.30179, saving model to best_model.hdf5
    Epoch 12/20
     56576/140454 [===========>..................] - ETA: 35s - loss: 0.2765 - acc: 0.8742

In [19]:

    model5 = build_model2(lr = 1e-3, lr_d = 1e-7, units = 64, spatial_dr = 0.3, kernel_size1=3, kernel_size2=3, dense_units=64, dr=0.4, conv_size=64)

    Train on 140454 samples, validate on 15606 samples
    Epoch 1/20
    140454/140454 [==============================] - 74s 528us/step - loss: 0.3683 - acc: 0.8363 - val_loss: 0.3301 - val_acc: 0.8490

    Epoch 00001: val_loss improved from inf to 0.33014, saving model to best_model.hdf5
    Epoch 2/20
    140454/140454 [==============================] - 70s 497us/step - loss: 0.3192 - acc: 0.8550 - val_loss: 0.3110 - val_acc: 0.8526

    Epoch 00002: val_loss improved from 0.33014 to 0.31102, saving model to best_model.hdf5
    Epoch 3/20
    140454/140454 [==============================] - 65s 462us/step - loss: 0.3071 - acc: 0.8594 - val_loss: 0.3084 - val_acc: 0.8561

    Epoch 00003: val_loss improved from 0.31102 to 0.30838, saving model to best_model.hdf5
    Epoch 4/20
    140454/140454 [==============================] - 65s 464us/step - loss: 0.2973 - acc: 0.8638 - val_loss: 0.3055 - val_acc: 0.8614

    Epoch 00004: val_loss improved from 0.30838 to 0.30555, saving model to best_model.hdf5
    Epoch 5/20
    140454/140454 [==============================] - 65s 461us/step - loss: 0.2898 - acc: 0.8673 - val_loss: 0.3061 - val_acc: 0.8596

    Epoch 00005: val_loss did not improve from 0.30555
    Epoch 6/20
    140454/140454 [==============================] - 65s 461us/step - loss: 0.2828 - acc: 0.8706 - val_loss: 0.3028 - val_acc: 0.8604

    Epoch 00006: val_loss improved from 0.30555 to 0.30282, saving model to best_model.hdf5
    Epoch 7/20
    140454/140454 [==============================] - 65s 461us/step - loss: 0.2780 - acc: 0.8730 - val_loss: 0.3092 - val_acc: 0.8589

    Epoch 00007: val_loss did not improve from 0.30282
    Epoch 8/20
    140454/140454 [==============================] - 65s 463us/step - loss: 0.2733 - acc: 0.8755 - val_loss: 0.3025 - val_acc: 0.8612

    Epoch 00008: val_loss improved from 0.30282 to 0.30245, saving model to best_model.hdf5
    Epoch 9/20
    140454/140454 [==============================] - 65s 462us/step - loss: 0.2698 - acc: 0.8778 - val_loss: 0.3046 - val_acc: 0.8605

    Epoch 00009: val_loss did not improve from 0.30245
    Epoch 10/20
    140454/140454 [==============================] - 65s 462us/step - loss: 0.2669 - acc: 0.8793 - val_loss: 0.3023 - val_acc: 0.8612

    Epoch 00010: val_loss improved from 0.30245 to 0.30235, saving model to best_model.hdf5
    Epoch 11/20
    140454/140454 [==============================] - 64s 459us/step - loss: 0.2636 - acc: 0.8805 - val_loss: 0.3036 - val_acc: 0.8619

    Epoch 00011: val_loss did not improve from 0.30235
    Epoch 12/20
     53120/140454 [==========>...................] - ETA: 39s - loss: 0.2587 - acc: 0.8828

In [20]:

    pred1 = model1.predict(X_test, batch_size = 1024, verbose = 1)
    pred = pred1
    pred2 = model2.predict(X_test, batch_size = 1024, verbose = 1)
    pred += pred2
    pred3 = model3.predict(X_test, batch_size = 1024, verbose = 1)
    pred += pred3
    pred4 = model4.predict(X_test, batch_size = 1024, verbose = 1)
    pred += pred4
    pred5 = model5.predict(X_test, batch_size = 1024, verbose = 1)
    pred += pred5

    66292/66292 [==============================] - 6s 84us/step
    66292/66292 [==============================] - 7s 105us/step
    66292/66292 [==============================] - 5s 81us/step
    66292/66292 [==============================] - 5s 80us/step
    66292/66292 [==============================] - 6s 87us/step

In [21]:

    predictions = np.round(np.argmax(pred, axis=1)).astype(int)
    sub['Sentiment'] = predictions
    sub.to_csv("blend.csv", index=False)
