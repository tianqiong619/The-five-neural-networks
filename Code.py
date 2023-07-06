import warnings
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold
import pandas as pd
from tensorflow.python.keras.utils import np_utils
warnings.filterwarnings(action='ignore')

def cross_validation(df_train, labels_train, df_test, labels_test, k, model,MODEL):
    test_predict = []
    test_precision = []
    test_recall = []
    test_TE_Score= []

    cv = StratifiedKFold(n_splits=k, shuffle=True, random_state=0)

    cnt = 0
    allacc1 = 0
    allacc2 = 0
    allacc3 = 0
    sum_fpr = 0
    sum_tpr = 0
    sum_auc = 0

    label_test = labels_test
    labels_test = np_utils.to_categorical(labels_test,num_classes=2,dtype='float32')
    df_test = df_test.reshape(-1, a, 1)

    print("The accuracy of five-fold cross-validation is as follows:")
    for i, (train, test) in enumerate(cv.split(df_train, labels_train)):
        cnt += 1
        print('the %d iterate: ' % cnt)
        label_train = labels_train[train]
        y_train = np_utils.to_categorical(label_train, num_classes=2,dtype='float32')
        data_train = df_train[train].reshape(-1, a, 1)

        label_val = labels_train[test]
        labels_val = np_utils.to_categorical(label_val, num_classes=2,dtype='float32')
        df_val = df_train[test].reshape(-1, a, 1)

        from keras.callbacks import ModelCheckpoint
        filepath = r'./best_model.h5'
        checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1,save_best_only='True', save_weights_only='True',mode='max',period=1)
        history = model.fit(data_train, y_train, batch_size=24, epochs=200, verbose=2, shuffle=True,validation_data=(df_val, labels_val), callbacks=[checkpoint])

        test_predict1 = model.predict(data_train)
        test_predict1 = np.argmax(test_predict1, axis=1)
        accuracy1 = accuracy_score(label_train, test_predict1)
        print('Training set accuracy: ', accuracy1)
        allacc1 = allacc1 + accuracy1

        test_predict2 = model.predict(df_val)
        test_predict2 = np.argmax(test_predict2, axis=1)
        accuracy2 = accuracy_score(label_val, test_predict2)
        print('Validation set accuracy: ', accuracy2)
        allacc2 = allacc2 + accuracy2

        test_predict3 = model.predict(df_test)
        test_predict3 = np.argmax(test_predict3, axis=1)
        accuracy3 = accuracy_score(label_test, test_predict3)
        print('Test set accuracy:', accuracy3)
        allacc3 = allacc3 + accuracy3

        predict_p = model.predict(df_test)
        y_score = predict_p
        test_predict.append(predict_p)

        matrix = confusion_matrix(label_test, test_predict3)
        accuracy4 = matrix[1][1]/(matrix[1][1]+matrix[1][0])
        test_precision.append(accuracy4)

        accuracy5 = matrix[1][1]/(matrix[1][1]+matrix[0][1])
        test_recall.append(accuracy5)

        accuracy7 = matrix[0][0]/(matrix[1][0]+matrix[0][0])
        test_TE_Score.append(accuracy7)

        fpr, tpr, _ = roc_curve(labels_test.ravel(), y_score.ravel())
        AUC = auc(fpr, tpr)
        mean_fpr = np.linspace(0, 1, 100)
        tpr = np.interp(mean_fpr, fpr, tpr)

        sum_fpr += mean_fpr
        sum_tpr += tpr
        sum_auc += AUC

        print('=========The cross-validation result of the '+str(cnt)+' fold==========')
        print('Average accuracy on the training set: ', accuracy1)
        print('Average accuracy on the validation set: ', accuracy2)
        print('Average accuracy on the test set: ', accuracy3)

        print('=========The test index of the test set after the '+str(cnt)+' fold cross-validation=======')
        print('Sensitivity on the test set: ', accuracy5.mean())
        print('Specificity on the test set:  ', accuracy7.mean())
        print('Accuracy on the test set: ', accuracy4.mean())

    mean_predict = test_predict[0]
    for i in range(1, 5):
        mean_predict += test_predict[i]
    mean_predict = mean_predict / 5
    print(mean_predict)
    avg_auc = sum_auc/5
    avg_fpr = sum_fpr/5
    avg_tpr = sum_tpr/5

    predict_idx = np.argmax(mean_predict, axis=1)
    mean_precision = test_precision[0]
    for i in range(1, 5):
        mean_precision += test_precision[i]
    mean_precision = mean_precision / 5

    mean_recall = test_recall[0]
    for i in range(1, 5):
        mean_recall += test_recall[i]
    mean_recall = mean_recall / 5

    mean_TE_Score = test_TE_Score
    for i in range(1, 5):
        mean_TE_Score += test_TE_Score[i]
    mean_TE_Score = mean_TE_Score / 5

    print('=========The final result after five-fold cross-validation==========')
    print('Average accuracy on the test set: ', allacc3 / 5)

    print('=======Test metrics for the test set after five-fold cross-validation=======')
    print('Average sensitivity on the test set: ', mean_recall.mean())
    print('Average specificity on the test set: ', mean_TE_Score.mean())
    print('Average precision on the test set: ', mean_precision.mean())
    print('Average AUC on test set ', avg_auc)

    n_classes = 2
    font2 = {'family': 'Times New Roman',
             'weight': 'normal',
             'size': 17,
             }

    np.savetxt(MODEL+"_fpr.txt", avg_fpr, fmt='%f')
    np.savetxt(MODEL+"_tpr.txt", avg_tpr, fmt='%f')

    lw = 2
    plt.figure()
    plt.plot(avg_fpr, avg_tpr, color='aqua', lw=lw,
             label='ROC curve  (AUC = {1:0.3f})'
                   ''.format(1, avg_auc))
    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    plt.xlabel('False Positive Rate', font2)
    plt.ylabel('True Positive Rate', font2)
    plt.title(str(ModelName), font2)
    plt.legend(loc="lower right")
    plt.show()

def ModelSelect(name):

#==============================================   AlexNet   ======================================================
    if name == 'AlexNet':
        from tensorflow.keras import Input
        from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Reshape, multiply, \
            GRU, UpSampling1D, \
            ZeroPadding1D, Activation, Dropout, BatchNormalization, concatenate, LeakyReLU, GlobalAvgPool1D
        from tensorflow.keras.layers import Dense, Softmax
        from tensorflow.keras.optimizers import Adam
        from tensorflow.keras.models import Model
        import numpy as np
        from keras.utils import np_utils
        import matplotlib.pyplot as plt
        from tensorflow.keras.layers import add
        import pandas as pd

        inpt = Input(shape=(a, 1))
        x = Conv1D(24, 11, strides=4, input_shape=(a, 1), padding='valid', activation='relu',
                         kernel_initializer='uniform')(inpt)
        x = Conv1D(64, 5, strides=1, padding='same', activation='relu', kernel_initializer='uniform')(x)
        x = Conv1D(92, 3, strides=1, padding='same', activation='relu', kernel_initializer='uniform')(x)
        x = Conv1D(92, 3, strides=1, padding='same', activation='relu', kernel_initializer='uniform')(x)
        x = Conv1D(64, 3, strides=1, padding='same', activation='relu', kernel_initializer='uniform')(x)
        x = MaxPooling1D(pool_size=2, strides=2)(x)
        x = Flatten()(x)
        x = Dense(36, activation='relu')(x)
        x = Dropout(0.5)(x)
        x = Dense(36, activation='relu')(x)
        x = Dropout(0.2)(x)
        x = Dense(2, activation='softmax')(x)
        modelAlexNet = Model(inputs=inpt, outputs=x, name='AlexNet')
        adam = Adam(lr=0.000001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=3e-8)
        modelAlexNet.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

#===============================================   ResNet   ==================================================================#
    if name == 'ResNet':
        from tensorflow.keras import Input
        from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Reshape, multiply, Embedding, \
            GRU, UpSampling1D, \
            ZeroPadding1D, Activation, Dropout, BatchNormalization, concatenate, LeakyReLU, GlobalAvgPool1D
        from tensorflow.keras.layers import Dense, Softmax
        from tensorflow.keras.optimizers import Adam
        from tensorflow.keras.models import Model
        import numpy as np
        from keras.utils import np_utils
        import matplotlib.pyplot as plt
        from tensorflow.keras.layers import add
        import pandas as pd

        def Conv1d_BN(x, nb_filter, kernel_size, strides=1, padding='same', name=None):
            if name is not None:
                bn_name = name + '_bn'
                conv_name = name + '_conv'
            else:
                bn_name = None
                conv_name = None

            x = Conv1D(nb_filter, kernel_size, padding=padding, strides=strides, activation='relu', name=conv_name)(x)
            x = BatchNormalization(axis=2, name=bn_name)(x)
            return x

        def Conv_Block(inpt, nb_filter, kernel_size, strides=1, with_conv_shortcut=False):
            x = Conv1d_BN(inpt, nb_filter=nb_filter, kernel_size=kernel_size, strides=strides, padding='same')
            x = Conv1d_BN(x, nb_filter=nb_filter, kernel_size=kernel_size, padding='same')
            if with_conv_shortcut:
                shortcut = Conv1d_BN(inpt, nb_filter=nb_filter, strides=strides, kernel_size=kernel_size)
                x = add([x, shortcut])
            else:
                x = add([x, inpt])
            return x

        inpt = Input(shape=(a, 1))
        x = ZeroPadding1D(3)(inpt)
        x = Conv1d_BN(x, nb_filter=24, kernel_size=7, strides=2, padding='valid')
        x = Conv_Block(x, nb_filter=24, kernel_size=3)
        x = Conv_Block(x, nb_filter=24, kernel_size=3)
        x = Conv_Block(x, nb_filter=48, kernel_size=3, strides=2, with_conv_shortcut=True)
        x = Conv_Block(x, nb_filter=48, kernel_size=3)
        x = Conv_Block(x, nb_filter=64, kernel_size=3, strides=2, with_conv_shortcut=True)
        x = Conv_Block(x, nb_filter=64, kernel_size=3)
        x = Conv_Block(x, nb_filter=128, kernel_size=3, strides=2, with_conv_shortcut=True)
        x = Conv_Block(x, nb_filter=128, kernel_size=3)
        x = Flatten()(x)
        x = Dense(2, activation='softmax')(x)
        modelResNet = Model(inputs=inpt, outputs=x)
        adam = Adam(lr=0.000001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=3e-8)
        modelResNet.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

#==================================================   MCNN   ================================================================#
    if name == 'MCNN':
        from tensorflow.keras import Input
        from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Reshape, Multiply, multiply, Embedding, \
            GRU, UpSampling1D, \
            ZeroPadding1D, Activation, Dropout, BatchNormalization, concatenate, LeakyReLU, GlobalAvgPool1D
        from tensorflow.keras.layers import Dense, Softmax
        from tensorflow.keras.optimizers import Adam
        from tensorflow.keras.models import Model
        import numpy as np
        from keras.utils import np_utils
        import matplotlib.pyplot as plt
        from tensorflow.keras.layers import add
        import pandas as pd

        inpt = Input(shape=(a, 1))
        x = Reshape((a, 1))(inpt)
        x = BatchNormalization()(x, training=False)
        x1 = Conv1D(filters=16, kernel_size=4, strides=1, padding='same', kernel_initializer='random_normal',
                    bias_initializer='zeros')(x)
        x1 = BatchNormalization()(x1, training=False)
        x1 = LeakyReLU(alpha=0.1)(x1)
        x2 = Conv1D(filters=16, kernel_size=8, strides=1, padding='same', kernel_initializer='random_normal')(x1)
        x2 = BatchNormalization()(x2, training=False)
        x2 = LeakyReLU(alpha=0.1)(x2)
        x3 = Conv1D(filters=16, kernel_size=16, strides=1, padding='same', kernel_initializer='random_normal')(x2)
        x3 = BatchNormalization()(x3, training=False)
        x3 = LeakyReLU(alpha=0.1)(x3)
        x = concatenate([x1, x2, x3])
        x = Flatten()(x)
        x = Dropout(0.5)(x)
        x = Dense(256, kernel_initializer='random_normal')(x)
        x = LeakyReLU(alpha=0.1)(x)
        x = Dropout(0.5)(x)
        x = Dense(2, activation='softmax', kernel_initializer='random_normal')(x)
        modelMCNN = Model(inputs=[inpt], outputs=[x])
        adam = Adam(lr=0.000001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=3e-8)
        modelMCNN.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

#==================================================   SqueezeNet   ================================================================#
    if name == 'SqueezeNet':
        from keras import Input
        from keras.layers import Conv1D, MaxPooling1D, Flatten, Reshape, Embedding, GRU, UpSampling1D, \
            ZeroPadding1D, Activation, Dropout, BatchNormalization, Concatenate, LeakyReLU, GlobalAveragePooling1D, \
            Convolution1D
        from keras.layers import Dense
        from tensorflow.keras.optimizers import Adam
        from keras.models import Model
        from sklearn.metrics import roc_curve, auc
        import numpy as np
        from keras.utils import np_utils
        from keras.layers import add

        adam = Adam(lr=0.000001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=3e-8)

        def Conv1d_BN(x, nb_filter, kernel_size, strides=1, padding='same', name=None):
            if name is not None:
                bn_name = name + '_bn'
                conv_name = name + '_conv'
            else:
                bn_name = None
                conv_name = None

            x = Conv1D(nb_filter, kernel_size, padding=padding, strides=strides, activation='relu', name=conv_name)(x)
            x = BatchNormalization(axis=2, name=bn_name)(x)
            return x

        def Conv_Block(inpt, nb_filter, kernel_size, strides=1, with_conv_shortcut=False):
            x = Conv1d_BN(inpt, nb_filter=nb_filter, kernel_size=kernel_size, strides=strides, padding='same')
            x = Conv1d_BN(x, nb_filter=nb_filter, kernel_size=kernel_size, padding='same')
            if with_conv_shortcut:
                shortcut = Conv1d_BN(inpt, nb_filter=nb_filter, strides=strides, kernel_size=kernel_size)
                x = add([x, shortcut])
                return x
            else:
                x = add([x, inpt])
                return x

        input_layer = Input(shape=(a, 1))

        x = ZeroPadding1D(1)(input_layer)
        conv1 = Conv1d_BN(x, nb_filter=16, kernel_size=3, strides=2, padding='valid')
        maxpool1 = MaxPooling1D(pool_size=3, strides=2, padding='same')(conv1)

        fire2_squeeze = Conv_Block(maxpool1, nb_filter=16, kernel_size=3)
        fire2_expand1 = Conv_Block(fire2_squeeze, nb_filter=16, kernel_size=3, strides=1, with_conv_shortcut=True)
        fire2_expand2 = Conv_Block(fire2_squeeze, nb_filter=16, kernel_size=3)
        merge2 = Concatenate(axis=2)([fire2_expand1, fire2_expand2])

        fire3_squeeze = Conv_Block(merge2, nb_filter=32, kernel_size=1)
        fire3_expand1 = Conv_Block(fire3_squeeze, nb_filter=32, kernel_size=1, strides=1, with_conv_shortcut=True)
        fire3_expand2 = Conv_Block(fire3_squeeze, nb_filter=32, kernel_size=1)
        merge3 = Concatenate(axis=2)([fire3_expand1, fire3_expand2])

        fire4_squeeze = Conv_Block(merge3, nb_filter=64, kernel_size=3)
        fire4_expand1 = Conv_Block(fire4_squeeze, nb_filter=64, kernel_size=3, strides=1, with_conv_shortcut=True)
        fire4_expand2 = Conv_Block(fire4_squeeze, nb_filter=64, kernel_size=3)
        merge4 = Concatenate(axis=2)([fire4_expand1, fire4_expand2])

        fire5_squeeze = Conv_Block(merge4, nb_filter=128, kernel_size=1)
        fire5_expand1 = Conv_Block(fire5_squeeze, nb_filter=128, kernel_size=1, strides=1, with_conv_shortcut=True)
        fire5_expand2 = Conv_Block(fire5_squeeze, nb_filter=128, kernel_size=1)
        merge5 = Concatenate(axis=2)([fire5_expand1, fire5_expand2])
        fire9_dropout = Dropout(0.5, name='fire9_dropout')(merge5)

        conv10 = Convolution1D(2, (1,), activation='relu', kernel_initializer='glorot_uniform',
                               padding='valid', name='conv10')(fire9_dropout)

        global_avgpool10 = GlobalAveragePooling1D()(conv10)
        softmax = Activation("softmax", name='softmax')(global_avgpool10)
        modelSqueezeNet = Model(inputs=input_layer, outputs=softmax)
        modelSqueezeNet.summary()
        modelSqueezeNet.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

#==================================================   TCN   ================================================================#
    if name == 'TCN':
        from tensorflow.keras.models import Model
        from tensorflow.keras.layers import add, Input, Conv1D, Activation, Flatten, Dense
        from tensorflow.keras.optimizers import Adam

        def ResBlock(x, filters, kernel_size, dilation_rate):
            r = Conv1D(filters, kernel_size, padding='causal', dilation_rate=dilation_rate, activation='relu')(x)
            r = Conv1D(filters, kernel_size, padding='causal', dilation_rate=dilation_rate)(r)
            if x.shape[-1] == filters:
                shortcut = x
            else:
                shortcut = Conv1D(filters, 1, padding='same')(x)
            o = add([r, shortcut])
            o = Activation('relu')(o)
            return o

        inputs = Input(shape=(a, 1))
        x = ResBlock(inputs, filters=32, kernel_size=3, dilation_rate=1)
        x = ResBlock(x, filters=32, kernel_size=3, dilation_rate=2)
        x = Flatten()(x)
        x = Dense(2, activation='softmax')(x)
        modelTCN = Model(inputs, x)
        adam = Adam(lr=0.000001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=3e-8)
        modelTCN.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    if name == 'AlexNet':
        model = modelAlexNet

    if name == 'ResNet':
        model = modelResNet

    if name == 'MCNN':
        model = modelMCNN

    if name == 'SqueezeNet':
        model = modelSqueezeNet

    if name == 'TCN':
        model = modelTCN
    return model

if __name__ == '__main__':
    K = 5
    a = 282

    train_data = pd.read_excel(r'the path of train_data', header=None).values
    test_data = pd.read_excel(r'the path of test_data', header=None).values

    df_train = train_data[:, 1:]
    labels_train = train_data[:, 0]
    df_test = test_data[:, 1:]
    labels_test = test_data[:, 0]

    '''
        the name of model：
        'AlexNet'
        'ResNet'
        'MCNN'
        'SqueezeNet'
        'TCN'
    '''

    ModelName = 'AlexNet'
    model = ModelSelect(ModelName)
    cross_validation(df_train, labels_train, df_test, labels_test, K, model,ModelName)
