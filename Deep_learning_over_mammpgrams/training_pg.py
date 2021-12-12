# -*- coding: utf-8 -*-
"""
@author: hasnae zerouaoui 
updated by : IHBACH Mohamed Yassine
"""
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import pandas as pd

# this function loads data and its labels from pickel files whitch we already saved (it's so fast and less RAM consuming)
def get_data(img_dim):
    x = np.load("/content/drive/MyDrive/Breast_Cancer/data_"+img_dim+"_mini.npy")
    y = np.load("/content/drive/MyDrive/Breast_Cancer/labels_mini.npy")
    return x[:4000],y[:4000]

def base_conv(base):
    if ( base == 'xception'):
        from tensorflow.keras.applications.xception import Xception
        return Xception(weights='imagenet', include_top = False, input_shape=(224, 224, 3))

    elif ( base == 'nasNet'):
        from tensorflow.keras.applications.nasnet import NASNetMobile
        return NASNetMobile(weights='imagenet', include_top = False, input_shape=(224, 224, 3))

    elif ( base == 'resNet50'):
        from tensorflow.keras.applications.resnet50 import ResNet50
        return ResNet50(weights='imagenet', include_top = False, input_shape=(224, 224, 3))

    elif ( base == 'denseNet'):
        #from tf.keras.applications.densenet import DenseNet201
        return tf.keras.applications.densenet.DenseNet201(weights='imagenet', include_top = False, input_shape=(224, 224, 3))

    elif ( base == 'inceptionResNet'):
        from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2
        return InceptionResNetV2(weights='imagenet', include_top = False, input_shape=(224, 224, 3))

    elif ( base == 'inceptionV3'):
        from tensorflow.keras.applications.inception_v3 import InceptionV3
        return InceptionV3(weights='imagenet', include_top = False, input_shape=(299,299, 3))

    elif ( base == 'vgg16'):
        from tensorflow.keras.applications.vgg16 import VGG16
        return VGG16(weights='imagenet', include_top = False, input_shape=(224, 224, 3))

    elif (base == 'mobileNet'):
      return tf.keras.applications.MobileNetV2(weights='imagenet', include_top = False, input_shape=(224, 224, 3))

    elif ( base == 'vgg19'):
        from tensorflow.keras.applications.vgg19 import VGG19
        return VGG19(weights='imagenet', include_top = False, input_shape=(224, 224, 3))


# this function takes the base model(VGG16 for exemple), defines the model compile it and returns it to us
# note: other parameters are just for testing (the same arch is defined for all) 
def get_model(base_model, trainable=True, trainable_layers=None, lr=0.000005):
    base_model.trainable = trainable
    if (trainable_layers != None):
        for layer in base_model.layers[:-trainable_layers]:
            layer.trainable = False
    model = tf.keras.Sequential()
    model.add(base_model)
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)))
    model.add(tf.keras.layers.Dropout(rate=0.5))
    model.add(tf.keras.layers.Dense(2, activation='softmax', kernel_regularizer=tf.keras.regularizers.l2(0.01)))
    model.compile( optimizer=tf.keras.optimizers.Adam(learning_rate=lr),loss='categorical_crossentropy', metrics=['accuracy'])
    return model
  
def get_baseline():
  model = tf.keras.Sequential()
  model.add(tf.keras.layers.Conv2D(32, (3,3), activation='relu'))
  model.add(tf.keras.layers.MaxPooling2D(2, 2))
  model.add(tf.keras.layers.Conv2D(64, (3,3), activation='relu'))
  model.add(tf.keras.layers.MaxPooling2D(2, 2))
  model.add(tf.keras.layers.Conv2D(128, (3,3), activation='relu'))
  model.add(tf.keras.layers.MaxPooling2D(2, 2))
  model.add(tf.keras.layers.Flatten())
  model.add(tf.keras.layers.Dense(512, activation='relu', 
            kernel_regularizer=tf.keras.regularizers.l2(0.01)))
  model.add(tf.keras.layers.Dropout(rate=0.5))
  model.add(tf.keras.layers.Dense(2, activation='softmax', 
            kernel_regularizer=tf.keras.regularizers.l2(0.01)))
  model.compile( optimizer=tf.keras.optimizers.Adam(learning_rate=0.00001),
            loss='categorical_crossentropy', metrics=['accuracy'])
  return model

def save_folds_vectors(data, tc_algo_folder, name):
  np.save("/content/drive/MyDrive/Breast_Cancer/results/vectors/"+tc_algo_folder+"/"+name,data)


# this function plots variation of accuracy and loss 
def plot_loss_acc(history_dict, epocks, title):
    train_loss = history_dict["loss"]
    train_accuracy = history_dict["accuracy"]
    val_loss = history_dict["val_loss"]
    val_accuracy = history_dict["val_accuracy"]
    
    n = len(val_loss)

    f = plt.figure(figsize=(13,4))
    ax1 = f.add_subplot(121)
    ax2 = f.add_subplot(122)
    
    #ax1.plot(range(epocks), train_loss, label='train_loss')
    #ax1.plot(range(epocks), val_loss, label='val_loss')
    ax1.plot(range(n), train_loss, label='train_loss')
    ax1.plot(range(n), val_loss, label='val_loss')

    ax1.legend()
    ax1.set_xlabel('epochs')
    ax1.set_ylabel('loss')
    ax1.set_title("loss "+title)
    
    #ax2.plot(range(epocks), train_accuracy, label='train_accuracy')
    #ax2.plot(range(epocks), val_accuracy, label='val_accuracy')
    ax2.plot(range(n), train_accuracy, label='train_accuracy')
    ax2.plot(range(n), val_accuracy, label='val_accuracy')

    ax2.legend()
    ax2.set_xlabel('epochs')
    ax2.set_ylabel('accuracy')
    ax2.set_title("accuracy "+title)
    ax2.set_ylim([0, 1.1])

# this function to save our dataframe to excel
def save_df_as_excel(path, dfs, names, startC=1, startR=1):
    writer = pd.ExcelWriter(path)
    for i, df in enumerate(dfs):
        df.to_excel(writer, names[i], startcol=startC, startrow=startR)
    writer.save()
    writer.close()
