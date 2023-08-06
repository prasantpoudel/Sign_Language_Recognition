import tensorflow as tf
from tensorflow import keras

from keras.layers import Dense,Dropout
from keras.optimizers import Adam, Adamax

from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.applications import EfficientNetV2B3

import numpy as np
import pandas as pd
import cv2 as cv2
from sklearn.model_selection import train_test_split
import os
import seaborn as sns
sns.set_style('darkgrid')
import logging

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL
logging.getLogger('tensorflow').setLevel(logging.FATAL)


sdir=r'Data'
filepaths=[]
labels=[]
classlist=os.listdir(sdir) 
for klass in classlist:
    classpath=os.path.join(sdir, klass)
    flist=os.listdir(classpath)
    for f in flist:
        fpath=os.path.join(classpath,f)        
        filepaths.append(fpath)
        labels.append(klass)
    Fseries= pd.Series(filepaths, name='filepaths')
    Lseries=pd.Series(labels, name='labels') 
    df=pd.concat([Fseries, Lseries], axis=1)    

# train test split
train_split=.9
valid_split=.1
train_df, valid_df=train_test_split(df, train_size=train_split, shuffle=True, random_state=123)

height=128
width=128
channels=3
batch_size=40
img_shape=(height, width, channels)
img_size=(height, width)

def scalar(img):
    #img=img/127.5-1  #image Normalization
    return img 
trgen=ImageDataGenerator(preprocessing_function=scalar, horizontal_flip=True)
tvgen=ImageDataGenerator(preprocessing_function=scalar)
train_gen=trgen.flow_from_dataframe( train_df, x_col='filepaths', y_col='labels', target_size=img_size, class_mode='categorical',
                                    color_mode='rgb', shuffle=True, batch_size=batch_size)
# test_gen=tvgen.flow_from_dataframe( test_df, x_col='filepaths', y_col='labels', target_size=img_size, class_mode='categorical',
#                                     color_mode='rgb', shuffle=False, batch_size=test_batch_size)
valid_gen=tvgen.flow_from_dataframe( valid_df, x_col='filepaths', y_col='labels', target_size=img_size, class_mode='categorical',
                                    color_mode='rgb', shuffle=True, batch_size=batch_size)
classes=list(train_gen.class_indices.keys())
class_count=len(classes)
train_steps=int(len(train_gen.labels)/batch_size)



pretrained_model = EfficientNetV2B3(
    input_shape=(224,224, 3),
    include_top=False,
    weights='imagenet',
)

for layer in pretrained_model.layers:
    layer.trainable = False

inputs = pretrained_model.input
x = pretrained_model.output
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dense(2048, activation='relu')(x)
x = Dropout(0.4)(x)
x = tf.keras.layers.Dense(1024, activation='relu')(x)
x = Dropout(0.3)(x)
x = tf.keras.layers.Dense(64, activation='relu')(x)
x = Dropout(0.3)(x)


## the output dense depedent upon the output alphabets
outputs = tf.keras.layers.Dense(26, activation='softmax')(x)

model = tf.keras.Model(inputs=inputs, outputs=outputs)

model.compile(optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=0.0001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Define the callbacks
ckpt = ModelCheckpoint('model/model.h5', #model save
                        monitor='val_loss', save_best_only=True, verbose=3)

estop = EarlyStopping(monitor='val_loss', patience=7, verbose=3, restore_best_weights=True)

lr = ReduceLROnPlateau(monitor='val_loss', patience=3, verbose=3)

callbacks = [ckpt, estop, lr]

# Train the model
history = model.fit(train_gen,
                    validation_data=valid_gen,
                    epochs=20,
                    callbacks=callbacks)


