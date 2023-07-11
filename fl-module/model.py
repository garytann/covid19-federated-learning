import tensorflow as tf
from tensorflow.keras.applications import (
    VGG16, MobileNetV2, NASNetMobile, ResNet50
)
from tensorflow.keras.layers import (
    AveragePooling2D,
    Dense,
    Dropout,
    Flatten,
    Input,
    BatchNormalization,
    ReLU,
    LeakyReLU
)
from sklearn.metrics import balanced_accuracy_score
from tensorflow.keras.callbacks import Callback
import tensorflow.keras as K
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
)
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import numpy as np
import os
from utils import fix_layers

MODEL_DIR = '../models'

class CovidLUSModel():
    
    def get_model(
        input_size: tuple = (224, 224, 3),
        hidden_size: int = 64,
        dropout: float = 0.5,
        num_classes: int = 3,
        trainable_layers: int = 1,
        log_softmax: bool = False,
        mc_dropout: bool = False,
        **kwargs
    ):
        act_fn = tf.nn.softmax if not log_softmax else tf.nn.log_softmax

        # load the VGG16 network, ensuring the head FC layer sets are left off
        baseModel = VGG16(
            weights="imagenet",
            include_top=False,
            input_tensor=Input(shape=input_size)
        )
        # construct the head of the model that will be placed on top of the
        # the base model
        headModel = baseModel.output
        headModel = AveragePooling2D(pool_size=(4, 4))(headModel)
        headModel = Flatten(name="flatten")(headModel)
        headModel = Dense(hidden_size)(headModel)
        headModel = BatchNormalization()(headModel)
        headModel = ReLU()(headModel)
        headModel = (
            Dropout(dropout)(headModel, training=True)
            if mc_dropout else Dropout(dropout)(headModel)
        )
        headModel = Dense(num_classes, activation=act_fn)(headModel)

        # place the head FC model on top of the base model
        model = Model(inputs=baseModel.input, outputs=headModel)

        model = fix_layers(model, num_flex_layers=trainable_layers + 8)

        return model


# Define callbacks
earlyStopping = EarlyStopping(
    monitor='loss',
    patience=20,
    verbose=1,
    mode='min',
    restore_best_weights=True
)

mcp_save = ModelCheckpoint(
    filepath = 'best_weights',
    # os.path.join(MODEL_DIR, 'fold_' + str(FOLD) + '_epoch_{epoch:02d}'),
    # os.path.join(MODEL_DIR, 'best_weights'),
    save_best_only=True,
    monitor='val_accuracy',
    mode='max',
    verbose=1
)
reduce_lr_loss = ReduceLROnPlateau(
    monitor='loss',
    factor=0.7,
    patience=7,
    verbose=1,
    min_delta=1e-4,
    mode='min'
)

trainAug = ImageDataGenerator(
        rotation_range=10,
        fill_mode='nearest',
        horizontal_flip=True,   
        vertical_flip=True,
        width_shift_range=0.1,
        height_shift_range=0.1
        )

# A class to show balanced accuracy.
class Metrics(Callback):

    def __init__(self, valid_data, model):
        super(Metrics, self).__init__()
        self.valid_data = valid_data
        self._data = []
        self.model = model

    def on_epoch_end(self, epoch, logs=None):
        # if epoch:
        #     for i in range(1):  # len(self.valid_data)):
        x_test_batch, y_test_batch = self.valid_data

        y_predict = np.asarray(self.model.predict(x_test_batch))

        y_val = np.argmax(y_test_batch, axis=1)
        y_predict = np.argmax(y_predict, axis=1)
        self._data.append(
            {
                'val_balanced': balanced_accuracy_score(y_val, y_predict),
            }
        )
        print(f'Balanced accuracy is: {self._data[-1]}')
        return

    def get_data(self):
        return self._data


    
