from dataSet import DataSet
import tensorflow as tf
from tensorflow.python.keras.models import Sequential, load_model
from tensorflow.python.keras.layers import Dense, Activation, Convolution2D, MaxPooling2D, Flatten, Dropout
from tensorflow.python.keras.layers import Conv2D
import numpy as np
import os


class Model(object):
    FILE_PATH = ".\model"
    IMAGE_SIZE = 128

    def __init__(self):
        self.model = None

    def read_trainData(self,dataset):
        self.dataset = dataset

    # 一层卷积、一层池化、一层卷积、一层池化、抹平之后进行全链接、最后进行分类
    def build_model(self):
        self.model = Sequential()
        self.model.add(
            Conv2D(
                filters=32,
                kernel_size=(5, 5),
                padding='same',
                # dim_ordering="tf",
                input_shape=self.dataset.X_train.shape[1:]
            )
        )

        self.model.add(Activation('relu'))
        self.model.add(
            MaxPooling2D(
                pool_size=(2, 2),
                strides=(2, 2),
                padding='same'
            )
        )
        

        self.model.add(Convolution2D(filters=64, kernel_size=(5, 5), padding='same'))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))
        

        self.model.add(Flatten())
        self.model.add(Dense(512))
        self.model.add(Activation('relu'))
        

        self.model.add(Dense(self.dataset.num_classes))
        self.model.add(Activation('softmax'))
        # self.model.summary()

    # 进行模型训练的函数，具体的optimizer、loss可以进行不同选择
    def train_model(self):
        self.model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy'])
        print(tf.test.is_gpu_available())

        self.model.fit(self.dataset.X_train,self.dataset.Y_train,epochs=30, batch_size=16)

    def evaluate_model(self):
        print('\nTesting---------------')
        loss, accuracy = self.model.evaluate(self.dataset.X_test, self.dataset.Y_test)

        print('test loss;', loss)
        print('test accuracy:', accuracy)

    def save(self, file_path=FILE_PATH):
        print('Model Saved.')
        self.model.save(file_path)

    def load(self, file_path=FILE_PATH):
        print('Model Loaded.')
        self.model = load_model(file_path)

    def predict(self,img):
        img = img.reshape((1, 1, self.IMAGE_SIZE, self.IMAGE_SIZE))
        img = img.astype('float32')
        img = img/255.0

        result = self.model.predict_proba(img)
        max_index = np.argmax(result)

        return max_index, result[0][max_index]


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    with tf.device("/gpu:0"):
        dataset = DataSet('dataset')
        model = Model()
        model.read_trainData(dataset)
        model.build_model()
        model.train_model()
        model.evaluate_model()
        model.save()














