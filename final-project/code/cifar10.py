import os
import numpy as np
from scipy.misc import toimage
import matplotlib.pyplot as plt

from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import SGD
from keras.utils import np_utils
# from keras.utils.visualize_util import plot
from keras.preprocessing.image import ImageDataGenerator


def plot_cifar10(X, y, result_dir):
  plt.figure()

  nclasses = 10
  pos = 1
  for targetClass in range(nclasses):
    targetIdx = []

    for i in range(len(y)):
      if y[i][0] == targetClass:
        targetIdx.append(i)


    np.random.shuffle(targetIdx)
    for idx in targetIdx[:10]:
      img = toimage(X[idx])
      plt.subplot(10, 10, pos)
      plt.imshow(img)
      plt.axis('off')
      pos += 1

  plt.savefig(os.path.join(result_dir, 'plot.png'))


def plot_history(history, result_dir):
  plt.figure()
  plt.plot(history.history['acc'], marker='.')
  plt.plot(history.history['val_acc'], marker='.')
  plt.title('model accuracy')
  plt.xlabel('epoch')
  plt.ylabel('accuracy')
  plt.grid()
  plt.ylim((0.0, 1.0))
  plt.legend(['train', 'test'], loc='upper left')
  plt.savefig(os.path.join(result_dir, 'acc.png'))

  plt.figure()
  plt.plot(history.history['loss'], marker='.')
  plt.plot(history.history['val_loss'], marker='.')
  plt.title('model loss')
  plt.xlabel('epoch')
  plt.ylabel('loss')
  plt.grid()
  plt.legend(['train', 'test'], loc='upper left')
  plt.savefig(os.path.join(result_dir, 'loss.png'))


if __name__ == '__main__':
  result_dir = 'result'
  if not os.path.exists(result_dir):
    os.mkdir(result_dir)

  batch_size = 128
  nb_classes = 10
  nb_epoch = 50

  data_augmentation = False

  img_rows, img_cols = 32, 32

  img_channels = 3

  (X_train, y_train), (X_test, y_test) = cifar10.load_data()

  plot_cifar10(X_train, y_train, result_dir)

  X_train = X_train.astype('float32')
  X_test = X_test.astype('float32')
  X_train /= 255.0
  X_test /= 255.0

  Y_train = np_utils.to_categorical(y_train, nb_classes)
  Y_test = np_utils.to_categorical(y_test, nb_classes)

  model = Sequential()

  model.add(Convolution2D(32, 3, 3, border_mode='same', input_shape=X_train.shape[1:]))
  model.add(Activation('relu'))
  model.add(Convolution2D(32, 3, 3))
  model.add(Activation('relu'))
  model.add(MaxPooling2D(pool_size=(2, 2)))
  model.add(Dropout(0.25))

  model.add(Convolution2D(64, 3, 3, border_mode='same'))
  model.add(Activation('relu'))
  model.add(Convolution2D(64, 3, 3))
  model.add(Activation('relu'))
  model.add(MaxPooling2D(pool_size=(2, 2)))
  model.add(Dropout(0.25))

  model.add(Flatten())
  model.add(Dense(512))
  model.add(Activation('relu'))
  model.add(Dropout(0.5))

  model.add(Dense(nb_classes))
  model.add(Activation('softmax'))

  model.compile(loss='categorical_crossentropy',
          optimizer='adam',
          metrics=['accuracy'])

  model.summary()
  # plot(model, show_shapes=True, to_file=os.path.join(result_dir, 'model.png'))

  if not data_augmentation:
    print('Not using data augmentation')
    history = model.fit(X_train, Y_train,
              batch_size=batch_size,
              nb_epoch=nb_epoch,
              verbose=1,
              validation_data=(X_test, Y_test),
              shuffle=True)
  else:
    print('Using real-time data augmentation')

    datagen = ImageDataGenerator(
      featurewise_center=False,
      samplewise_center=False,
      featurewise_std_normalization=False,
      samplewise_std_normalization=False,
      zca_whitening=False,
      rotation_range=0,
      width_shift_range=0.1,
      height_shift_range=0.1,
      horizontal_flip=True,
      vertical_flip=False)


    datagen.fit(X_train)


    train_generator = datagen.flow(X_train, Y_train, batch_size=batch_size)


    history = model.fit_generator(train_generator,
                    samples_per_epoch=X_train.shape[0],
                    nb_epoch=nb_epoch,
                    validation_data=(X_test, Y_test))

  model_json = model.to_json()
  with open(os.path.join(result_dir, 'model.json'), 'w') as json_file:
    json_file.write(model_json)
  model.save_weights(os.path.join(result_dir, 'model.h5'))

  plot_history(history, result_dir)

  loss, acc = model.evaluate(X_test, Y_test, verbose=0)
  print('Test loss:', loss)
  print('Test acc:', acc)