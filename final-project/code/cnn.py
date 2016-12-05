from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD
import cv2
import numpy as np
import pathlib
import pickle

def vgg_16(weights_path=None):
  model = Sequential()
  model.add(ZeroPadding2D((1,1),input_shape=(3,224,224)))
  model.add(Convolution2D(64, 3, 3, activation='relu'))
  model.add(ZeroPadding2D((1,1)))
  model.add(Convolution2D(64, 3, 3, activation='relu'))
  model.add(MaxPooling2D((2,2), strides=(2,2)))

  model.add(ZeroPadding2D((1,1)))
  model.add(Convolution2D(128, 3, 3, activation='relu'))
  model.add(ZeroPadding2D((1,1)))
  model.add(Convolution2D(128, 3, 3, activation='relu'))
  model.add(MaxPooling2D((2,2), strides=(2,2)))

  model.add(ZeroPadding2D((1,1)))
  model.add(Convolution2D(256, 3, 3, activation='relu'))
  model.add(ZeroPadding2D((1,1)))
  model.add(Convolution2D(256, 3, 3, activation='relu'))
  model.add(ZeroPadding2D((1,1)))
  model.add(Convolution2D(256, 3, 3, activation='relu'))
  model.add(MaxPooling2D((2,2), strides=(2,2)))

  model.add(ZeroPadding2D((1,1)))
  model.add(Convolution2D(512, 3, 3, activation='relu'))
  model.add(ZeroPadding2D((1,1)))
  model.add(Convolution2D(512, 3, 3, activation='relu'))
  model.add(ZeroPadding2D((1,1)))
  model.add(Convolution2D(512, 3, 3, activation='relu'))
  model.add(MaxPooling2D((2,2), strides=(2,2)))

  model.add(ZeroPadding2D((1,1)))
  model.add(Convolution2D(512, 3, 3, activation='relu'))
  model.add(ZeroPadding2D((1,1)))
  model.add(Convolution2D(512, 3, 3, activation='relu'))
  model.add(ZeroPadding2D((1,1)))
  model.add(Convolution2D(512, 3, 3, activation='relu'))
  model.add(MaxPooling2D((2,2), strides=(2,2)))

  model.add(Flatten())
  model.add(Dense(4096, activation='relu'))
  model.add(Dropout(0.5))
  model.add(Dense(4096, activation='relu'))
  model.add(Dropout(0.5))
  model.add(Dense(1000, activation='softmax'))

  if weights_path:
    model.load_weights(weights_path)

  return model

def read_image(file):
  im = cv2.resize(cv2.imread(file), (224, 224)).astype(np.float32)
  im[:,:,0] -= 103.939
  im[:,:,1] -= 116.779
  im[:,:,2] -= 123.68
  im = im.transpose((2,0,1))
  im = np.expand_dims(im, axis=0)
  return im

def load_model():
  model = vgg_16('models/vgg16/weights.h5')
  sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
  model.compile(optimizer=sgd, loss='categorical_crossentropy')
  return model

def show_result(model, im):
  with open('models/vgg16/synset_words.txt') as f:
    labels = f.readlines()
  out = model.predict(im)
  index = np.argmax(out)
  prob = out[0][index]
  label = labels[index]
  return label, index, prob

def load_data(path):
  print('load data from pickle...')
  with open(path, 'rb') as f:
    data = pickle.load(f)
  print('done')
  return data

def save_data(data, path):
  print('save data as pickle...')
  with open(path, 'wb') as f:
    pickle.dump(data, f)
  print('done')

def classify_images():
  model = load_model()
  images = tupleimages = tuple(pathlib.Path('./images/cnn/').glob('**/*.jpg'))
  label_map = {} # label_id -> image_info
  name_map = {}  # name -> label_id
  image_map = {} # image -> image_info
  for i, path in enumerate(images):
    print('read {0}/{1}({2:.2%}) {3}'.format(i+1, float(len(images)), (i+1)/float(len(images)), path))
    path = str(path)
    try:
      im = read_image(path)
      label, index, prob = show_result(model, im)
      print(label)
      label_id = label.split(' ')[0]
      label_name = ' '.join(label.split(' ')[1:])
      image_info = {
        'id': label_id,
        'path': path,
        'name': label_name,
        'prob': prob
      }
      if not label_map.has_key(label_id):
        label_map[label_id] = []
      label_map[label_id].append(image_info)
      label_map[label_id].sort(key=lambda x:x['prob'], reverse=True)

      name_map[label_name] = {
        'id': label_id,
        'count': len(label_map[label_id])
      }
      image_map[path] = image_info
    except:
      print('skip')
      continue
  name_array = []
  for name in name_map:
    name_info = {
      'id': name_map[name]['id'],
      'name': name,
      'count': name_map[name]['count']
    }
    name_array.append(name_info)
  name_array.sort(key=lambda x:x['count'], reverse=True)
  name_map = name_array
  save_data(label_map, 'models/vgg16/label_map.pickle')
  save_data(name_map, 'models/vgg16/name_map.pickle')
  save_data(image_map, 'models/vgg16/image_map.pickle')

if __name__ == "__main__":
  im = read_image('cat.jpg')
  model = load_model()

