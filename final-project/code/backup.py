import functools
import pathlib
import shutil
import pickle

import sklearn.cluster
import cv2
import numpy

input_dir = './images/'
output_dir = './output/'

akaze = cv2.AKAZE_create()
images = tuple(pathlib.Path(input_dir).glob('*.jpg'))
features = list()

def read_image(path, size=(320, 240)):
  img = cv2.imread(str(path))
  if img.shape[0] > img.shape[1]:
    return cv2.resize(img, (size[1], size[1]*img.shape[0]//img.shape[1]))
  else:
    return cv2.resize(img, (size[0]*img.shape[1]//img.shape[0], size[0]))

def compute_descriptor(path):
  kp, descriptor = akaze.detectAndCompute(read_image(path), None)
  descriptor = descriptor.astype(numpy.float32)
  return descriptor

def compute_features():
  print('start computing features..')
  features = list()
  for i, path in enumerate(images):
    print('read {0}/{1}({2:.2%}) {3}'.format(i+1, float(len(images)), (i+1)/float(len(images)), path))
    try:
      descriptor = compute_descriptor(path)
      features += list(descriptor)
    except TypeError as e:
      print(e)
  features = numpy.array(tuple(features))
  print('done')
  return features

def compute_visual_words(features):
  print('start computing visual words..')
  kmeans = sklearn.cluster.MiniBatchKMeans(max_iter=100, n_clusters=128)
  visual_words = kmeans.fit(features).cluster_centers_
  print('done')
  return visual_words

def compute_histgrams(visual_words):
  print('start computing histragms...')
  histragms = list()
  for i, path in enumerate(images):
    print('read {0}/{1}({2:.2%}) {3}'.format(i+1, float(len(images)), (i+1)/float(len(images)), path))
    histgram = numpy.zeros(visual_words.shape[0])
    for descriptor in compute_descriptor(path):
      histgram[((visual_words - descriptor)**2).sum(axis=1).argmin()] += 1
    histgrams.append(hist)
  print('done')
  return histragms

def load_data():
  path = 'data.pickle'
  if (pathlib.Path(path).is_file()):
    print('load data from pickle...')
    with open(path, 'rb') as f:
      data = pickle.load(f)
    print('done')
    return data
  else:
    return False

def save_data(data):
  path = 'data.pickle'
  print('save data as pickle...')
  with open(path, 'wb') as f:
    pickle.dump(data, f)
  print('done')

def find_nears(vws, hist, n=5):
  nears = []
  for i, path in enumerate(images):
    print('read {0}/{1}({2:.2%}) {3}'.format(i+1, len(images), (i+1)/float(len(images)), path))
    try:
      h = make_hist(vws, path)
    except TypeError:
      continue

    nears.append((((h - hist)**2).sum(), h, path))
    nears.sort(key=lambda x:x[0])
    nears = nears[:n]
  return nears


if __name__ == '__main__':
  data = load_data()
  if (not data):
    features = compute_features()
    visual_words = compute_visual_words(features)
    histgrams = compute_histgrams(visual_words)

  save_data(histgrams)


  # nears = find_nears(vws, hist, n=20)
  # for x in nears:
  #   print('{0:.2f} - {2}'.format(*x))
  #   shutil.copy(str(x[2]), '{0}{1:.2f}.jpg'.format(output_dir, x[0]))

