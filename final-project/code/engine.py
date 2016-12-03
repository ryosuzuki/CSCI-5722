import functools
import pathlib
import shutil
import pickle

import cv2
import numpy
from sklearn.cluster import MiniBatchKMeans
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics.pairwise import linear_kernel

input_dir = './images/'
images = tuple(pathlib.Path(input_dir).glob('*.jpg'))
features = list()

def read_image(path, size=(320, 240)):
  img = cv2.imread(str(path))
  if img.shape[0] > img.shape[1]:
    return cv2.resize(img, (size[1], size[1]*img.shape[0]//img.shape[1]))
  else:
    return cv2.resize(img, (size[0]*img.shape[1]//img.shape[0], size[0]))

def compute_descriptor(path):
  akaze = cv2.AKAZE_create()
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
  kmeans = MiniBatchKMeans(n_clusters=300)
  visual_words = kmeans.fit(features).cluster_centers_
  print('done')
  return visual_words

def compute_histgrams(visual_words):
  print('start computing histgrams...')
  histgrams = list()
  for i, path in enumerate(images):
    print('read {0}/{1}({2:.2%}) {3}'.format(i+1, float(len(images)), (i+1)/float(len(images)), path))
    histgram = numpy.zeros(visual_words.shape[0])
    for descriptor in compute_descriptor(path):
      histgram[((visual_words - descriptor)**2).sum(axis=1).argmin()] += 1
    histgrams.append(histgram)
  print('done')
  tfidf = TfidfTransformer(smooth_idf=False)
  histgrams = tfidf.fit_transform(histgrams)
  return histgrams


def compute_similarities(histgrams):
  print('start computing similarities...')
  sorted_images = []
  for i, path in enumerate(images):
    print('read {0}/{1}({2:.2%}) {3}'.format(i+1, len(images), (i+1)/float(len(images)), path))
    cosine_similarities = linear_kernel(histgrams[i:i+1], histgrams).flatten()
    nears = []
    for j, path_j in enumerate(images):
      similarity = cosine_similarities[j]
      nears.append({ 'id': j, 'similarity': similarity, 'path': path_j})
      nears.sort(key=lambda x:x['similarity'], reverse=True)
    sorted_images.append(nears)
  print('done')
  return sorted_images


def compute_clusters(histgrams):
  print('start clustering...')
  clusters = {}
  kmeans = MiniBatchKMeans(n_clusters=5)
  model = kmeans.fit(histgrams)
  for i, path in enumerate(images):
    print('read {0}/{1}({2:.2%}) {3}'.format(i+1, len(images), (i+1)/float(len(images)), path))
    cluster_id = model.predict(histgrams[i])[0]
    if (not clusters.has_key(cluster_id)):
      clusters[cluster_id] = []
    clusters[cluster_id].append({ 'id': i, 'path': path})
  print('done')
  return clusters


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


if __name__ == '__main__':
  path = 'data.pickle'
  if (not pathlib.Path(path).is_file()):
    if (not pathlib.Path('features.pickle').is_file()):
      features = compute_features()
      save_data(features, 'features.pickle')
    else:
      features= load_data('features.pickle')

    if (not pathlib.Path('visual_words.pickle').is_file()):
      visual_words = compute_visual_words(features)
      save_data(visual_words, 'visual_words.pickle')
    else:
      visual_words = load_data('visual_words.pickle')

    if (not pathlib.Path('histgrams.pickle').is_file()):
      histgrams = compute_histgrams(visual_words)
      save_data(histgrams, 'histgrams.pickle')
    else:
      histgrams = load_data('histgrams.pickle')

    sorted_images = compute_similarities(histgrams)
    clusters = compute_clusters(histgrams)

    data = {}
    data['sorted_images'] = sorted_images
    data['clusters'] = clusters

    save_data(data, path)
  data = load_data(path)
  print('finish')
