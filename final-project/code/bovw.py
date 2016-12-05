import functools
import pathlib
import shutil
import pickle

import cv2
import numpy
from sklearn.cluster import MiniBatchKMeans
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics.pairwise import linear_kernel

input_dir = './images/bovw/'
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

def compute_histograms(visual_words):
  print('start computing histograms...')
  histograms = list()
  for i, path in enumerate(images):
    print('read {0}/{1}({2:.2%}) {3}'.format(i+1, float(len(images)), (i+1)/float(len(images)), path))
    histogram = numpy.zeros(visual_words.shape[0])
    for descriptor in compute_descriptor(path):
      histogram[((visual_words - descriptor)**2).sum(axis=1).argmin()] += 1
    histograms.append(histogram)
  print('done')
  tfidf = TfidfTransformer(smooth_idf=False)
  histograms = tfidf.fit_transform(histograms)
  return histograms


def compute_similarities(histograms):
  print('start computing similarities...')
  sorted_images = []
  for i, path in enumerate(images):
    print('read {0}/{1}({2:.2%}) {3}'.format(i+1, len(images), (i+1)/float(len(images)), path))
    cosine_similarities = linear_kernel(histograms[i:i+1], histograms).flatten()
    nears = []
    for j, path_j in enumerate(images):
      similarity = cosine_similarities[j]
      nears.append({ 'id': j, 'similarity': similarity, 'path': path_j})
      nears.sort(key=lambda x:x['similarity'], reverse=True)
    sorted_images.append(nears)
  print('done')
  return sorted_images


def compute_clusters(histograms):
  print('start clustering...')
  clusters = {}
  cluster_map = {}
  kmeans = MiniBatchKMeans(n_clusters=5)
  model = kmeans.fit(histograms)
  centers = model.cluster_centers_
  for i, path in enumerate(images):
    print('read {0}/{1}({2:.2%}) {3}'.format(i+1, len(images), (i+1)/float(len(images)), path))
    cluster_id = model.predict(histograms[i])[0]
    if (not clusters.has_key(cluster_id)):
      clusters[cluster_id] = []

    center = centers[cluster_id]
    vector = histograms[i].toarray()[0]
    distance = numpy.linalg.norm(vector-center)
    clusters[cluster_id].append({ 'id': i, 'path': path, 'distance': distance })
    clusters[cluster_id].sort(key=lambda x:x['distance'])

    cluster_map[i] = cluster_id
  print('done')
  return clusters, cluster_map


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
  path = 'models/bovw/data.pickle'
  if (not pathlib.Path(path).is_file()):
    features = compute_features()
    visual_words = compute_visual_words(features)
    histograms = compute_histograms(visual_words)
    sorted_images = compute_similarities(histograms)
    clusters, cluster_map = compute_clusters(histograms)

    data = {}
    data['sorted_images'] = sorted_images
    data['clusters'] = clusters
    data['cluster_map'] = cluster_map
    save_data(features, 'models/bovw/features.pickle')
    save_data(visual_words, 'models/bovw/visual_words.pickle')
    save_data(histograms, 'models/bovw/histograms.pickle')
    save_data(data, path)

  data = load_data(path)
  print('finish')
