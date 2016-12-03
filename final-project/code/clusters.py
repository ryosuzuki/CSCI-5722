import functools
import pathlib
import shutil
import pickle
from sklearn.cluster import MiniBatchKMeans
import cv2
import numpy


def read_image(path, size=(320, 240)):
  img = cv2.imread(str(path))
  if img.shape[0] > img.shape[1]:
    return cv2.resize(img, (size[1], size[1]*img.shape[0]//img.shape[1]))
  else:
    return cv2.resize(img, (size[0]*img.shape[1]//img.shape[0], size[0]))


def load_features():
  features = list()
  try:
    print('loading features from pickle...')
    with open('features.pickle', 'rb') as f:
      features = pickle.load(f)
  except TypeError as e:
    print(e)
  return features


def make_visual_words():
  if pathlib.Path('clusters.pickle').is_file():
    print('loading clusters from pickle...')
    with open('cluster_centers.pickle', 'rb') as f:
      clusters = pickle.load(f)
  else:
    features = load_features()
    kmeans = MiniBatchKMeans(max_iter=100, n_clusters=300)
    print('start clustering with 300 clusters...')
    clusters = kmeans.fit(features).cluster_centers_
    print('dumping pickle...')
    with open('cluster_centers.pickle', 'wb') as f:
      pickle.dump(clusters, f)
    print('done')
  return clusters


def make_hist(vws, path):
  hist = numpy.zeros(vws.shape[0])
  for kp in load_kps(path):
    hist[((vws - kp)**2).sum(axis=1).argmin()] += 1
  return hist


def find_nears(vws, hist, n=5, verbose=False):
  nears = []
  for i, path in enumerate(images):
    if verbose:
      print('read {0}/{1}({2:.2%}) {3}'.format(i+1, len(images), (i+1)/len(images), path))

    try:
      h = make_hist(vws, path)
    except TypeError:
      continue

    nears.append((((h - hist)**2).sum(), h, path))
    nears.sort(key=lambda x:x[0])
    nears = nears[:n]
  return nears


if __name__ == '__main__':
  vws = make_visual_words()
  print(vws)

  images = tuple(pathlib.Path('./images').glob('*.jpg'))

  path = images[0]
  img = read_image(path)
  hist = make_hist(vws, path)

  # nears = find_nears(vws, hist, n=20, verbose=True)
  # for x in nears:
  #   print('{0:.2f} - {2}'.format(*x))
  #   shutil.copy(str(x[2]), '{0}{1:.2f}.jpg'.format(output_dir, x[0]))


