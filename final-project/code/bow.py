import functools
import pathlib
import shutil
import pickle
import scipy.cluster
from sklearn.cluster import MiniBatchKMeans
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


def load_kps(path):
  kp, descriptor = akaze.detectAndCompute(read_image(path), None)
  descriptor = descriptor.astype(numpy.float32)
  return descriptor


def detect_all():
  features = list()
  for i, path in enumerate(images):
    print('read {0}/{1}({2:.2%}) {3}'.format(i+1, float(len(images)), (i+1)/float(len(images)), path))
    try:
      descriptor = load_kps(path)
      features += list(descriptor)
    except TypeError as e:
      print(e)
  return features

def load_features():
  file_path = 'features.pickle'
  if pathlib.Path(file_path).is_file():
    print('load features from pickle...')
    with open(file_path, 'rb') as f:
      features = pickle.load(f)
    print('done')
  else:
    print('start getting features...')
    features = numpy.array(tuple(detect_all()))
    with open(file_path, 'wb') as f:
      print('dumping features...')
      pickle.dump(features, f)
    print('done')
  return features


def load_model(features):
  file_path = 'model.pickle'
  if pathlib.Path(file_path).is_file():
    print('load model from pickle...')
    with open(file_path, 'rb') as f:
      model = pickle.load(f)
  else:
    print('start creating model...')
    codebook, desortion = scipy.cluster.vq.kmeans(features, 128, iter=100)
    kmeans = MiniBatchKMeans(max_iter=100, n_clusters=128)
    model = kmeans.fit(features)
    with open(file_path, 'wb') as f:
      print('dumping model...')
      pickle.dump(model, f)
    print('done')
  return model


# def make_visual_words():
#   return model


def make_hist(vws, path):
  hist = numpy.zeros(vws.shape[0])
  for kp in load_kps(path):
    hist[((vws - kp)**2).sum(axis=1).argmin()] += 1
  return hist


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

def hoge():
  features = load_features()
  print('start clustering')
  centroids, labels = scipy.cluster.vq.kmeans2(features, 5, iter=4)

if __name__ == '__main__':
  print(code)

  # clusters = []
  # labels = model.labels_
  # for i in range(len(labels)):
  #   cluster = []
  #   ii = numpy.where(labels==i)[0]
  #   dd = dists[ii]
  #   di = numpy.vstack([dd,ii]).transpose().tolist()
  #   di.sort()
  #   for d, j in di:
  #     cluster.append(texts[int(j)])
  #   clusters.append(cluster)



  # path = images[30]
  # img = read_image(path)
  # hist = make_hist(vws, path)

  # nears = find_nears(vws, hist, n=20)
  # for x in nears:
  #   print('{0:.2f} - {2}'.format(*x))
  #   shutil.copy(str(x[2]), '{0}{1:.2f}.jpg'.format(output_dir, x[0]))

