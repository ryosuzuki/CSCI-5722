import sys
import cv2
import pickle
import pathlib

"""
In this file, we take image as input and extract all the sift descriptors to output.
The data will be further used to generate cluster center of all visual words
"""

sift = cv2.SIFT()
images = tuple(pathlib.Path('./images').glob('*.jpg'))

def read_image(path, size=(320, 240)):
  img = cv2.imread(str(path))
  if img.shape[0] > img.shape[1]:
    return cv2.resize(img, (size[1], size[1]*img.shape[0]//img.shape[1]))
  else:
    return cv2.resize(img, (size[0]*img.shape[1]//img.shape[0], size[0]))


def detect_all(verbose=False):
  features = list()
  for i, path in enumerate(images):
    if verbose:
      print('read {0}/{1}({2:.2%}) {3}'.format(i+1, len(images), (i+1)/float(len(images)), path))
    try:
      img = read_image(path)
      kp, descriptor = sift.detectAndCompute(img, None)
      features += list(descriptor)
    except TypeError as e:
      print(e)
  return features

if __name__ == '__main__':
  features = detect_all(True)
  print('dumping pickle...')
  with open('features.pickle', 'wb') as f:
    pickle.dump(features, f)
  print('done')
