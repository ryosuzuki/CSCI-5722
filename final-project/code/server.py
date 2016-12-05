# -*- coding: utf-8 -*-

import os
from sqlite3 import dbapi2 as sqlite3
from flask import Flask, request, render_template, send_from_directory, url_for, jsonify

import bovw
import cnn

app = Flask(__name__, static_url_path='')
bovw_data = bovw.load_data('models/bovw/data.pickle')
cnn_model = cnn.load_model()
label_map = cnn.load_data('models/vgg16/label_map.pickle')
name_map = cnn.load_data('models/vgg16/name_map.pickle')
# image_map = cnn.load_data('models/vgg16/image_map.pickle')

@app.route('/')
def index():
  return render_template('index.html', images=bovw_data['sorted_images'][0])

@app.route('/<image_id>')
def show_image(image_id):
  image_id = int(image_id)
  return render_template('index.html', type='image', offset=1, images=bovw_data['sorted_images'][image_id], cluster_map=bovw_data['cluster_map'])

@app.route('/categories/<cluster_id>')
def show_cluster(cluster_id):
  cluster_id = int(cluster_id)
  return render_template('index.html', type='category', offset=0, images=bovw_data['clusters'][cluster_id], cluster_map=bovw_data['cluster_map'])

@app.route('/cnn')
def show_cnn():
  return render_template('cnn.html')

@app.route('/cnn/labels')
def index_label():
  return render_template('index.html', type='label', offset=0, images=[], name_map=name_map)

@app.route('/cnn/labels/<label_id>')
def show_label(label_id):
  return render_template('index.html', type='label', offset=0, images=label_map[label_id], name_map=name_map, label_id=label_id, label_name=label_map[label_id][0]['name'])


@app.route('/upload', methods=['POST'])
def upload():
  file = request.files['file']
  path = os.path.join("tmp", file.filename)
  file.save(path)
  image = cnn.read_image(path)
  label, index, prob = cnn.show_result(cnn_model, image)
  result = {
    'label': label,
    'prob': float(prob),
    'path': path
  }
  return jsonify(result)

@app.route('/favicon.ico')
def send_favicon():
  return send_from_directory('static', 'favicon.ico')

@app.route('/tmp/<path:path>')
def send_tmp(path):
  return send_from_directory('tmp', path)

@app.route('/images/<path:path>')
def send_image(path):
  return send_from_directory('images', path)

@app.route('/static/<path:path>')
def send_static(path):
  return send_from_directory('static', path)

if __name__ == '__main__':
  app.config.update(
    DEBUG=True,
    TEMPLATES_AUTO_RELOAD=True
  )
  app.run()


