# -*- coding: utf-8 -*-

import os
import bovw
from sqlite3 import dbapi2 as sqlite3
from flask import Flask, render_template, send_from_directory

app = Flask(__name__, static_url_path='')
bovw_data = bovw.load_data('models/bovw/data.pickle')

@app.route('/')
def index():
  return render_template('index.html', images=bovw_data['sorted_images'][0])

@app.route('/<image_id>')
def show(image_id):
  image_id = int(image_id)
  return render_template('index.html', type='show', offset=1, images=bovw_data['sorted_images'][image_id], cluster_map=bovw_data['cluster_map'])

@app.route('/categories/<cluster_id>')
def show_cluster(cluster_id):
  cluster_id = int(cluster_id)
  return render_template('index.html', type='category', offset=0, images=bovw_data['clusters'][cluster_id], cluster_map=bovw_data['cluster_map'])

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


