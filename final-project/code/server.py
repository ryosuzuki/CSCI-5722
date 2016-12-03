# -*- coding: utf-8 -*-

import os
import engine
from sqlite3 import dbapi2 as sqlite3
from flask import Flask, render_template, send_from_directory

app = Flask(__name__, static_url_path='')
data = engine.load_data('data.pickle')

@app.route('/')
def index():
  return render_template('index.html', images=data['sorted_images'][0])

@app.route('/<id>')
def show(id):
  id = int(id)
  print(id)
  return render_template('index.html', images=data['sorted_images'][id])

@app.route('/clusters/<cluster_id>')
def show_cluster(cluster_id):
  cluster_id = int(cluster_id)
  return render_template('index.html', images=data['clusters'][cluster_id])


@app.route('/images/<path:path>')
def send_image(path):
  return send_from_directory('images', path)

if __name__ == '__main__':
  app.config.update(
    DEBUG=True,
    TEMPLATES_AUTO_RELOAD=True
  )
  app.run()


