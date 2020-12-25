import argparse
from flask import Flask, jsonify, request
from flask import render_template, send_from_directory
import os
import re
import joblib
import socket
import json
import numpy as np
import pandas as pd


## import model specific functions and variables
#from model.model import model_train, model_load, model_predict
#from model.model import MODEL_VERSION, MODEL_VERSION_NOTE
from model import model_train, model_load, model_predict
from model import MODEL_VERSION, MODEL_VERSION_NOTE





app = Flask(__name__)

@app.route("/")
def landing():
    return render_template('index.html')

@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

@app.route('/running', methods=['POST'])
def running():
    return render_template('running.html')

@app.route('/predict', methods=['GET','POST'])
def predict():
   
    
    ## input checking
    if not request.json:
        print("ERROR: API (predict): did not receive request data")
        return jsonify([])

    if 'country' not in request.json:
        print("ERROR API (predict): received request, but no 'country' found within")
        return jsonify([])

    ## set the test flag
    test = False
    if 'mode' in request.json and request.json['mode'] == 'test':
        test = True

    ## extract the query parameters
    country = request.json['country']
    year = request.json['year']
    month = request.json['month']
    day = request.json['day']
        
    ## load model
    data_dir = os.path.join("data","cs-train")
    all_data, all_models = model_load(data_dir=data_dir)
    model = all_models[country]
    
    if not model:
        print("ERROR: model is not available")
        return jsonify([])

    _result = model_predict(country, year, month, day, test=test)
    result = {}
    
    ## convert numpy objects to ensure they are serializable
    for key,item in _result.items():
        if isinstance(item,np.ndarray):
            result[key] = item.tolist()
        else:
            result[key] = item
    
    return(jsonify(result))

@app.route('/train', methods=['GET','POST'])
def train():
    
    ## check for request data
    if not request.json:
        print("ERROR: API (train): did not receive request data")
        return jsonify(False)

    ## set the test flag
    test = False
    if 'mode' in request.json and request.json['mode'] == 'test':
        test = True

    print("... training model")
    data_dir = os.path.join(".", "data","cs-train")
    print(data_dir)
    model = model_train(data_dir, test=test)
    print("... training complete")

    return(jsonify(True))
        
@app.route('/logs/<filename>',methods=['GET'])
def logs(filename):


    if not re.search(".log",filename):
        print("ERROR: API (log): file requested was not a log file: {}".format(filename))
        return jsonify([])

    log_dir = os.path.join(".","logs")
    if not os.path.isdir(log_dir):
        print("ERROR: API (log): cannot find log dir")
        return jsonify([])

    file_path = os.path.join(log_dir,filename)
    if not os.path.exists(file_path):
        print("ERROR: API (log): file requested could not be found: {}".format(filename))
        return jsonify([])
    
    return send_from_directory(log_dir, filename, as_attachment=True)

if __name__ == '__main__':

    ## parse arguments for debug mode
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--debug", action="store_true", help="debug flask")
    args = vars(ap.parse_args())

    if args["debug"]:
        app.run(debug=True, port=8080)
    else:
        app.run(host='0.0.0.0', threaded=True ,port=8080)

