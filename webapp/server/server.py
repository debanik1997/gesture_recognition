import numpy as np
import pandas as pd
import pickle
import random
import json
from flask import Flask, request, jsonify
import urllib

app = Flask(__name__)

def sendResponse(responseObj):
	print(responseObj)
	response = jsonify(responseObj)
	response.headers.add('Access-Control-Allow-Origin', '*')
	response.headers.add('Access-Control-Allow-Methods', 'GET')
	response.headers.add('Access-Control-Allow-Headers', 'accept,content-type,Origin,X-Requested-With,Content-Type,access_token,Accept,Authorization,source')
	response.headers.add('Access-Control-Allow-Credentials', True)
	return response

@app.route("/test", methods=["GET"])
def test():
	return sendResponse({"test": "TEST"})

@app.route("/classify", methods=["POST"])
def classify():
	if not 'image' in request.files:
		return jsonify({'error': 'no image'}), 400

	img_file = request.files.get('image')
	img_name = img_file.filename

	# Write image to static directory and classify

	# Delete image when done
	return jsonify({})

if __name__ == "__main__":	
	app.run(threaded=True)