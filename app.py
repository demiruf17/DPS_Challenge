from flask import Flask, request, jsonify
from train import load
import numpy as np


app = Flask(__name__)
app.config["JSON_AS_ASCII"] = False
app.config["JSONIFY_MIMETYPE"] = "application/json; charset=utf-8"

model = load("models/voting_rf_et.pickle")

@app.route('/result',methods = ['POST'])
def result():
    # parse request
    jo = request.get_json()
    year = jo['year']
    month = jo['month']

    input = np.zeros((9,4))
    

    for i in range(3):
        for j in range(3):
            input[i*3 + j,:] = np.array([i, j, year, month])
        

    # make prediction
    value = model.predict(input)
    value = value.sum()

    # return response
    response = jsonify({"prediction":value })
    response.status_code = 200
    response.headers["Content-Type"] = "application/json; charset=utf-8"
    return response


if __name__ == '__main__':
   app.run(debug = True)

