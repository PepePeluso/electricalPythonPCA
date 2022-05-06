import numpy as np
from sklearn.decomposition import PCA
from sklearn import preprocessing
import pandas as pd
from tensorflow.keras.models import load_model
from flask import Flask, render_template, request

model_electrical = load_model("ElectFaultModel.h5")

minmax = pd.read_csv("minmaxElectrical.csv", header=None, sep=";")
datos = minmax.values

def pred(la, lb, lc, Va, Vb, Vc, datos):
  electrical_values = [la, lb, lc, Va, Vb, Vc]
  datos = np.concatenate((datos, [electrical_values], datos), axis=0)
  min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0,1))
  datos = min_max_scaler.fit_transform(datos)
  pca = PCA(n_components= 3)
  datos = pca.fit_transform(datos)
  datos = np.array([datos[2]])
  datos = np.array(datos, "float32")

  prediction = model_electrical.predict(datos)
  if(prediction<0.45 and prediction>0.355):
    return 0
  else:
    return 1

'''
[0.42286977]		0
[0.44456112]		0
[0.4110355 ]		0
[0.36143905]		0
[0.1446026 ]		0
[0.43477112]		0

[0.15861315]		1
[0.08047763]		1
[0.5154683 ] 		1
[0.5272359]		  1
[0.05324695]		1
[0.5154683 ]		1
[0.34944636]		1
'''

app = Flask(__name__)

@app.route("/")
def hello():
    return render_template("index.html")

@app.route("/sub", methods = ["POST"])
def submit():
    # HTML -- py
    if request.method == "POST":
        la =float(request.form["la"])
        lb =float(request.form["lb"])
        lc =float(request.form["lc"])
        Va =float(request.form["Va"])
        Vb =float(request.form["Vb"])
        Vc =float(request.form["Vc"])

    prediction = pred(la,lb,lc,Va,Vb,Vc, datos)
    if prediction==0:
        result = "Your electrical system is working."
        icons = "fa fa-check text-success"
    else:
        result = "Your electrical system had a failure."
        icons = "fa fa-times text-danger"
    # py -- HTML
    
    print(result)

    return render_template("sub.html", result=result, icons = icons)

if __name__ == "__main__":
    app.run()

