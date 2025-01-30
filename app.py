import os

from flask import Flask, render_template, request
import numpy as np
import pickle
import pandas as pd

model_path = os.path.join(os.path.dirname(__file__), "milkgrade1.pkl")

if os.path.exists(model_path):
    model = pickle.load(open(model_path, "rb"))
    print("Model loaded successfully.")
else:
    print("Error: Model file not found!")

app = Flask(__name__)


@app.route("/")
def about():
    return render_template('home.html')



@app.route("/predict")      
def home1():
    return render_template('predict.html')





@app.route("/pred", methods=['POST','GET'])
def predict():
   x = [[x for x in request.form.values()]]
   print(x)
  
   x = np.array(x)
   print(x.shape)
     
     
   print(x)
   pred = model.predict(x)
   print(pred[0])
   return render_template('submit.html', prediction_text=str(pred))

if __name__ == "__main__":
 app.run(debug=False)
