from flask import Flask
from flask import render_template,request
import pickle
import numpy as np
import pandas as pd
app = Flask(__name__)
model = pickle.load(open('regresion.sav', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST', 'GET'])
def predict():
    text1 = request.form['bedrooms']      ## Fetching each input field value one by one
    text2 = request.form['bathrooms'] 
    text3 = request.form['sqft_living']
    text4 = request.form['sqft_lot']
    text5 = request.form['floors']
    text6 = request.form['waterfront']
    text7 = request.form['view']
    text8 = request.form['condition']
    text9 = request.form['sqft_above']
    text10 = request.form['sqft_basement']
    text11= request.form['yr_built']
    text12 = request.form['yr_renovated']
 
    row_df = pd.DataFrame([pd.Series([text1,text2,text3,text4,text5,text6,text7,text8,text9,text10,text11, text12])])  ### Creating a dataframe using all the values
    print(row_df)
    prediction=model.predict(row_df) 
    output=float(prediction)
    return render_template('index.html',prediction_text = "house sold in {}" .format([output]))



if __name__ == '__main__':
    app.run(debug = True)