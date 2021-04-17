from flask import Flask
from flask import render_template,request
import pickle
import numpy as np
import pandas as pd
app = Flask(__name__)
model=pickle.load(open('model', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    text1 = request.form['Pregnancies']      ## Fetching each input field value one by one
    text2 = request.form['Glucose'] 
    text3 = request.form['BloodPressure']
    text4 = request.form['SkinThickness']
    text5 = request.form['Insulin']
    text6 = request.form['BMI']
    text7 = request.form['DiabetesPedigreeFunction']
    text8 = request.form['Age']
 
    row_df = pd.DataFrame([pd.Series([text1,text2,text3,text4,text5,text6,text7,text8])])  ### Creating a dataframe using all the values
    print(row_df)
    prediction=model.predict_proba(row_df) 
    output = '{0:.{1}f}'.format(prediction[0][1], 2)    

    if output>str(0.5):
        return render_template('index.html',prediction_text='You have chance of having diabetes.\nProbability of having Diabetes is {}'.format(output)) ## Returning the message for use on the same index.html page
    else:
        return render_template('index.html',prediction_text='You are safe.\n Probability of having diabetes is {}'.format(output)) 




if __name__ == '__main__':
    app.run(debug = True)