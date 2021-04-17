from flask import Flask
from flask import render_template,request
import pickle
import numpy as np
import pandas as pd
app = Flask(__name__)
model=pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    text1 = request.form['ssc_p']      ## Fetching each input field value one by one
    text2 = request.form['hsc_p'] 
    text3 = request.form['degree_p']
    text4 = request.form['etest_p']
    text5 = request.form['mba_p']
    text6 = request.form['gender_M']
    text7 = request.form['ssc_b_Others']
    text8 = request.form['hsc_b_Others']
    text9 = request.form['hsc_s_Commerce']
    text10 = request.form['hsc_s_Science']
    text11= request.form['degree_t_Others']
    text12 = request.form['degree_t_Sci&Tech']
    text13 = request.form['workex_Yes']
    text14 = request.form['specialisation_Mkt&HR']
    

    row_df = pd.DataFrame([pd.Series([text1,text2,text3,text4,text5,text6,text7,text8,text9,text10,text11, 
  text12,text13,text14])])  ### Creating a dataframe using all the values
    print(row_df)
    prediction=model.predict_proba(row_df) 
    output = '{0:.{1}f}'.format(prediction[0][1], 2)    

    if output>str(0.8):
        return render_template('index.html',prediction_text='You have chance of having company select in interview.\nProbability of having selection is {}'.format(output)) ## Returning the message for use on the same index.html page
    else:
        return render_template('index.html',prediction_text='You dont have chance of having company select in interview.\nProbability of having selection is {}'.format(output)) 


if __name__ == '__main__':
    app.run(debug = True)