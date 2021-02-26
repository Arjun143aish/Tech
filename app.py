import numpy as np
import pandas as pd
from flask import Flask,request,render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl','rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods = ['POST'])
def predict():
    """if request.method == 'POST':
        p_i_p = float(request.form['p_i_p'])
        Nop = float(request.form['Nop'])
        Adoh = float(request.form['Adoh'])
        Anoh = float(request.form['Anoh'])
        Estimated = float(request.form['Estimated'])
        
        data = np.array([[p_i_p,Nop,Adoh,Anoh,Estimated]])
        prediction = model.predict(data)
    
        output = round(prediction[0],0)
    
    return render_template('index.html',prediction_text = 'Number of Technicians required per day           is {}'.format(output))"""

    int_features = [float(x)for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)
    
    output = round(prediction[0],0)
    
    return render_template('index.html',prediction_text = 'Number of Technicians required per day           is {}'.format(output)) 

if __name__ == '__main__':
    app.run(debug = True)
    