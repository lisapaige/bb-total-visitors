import numpy as np
from flask import Flask, render_template, request
import pickle
import os

app = Flask(__name__)
gb = pickle.load(open('model_gb.pkl', 'rb'))
rf = pickle.load(open('model_rf.pkl', 'rb'))
lm = pickle.load(open('model_lm.pkl', 'rb'))

@app.route('/', methods=['GET', 'POST'])
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    snow = request.form["snow value"]
    if snow == 'poor':
        snow = -1
        snow_response=f'for a poor snow season'
    elif snow == 'average':
        snow = 0
        snow_response=f'for an average snow season'
    elif snow == 'good':
        snow = 1
        snow_response=f'for a good snow season'
    
    net_sales = 8.4

    final_features = [np.array([snow, net_sales])]
    gbhat = int(gb.predict(final_features)[0])
    rfhat = int(rf.predict(final_features)[0])
    lmhat = int(lm.predict(final_features)[0])

    prediction = int((gbhat+rfhat+lmhat)/3)

    output = prediction

    return render_template('res.html', 
                            prediction=f'{output/1000:2.0f},000 visitors',
                            snow_response=snow_response)


if __name__=="__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=True, host='0.0.0.0', port=port)
