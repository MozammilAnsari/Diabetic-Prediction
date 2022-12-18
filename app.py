from flask import Flask,render_template,request
import pickle
import numpy as np
model = pickle.load(open('diabities.pkl','rb'))
app = Flask(__name__,template_folder='templates')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict',methods = ['POST'])
def predict_diabitics():
    Pregnancies = int(request.form.get('Pregnancies'))
    Glucose = int(request.form.get('Glucose'))
    BloodPressure = int(request.form.get('BloodPressure'))
    SkinThickness = int(request.form.get('SkinThickness'))
    Insulin = int(request.form.get('Insulin'))
    BMI = float(request.form.get('BMI'))
    DiabetesPedigreeFunction = float(request.form.get('DiabetesPedigreeFunction'))
    Age = int(request.form.get('Age'))

    result = model.predict(np.array([Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age]).reshape(1,8))

    if result == 1:
        result = 'Diabitic'
    else:
        result = 'Non-Diabitic'
        
    return render_template('page2.html',result=result)

if __name__ == "__main__":
    app.run(host='0.0.0.0',port=5000)


