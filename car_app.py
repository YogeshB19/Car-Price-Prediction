from flask import Flask,render_template,request
import pickle
import pandas as pd
import numpy as np
car=pd.read_csv(r'C:\Users\yoges\Car Sale Price Prediction\cleaned_car')
model=pickle.load(open('linear_car.pkl','rb'))

app= Flask(__name__)

@app.route('/',methods=['GET','POST'])
def index():
    Companies=sorted(car['Company'].unique())
    Car_model=sorted(car['Name'].unique())
    Year=sorted(car['Year'].unique())
    Fuel_type=car['Fuel_type'].unique()
    Companies.insert(0,"Select Company")
    return render_template('index.html',Companies=Companies,Car_model=Car_model,Year=Year,Fuel_type=Fuel_type)

@app.route('/predict', methods=['POST'])
def predict():
    Company=request.form.get('Company')
    Car_model=request.form.get('Car_model')
    Year=request.form.get('Year')
    Fuel_type=request.form.get('Fuel_type')
    Kms_driven=int(request.form.get('Kms_driven'))
    prediction=model.predict(pd.DataFrame(columns=['Name','Company','Year','Kms_driven','Fuel_type'],data=np.array([[Car_model,Company,Year,Kms_driven,Fuel_type]]))).round(2)[0]
    return str(prediction)
if __name__=='__main__':
    app.run(debug=True)