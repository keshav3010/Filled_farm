import numpy as np
import pandas as pd
from datetime import datetime
import random
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from flask import Flask, json,jsonify, render_template
app = Flask(__name__)

commodity_dict = {
    "arhar": "Arhar.csv",
    "bajra": "Bajra.csv",
    "barley": "Barley.csv",
    "copra": "Copra.csv",
    "cotton": "Cotton.csv",
    "sesamum": "Sesamum.csv",
    "gram": "Gram.csv",
    "groundnut": "Groundnut.csv",
    "jowar": "Jowar.csv",
    "maize": "Maize.csv",
    "masoor": "Masoor.csv",
    "moong": "Moong.csv",
    "niger": "Niger.csv",
    "paddy": "Paddy.csv",
    "ragi": "Ragi.csv",
    "rape": "Rape.csv",
    "jute": "Jute.csv",
    "safflower": "Safflower.csv",
    "soyabean": "Soyabean.csv",
    "sugarcane": "Sugarcane.csv",
    "sunflower": "Sunflower.csv",
    "urad": "Urad.csv",
    "wheat": "Wheat.csv"
}

annual_rainfall = [29, 21, 37.5, 30.7, 52.6, 150, 299, 251.7, 179.2, 70.5, 39.8, 10.9]
base = {
    "Paddy": 1245.5,
    "Arhar": 3200,
    "Bajra": 1175,
    "Barley": 980,
    "Copra": 5100,
    "Cotton": 3600,
    "Sesamum": 4200,
    "Gram": 2800,
    "Groundnut": 3700,
    "Jowar": 1520,
    "Maize": 1175,
    "Masoor": 2800,
    "Moong": 3500,
    "Niger": 3500,
    "Ragi": 1500,
    "Rape": 2500,
    "Jute": 1675,
    "Safflower": 2500,
    "Soyabean": 2200,
    "Sugarcane": 2250,
    "Sunflower": 3700,
    "Urad": 4300,
    "Wheat": 1350

}

month = ["January","February","March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]

class Commodity:

    def __init__(self, csv_name):
        self.name = csv_name
        dataset = pd.read_csv(csv_name)
        self.X = dataset.iloc[:, :-1].values
        self.Y = dataset.iloc[:, 3].values

        
        X_train, X_test, Y_train, Y_test = train_test_split(self.X, self.Y, test_size=0.1, random_state=0)

        # Fitting decision tree regression to dataset
        
        depth = random.randrange(7,18)
        self.regressor = DecisionTreeRegressor(max_depth=depth)
        self.regressor.fit(self.X, self.Y)
        
     
    def getPredictedValue(self):
        current_month = datetime.now().month
        current_year = datetime.now().year
        current_rainfall = annual_rainfall[current_month - 1]
        fsa = np.array([current_month,current_year,current_rainfall]).reshape(1, 3)
        y_pred = self.regressor.predict(fsa)[0]
        return y_pred * base[self.name.split('.')[0]] / 100

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/posting')
def posting():
    result = []
    for items in commodity_dict.keys():
        product = Commodity(commodity_dict[items])
        
        pred = product.getPredictedValue()
        dic = {"Name": product.name.split('.')[0],
              "Month": month[datetime.now().month-1],
              "Year": datetime.now().year,
              "Rainfall": annual_rainfall[datetime.now().month - 1],
              "Predicted Price": pred}
        
        result.append(dic)
    #jsonstr = json.dumps(result)
    return jsonify(result)

if __name__ == 'main':
    app.run(debug=True)
    
       

    