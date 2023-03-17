# Importing libraries
from flask import Flask, url_for, render_template, request
import pickle


import warnings
warnings.filterwarnings('ignore')

# Creating an instance of Flask
app=Flask(__name__)

import numpy as np
import pandas as pd

# Importing the preprocessed Salary Dataset
data=pd.read_csv("Preprocessed salarydata.csv")

# Splitting X and y into features and target...y is the dependent variable
X=data.drop('salary',axis=1)
y=data['salary']

# Splitting the data into train and test
import sklearn
from sklearn.model_selection import train_test_split

# Model
X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=42,test_size=.2)

# Creating Gradient Boosting Classification Model
from sklearn.ensemble import GradientBoostingClassifier
grad_boost_model=GradientBoostingClassifier().fit(X_train,y_train)
# Predicting the Test set Result
y_pred_grad=grad_boost_model.predict(X_test)


# Fine Tuning using RandomizedSearchCV
model_random=GradientBoostingClassifier(n_estimators=100,max_depth=3,learning_rate=0.1)
model2=model_random.fit(X_train,y_train)
y_pred_random=model2.predict(X_test)

# Training the preprocessed data with the best Hyperparameters
model_random=GradientBoostingClassifier(n_estimators=100,max_depth=3,learning_rate=0.1)
model_random.fit(X,y)

# Saving model using pickle
pickle.dump(model_random,open('randomcv_grad_model.pkl','wb'))


# Loading the model
model=pickle.load(open('randomcv_grad_model.pkl','rb'))


# Creating homepage
@app.route('/')
def home():
    return render_template('index.html')

# Making Predictions
@app.route('/predict',methods=['POST'])
# @app.route("/", methods=["GET", "POST"])
def predict():
    if request.method == "POST":

        Age = float(request.form['age'])

        Workclass = str(request.form['workclass'])

        Education = str(request.form['education'])

        Marital_status = str(request.form['marital-status'])

        Occupation = str(request.form['occupation'])

        Relationship = str(request.form['relationship'])

        Race = str(request.form['race'])

        Sex= str(request.form['sex'])

        Hours_per_week = float(request.form['hours-per-week'])

        Native_country = str(request.form['native-country'])

        # storing the data in 2-D array
        predict_list = [[Age,Workclass,Education,Marital_status,
                         Occupation,Relationship,Race,Sex,
                         Hours_per_week,Native_country]]
                            

# Predicting the results using the model loaded from a pickle file(randomcv_grad_model.pkl)
        output = model.predict(predict_list)

# Loading the templates for respective outputs (0 or 1)
        if output == 1:
            return render_template('high salary.html')
        else:
            return render_template('low salary.html')

    return render_template('index.html')


# Main driver function
if __name__ == '__main__':
    app.run(debug=True)
