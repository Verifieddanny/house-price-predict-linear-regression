import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import os

def load_andpreprocess_data():
    #load data
    housing_price_df = pd.read_csv('Housing.csv')

    #convert categorical variables to numerical
    binary_cols = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea']
    for col in binary_cols:
        housing_price_df[f'{col}_code'] = housing_price_df[col].map({'no': 0, 'yes': 1})
    
    #One-hot encode furnishingstatus
    furnishing_dummies = pd.get_dummies(housing_price_df['furnishingstatus'], prefix='furnishing')
    housing_price_df = pd.concat([housing_price_df, furnishing_dummies], axis=1)

    return housing_price_df

def train_and_save_model():
    df = load_andpreprocess_data()

    #Defining features and target variable
    input_cols = ['area', 'bathrooms', 'stories', 'airconditioning_code', 'parking', 
                 'bedrooms', 'prefarea_code', 'mainroad_code', 'guestroom_code', 
                 'furnishing_furnished', 'basement_code', 'hotwaterheating_code', 
                 'furnishing_semi-furnished', 'furnishing_unfurnished']
    
    X = df[input_cols]
    y = df['price']

    #Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    #Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    #Train model
    model = LinearRegression()
    model.fit(X_train_scaled, y_train)

    #Save model and clear
    os.makedirs('model', exist_ok=True)
    joblib.dump(model, 'model/model.pkl')
    joblib.dump(scaler, 'model/scaler.pkl')

    print("Model and scaler saved successfully.")

if __name__ == "__main__":
    train_and_save_model()