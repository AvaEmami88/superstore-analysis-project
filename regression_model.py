import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score


class RegressionModel:
    def __init__(self):
        self.linear = None
        self.poly_models = {}
        self.poly_transformers = {}
        self.scaler = StandardScaler()
        
    def train_linear(self, x, y, test_size=0.2, random_state=42):
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=random_state)
        self.x_test_raw = x_test 
        self.y_test_raw = y_test
        self.x_train_scaled = self.scaler.fit_transform(x_train)
        self.x_test_scaled = self.scaler.transform(x_test)
        self.y_train = y_train
        self.y_test = y_test

        self.linear = LinearRegression()
        self.linear.fit(self.x_train_scaled, self.y_train)
        return self.linear
    
    def train_polynomial(self, degree=2):
        poly = PolynomialFeatures(degree=degree)
        x_train_poly = poly.fit_transform(self.x_train_scaled)
        model = LinearRegression()
        model.fit(x_train_poly, self.y_train)
        self.poly_models[degree] = model
        self.poly_transformers[degree] = poly
        return model
    
    def evaluate(self, degree=None):
        linear_pred = self.linear.predict(self.x_test_scaled)
        linear_r2 = r2_score(self.y_test, linear_pred)
        linear_mse = mean_squared_error(self.y_test, linear_pred)
        return linear_pred, linear_r2, linear_mse
    
    
    def evaluate_poly(self, degree):
        poly = self.poly_transformers[degree]
        model = self.poly_models[degree]

        x_test_poly = poly.transform(self.x_test_scaled)

        y_pred = model.predict(x_test_poly)

        r2 = r2_score(self.y_test, y_pred)
        mse = mean_squared_error(self.y_test, y_pred)

        return y_pred, r2, mse