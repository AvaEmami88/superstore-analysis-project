import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.inspection import DecisionBoundaryDisplay
class Visualizer:
    def __init__(self):
        pass
    def plot_regression(self,x_test,y_test,linear_pred,poly2_pred,poly3_pred):
        order = np.argsort(x_test['Month'])
        months = x_test.iloc[order]['Month']
        actual = y_test.iloc[order]
        plt.figure(figsize=(12, 6))
        plt.plot(months,actual,label='Actual Sales',color='black')
        plt.plot(months,linear_pred[order],label='Linear Regression',color='blue')
        plt.plot(months, poly2_pred[order],label='Poly (degree=2)', color='red')
        plt.plot(months,poly3_pred[order],label='Poly (degree=3)',color='purple')
        plt.title('Actual vs Regression Models')
        plt.xlabel('Month')
        plt.ylabel('Sales')
        plt.legend()
        plt.show()
        # monthly_sorted = monthly_data.sort_values('Month').reset_index(drop=True)
        # plt.figure(figsize=(12, 6))
        # plt.plot(monthly_data['Month'], monthly_data['Sales'],label='Actual Sales',color='black')
        # plt.plot(monthly_data['Month'][:len(linear_pred)], linear_pred,label='Linear Regression', color='blue')
        # plt.plot(monthly_data['Month'][:len(poly_pred)], poly_pred, label='Poly(degree=2)', color='red')
        # plt.plot(monthly_data['Month'][:len(poly_pred)],poly_pred,label ='Poly(degree=3)' , color = 'purple')
        # plt.title('Actual vs linear vs polynomial regressions', fontsize=14)
        # plt.xlabel('Month', fontsize=12)
        # plt.ylabel('Sales', fontsize=12)
        # plt.show()


    def plot_knn_boundary(self,x,y):
        DecisionBoundaryDisplay.from_estimator(self.model,x, response_method="predict")
        plt.scatter(x[:, 0], x[:, 1], c=y)
        plt.title(f'KNN marz = {self.model.n_neighbors})')
        plt.show()
    def export_for_powerbi(self, df, filename="data.csv"):
        df.to_csv('powerbi/' + filename, index=False)