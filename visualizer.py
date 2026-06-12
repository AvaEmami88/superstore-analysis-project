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


    def plot_knn_boundary(self,model,scaler,x,y):
        x_scaled = scaler.transform(x)
        x_min, x_max = x_scaled[:, 0].min() - 0.5, x_scaled[:, 0].max() + 0.5
        y_min, y_max = x_scaled[:, 1].min() - 0.5, x_scaled[:, 1].max() + 0.5
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))
        Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        plt.figure(figsize=(10, 8))
        plt.contourf(xx, yy, Z, alpha=0.4, cmap='RdYlBu')
        plt.scatter(x_scaled[:, 0], x_scaled[:, 1], c=y, cmap='RdYlBu', s=50)
        plt.xlabel('Sales (scaled)')
        plt.ylabel('Discount (scaled)')
        plt.title(f'KNN Decision Boundary (k={model.n_neighbors})')
        plt.show()

    def export_for_powerbi(self, df, filename="data.csv"):
        df.to_csv('powerbi/' + filename, index=False)
