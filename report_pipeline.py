from data_loader import DataLoader
from data_cleaner import DataCleaner
from feature_engineer import FeatureEngineer
from sales_analyzer import SalesAnalyzer
from regression_model import RegressionModel
from knn_classifier import KNNClassifier
from visualizer import Visualizer
from sklearn.model_selection import train_test_split
import pandas as pd


class ReportPipeline:
    def __init__(self, data_path):
        self.data_path= data_path
        self.df= None
        self.clean_df= None
        self.monthly_data= None
        self.results ={'regression': {},'knn': {},'sales_analysis': {}}
    def run_all(self):
        #estefade as class dataloader baraye bargozari data
        loader= DataLoader(self.data_path)
        self.df= loader.load_csv()
        loader.validate_schema()
        #estefade as data cleaner baraye tamiz kardan data
        cleaner = DataCleaner(self.df)
        cleaner.remove_duplicates()
        cleaner.remove_nulls()
        cleaner.fix_dtypes()
        cleaner.encode_categoricals()
        self.clean_df = cleaner.get_clean_df()
        
        #estefade as feature engineer baraye vizhegi hash
        engineer =FeatureEngineer(self.clean_df)
        engineer.add_month_year()
        engineer.add_profit_margin()
        engineer.feature_importance('Sales')
        self.monthly_data =engineer.aggregate_monthly()
        

        #tahlil foroosj
        analyzer = SalesAnalyzer(self.clean_df)
        top_products = analyzer.top_products()
        sales_by_region = analyzer.plot_summary()
        
        heatmap = analyzer.monthly_heatmap() 
        print("top 6 best products : ",top_products)

        #regression
        
        x =self.monthly_data[['Month']]
        y =self.monthly_data['Sales']
        reg =RegressionModel()
        reg.train_linear(x,y)
        linear_pred, linear_r2, linear_mse = reg.evaluate()

        reg.train_polynomial(degree=2)
        poly2_pred, poly2_r2, poly2_mse = reg.evaluate_poly(degree=2)

        reg.train_polynomial(degree=3)
        poly3_pred, poly3_r2, poly3_mse = reg.evaluate_poly(degree=3)

        print(f"Linear r^2: {linear_r2}")
        print(f"Polynomial degree 2 r^2: {poly2_r2}")
        print(f"Polynomial degree 3 R^2: {poly3_r2}")

        #KNN
        self.clean_df['Profitable'] = (self.clean_df['Profit'] > 0).astype(int)
        features=self.clean_df[['Sales', 'Discount']].fillna(0)
        target=self.clean_df['Profitable']
        x_train,x_test,y_train,y_test =train_test_split(features, target, test_size=0.2, random_state=42)
        
        best_k = 3
        best_acc = 0
        best_knn = None
        for k in [3, 5, 7]:
            knn = KNNClassifier()
            knn.train(x_train, y_train, x_test, y_test, k=k)
            acc = knn.accuracy()
            print(f"k = {k} and accuracy = {acc:.4f}") 
            if acc > best_acc:
                best_acc = acc
                best_k = k
                best_knn = knn
        print(f'best k is {best_k}')
        viz = Visualizer()
        viz.plot_regression(reg.x_test_raw,reg.y_test_raw, linear_pred,poly2_pred,poly3_pred)
        viz.plot_knn_boundary(best_knn.model, best_knn.scaler, x_test.values, y_test.values)
        self.clean_df.to_csv("clean_superstore.csv", index=False)
        return self.results
    


    def save_results(self,filename='results.txt'):
        with open(filename, 'w') as f:
            f.write(str(self.results))


    def generate_csv(self,filename="output_for_powerbi.csv"):
        self.clean_df['YearMonth'] = self.clean_df['Order Date'].dt.to_period('M').astype(str)
        self.clean_df['Profit_Margin_Pct'] = self.clean_df['Profit_Margin'] * 100
        self.clean_df['Is_Profitable'] = (self.clean_df['Profit'] > 0).astype(int)
        self.clean_df.to_csv(filename, index=False)
        print(f"Data exported to {filename} for Power BI")
        print(f"shape: {self.clean_df.shape}")
        return filename
