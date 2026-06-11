from data_loader import DataLoader
from data_cleaner import DataCleaner
from feature_engineer import FeatureEngineer
from sales_analyzer import SalesAnalyzer
from regression_model import RegressionModel
from knn_classifier import KNNClassifier
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
        self.monthly_data =engineer.aggregate_monthly()
        

        #tahlil foroosj
        analyzer= SalesAnalyzer(self.clean_df)
        self.results['sales_analysis']['top_products']=analyzer.top_products()
        self.results['sales_analysis']['sales_by_region'] =analyzer.sales_by_region()
        self.results['sales_analysis']['monthly_trend'] =analyzer.monthly_trend()
        print("top 3 products:",self.results['sales_analysis']['top_products'].head(3))
        
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
        
        best_accuracy = 0
        best_k = 3
        for k in [3, 5, 7]:
            knn = KNNClassifier()
            knn.train(x_train,y_train,x_test,y_test,k=k)
            accuracy = knn.accuracy()
            self.results['knn'][f'k = {k}'] = accuracy
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_k = k
        print(f"best k is {k} and best accuracy is{best_accuracy}")
        return self.results
    


    def save_results(self,filename='results.txt'):
        with open(filename, 'w') as f:
            f.write(str(self.results))



    def generate_csv(self,filename="output_for_powerbi.csv"):
        self.monthly_data.to_csv(filename)
        return filename