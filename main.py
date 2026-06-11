from data_loader import DataLoader
from data_cleaner import DataCleaner
from feature_engineer import FeatureEngineer
from sales_analyzer import SalesAnalyzer
from regression_model import RegressionModel
from knn_classifier import KNNClassifier
from sklearn.model_selection import train_test_split
from visualizer import Visualizer
from report_pipeline import ReportPipeline
import matplotlib.pyplot as plt

#1- tedad satr o sootoon o anvae data
loader = DataLoader("Superstore.csv")
df = loader.load_csv()         
# loader.validate_schema()   

#2- data cleaning
cleaner = DataCleaner(df)
cleaner.remove_duplicates()
cleaner.remove_nulls()
cleaner.fix_dtypes()
cleaner.encode_categoricals()
clean_df = cleaner.get_clean_df()
# print(clean_df)



correlations = clean_df.select_dtypes(include=['number']).corr()['Sales'].drop('Sales').abs().sort_values(ascending=False)
print(correlations)
plt.figure(figsize=(10, 6))
plt.title('Feature Importance ->Sales')
plt.show()







#3- vizhegi ha
engineer = FeatureEngineer(clean_df)
engineer.add_month_year()
engineer.add_profit_margin()
monthly_data = engineer.aggregate_monthly()
# print(monthly_data.head())

#4-EDA
analyzer = SalesAnalyzer(clean_df)
top_products = analyzer.top_products()
sales_by_region = analyzer.plot_summary()
heatmap = analyzer.monthly_heatmap()
print("top 6 best products : ",top_products)


x = monthly_data[['Year', 'Month']]
y = monthly_data['Sales']

reg = RegressionModel()
reg.train_linear(x, y)
linear_pred, linear_r2, linear_mse = reg.evaluate()
print("r^2:", linear_r2)
print("MSE: ", linear_mse)


reg.train_polynomial(degree=2)
poly2_pred, poly2_r2, poly2_mse = reg.evaluate_poly(degree=2)
print("r^2:", poly2_r2)
print("MSE: ", poly2_mse)


reg.train_polynomial(degree=3)
poly3_pred, poly3_r2, poly3_mse = reg.evaluate_poly(degree=3)
print("r^2:", poly3_r2)
print("MSE:", poly3_mse)


#knn
clean_df['Profitable'] = (clean_df['Profit'] > 0).astype(int)



features = clean_df[['Sales', 'Discount']]
target = clean_df['Profitable']
x_train, x_test, y_train, y_test = train_test_split(features,target,test_size=0.2, random_state=42)

for k in [3, 5, 7]:
    knn = KNNClassifier()
    knn.train(x_train,y_train,x_test,y_test,k=k)
    accuracy=knn.accuracy()
    print(f"k = {k} and accuracy = {accuracy:.4f}")



pipeline = ReportPipeline("Superstore.csv")

results = pipeline.run_all()
pipeline.save_results()
pipeline.generate_csv()


viz = Visualizer()
viz.plot_regression(reg.x_test_raw,reg.y_test_raw, linear_pred,poly2_pred,poly3_pred)


engineer = FeatureEngineer(clean_df)
engineer.feature_importance('Sales')


