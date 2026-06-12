import pandas as pd
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
class FeatureEngineer:
    def __init__(self, df):
        self.df = df
        self.monthly_data = None
    
    def add_month_year(self):
        self.df['Order Date'] = pd.to_datetime(self.df['Order Date'], format='mixed')
        self.df['Month'] = self.df['Order Date'].dt.month
        self.df['Year'] = self.df['Order Date'].dt.year
        return self.df
    
    def add_profit_margin(self):
        self.df['Profit_Margin'] = self.df['Profit'] / self.df['Sales']
        self.df['Profit_Margin'] = self.df['Profit_Margin'].fillna(0).replace([float('inf'), -float('inf')], 0)
        return self.df
    def aggregate_monthly(self):
        self.monthly_data = self.df.groupby(['Year', 'Month'])[['Sales', 'Profit']].sum().reset_index()
        return self.monthly_data
    
    #*******
    def feature_importance(self, target_column='Sales'):
        numeric_df = self.df.select_dtypes(include=['number'])
        correlations = numeric_df.corr()[target_column].drop(target_column).abs().sort_values(ascending=False)
        plt.figure(figsize=(10, 6))
        correlations.plot(kind='barh')
        plt.xlabel(f'{target_column}')
        plt.title(f'Feature Importance {target_column}')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.show()
        
        print(correlations)
        return correlations
