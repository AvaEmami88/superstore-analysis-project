import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from feature_engineer import FeatureEngineer
class SalesAnalyzer:
    def __init__(self, df):
        self.df = df
    
    def top_products(self):
        top = self.df.groupby('Product Name')['Profit'].sum().sort_values(ascending=False).head(6)
        plt.figure(figsize=(10, 6))
        plt.bar(top.index, top.values)
        plt.title("top products")
        plt.xlabel("product")
        plt.ylabel("Profit")
        plt.xticks(rotation=20)
        plt.show()
        return top
    
    def sales_by_region(self):
        region_sales = self.df.groupby('Region')['Sales'].sum().reset_index()
        return region_sales
    
    def monthly_trend(self):
        monthly = self.df.groupby('Month')['Sales'].sum().reset_index()
        return monthly
    
    def plot_summary(self):
        sales_by_region = self.df.groupby('Region')['Sales'].sum()
        plt.figure(figsize=(10, 8))
        plt.plot(sales_by_region.index,sales_by_region.values)
        plt.title("Sales by region")
        plt.xlabel('region')
        plt.ylabel('sales')
        plt.show()
    def monthly_heatmap(self):
        monthly = self.df.groupby(['Year', 'Month'])['Sales'].sum().unstack()
        month_order = list(range(1, 13))
        monthly = monthly.reindex(columns=month_order)
        plt.figure(figsize=(12, 6))
        sns.heatmap(monthly, linewidths=0.5)
        plt.title('Monthly Sales Heatmap (Year vs Month)')
        plt.xlabel('Month')
        plt.ylabel('Year')
        plt.show()
        

