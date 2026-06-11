import pandas as pd
from sklearn.preprocessing import LabelEncoder
class DataCleaner:
    def __init__(self,df):
        self.df = df

    def remove_nulls(self):
        self.df = self.df.dropna()
        return self.df
    
    def fix_dtypes(self):
        self.df['Order Date'] =  pd.to_datetime(self.df['Order Date'], format='mixed')
        self.df['Ship Date'] = pd.to_datetime(self.df['Ship Date'], format='mixed')
        return self.df
    
    def remove_duplicates(self):
        self.df = self.df.drop_duplicates()
        return self.df
    
    def encode_categoricals(self):
        le = LabelEncoder()
        self.df['Category_encoded'] = le.fit_transform(self.df['Category'])
        self.df['Region_encoded'] = le.fit_transform(self.df['Region'])
        return self.df
    def get_clean_df(self):
        return self.df
