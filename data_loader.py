import pandas as pd
class DataLoader: 
    def __init__(self,path):
        self.path = path
        self.df = None
    


    def load_csv(self):
        self.df = pd.read_csv(self.path, encoding='latin-1')
        return self.df
    
    def validate_schema(self):
        print("rows and columns:  ", self.df.shape)
        print("null counts:  ",self.df.isnull().sum())
        print("Data Types:  ",self.df.dtypes)
        return {
            'shape : ': self.df.shape,
            'nulls : ': self.df.isnull().sum().to_dict(),
            'dtypes : ': self.df.dtypes.astype(str).to_dict()
        }
    

    def get_raw_df(self):
        return self.df
    