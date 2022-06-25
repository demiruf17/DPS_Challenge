import pandas as pd
import numpy as np
from sklearn import preprocessing


class DataModel():
    def __init__(self, file_path):
        self.df = pd.read_csv(file_path)
        
        # Clean the data, drop unused columns
        self._preprocess()

    
    def _preprocess(self):
        # drop the unused columns
        self.df = self.df.drop(columns=list(self.df.columns)[5:])
        
        # drop the rows that contains total yearly sum
        self.df = self.df[~self.df["MONAT"].isin(["Summe"])]


        # extract months
        self.df["MONAT"] = self.df["MONAT"].apply(lambda x : x[-2:])   
        self.df = self.df.reset_index(drop=True)
        
        # relabel the category and type of the accidents
        le_category = preprocessing.LabelEncoder()
        le_category.fit(self.df["MONATSZAHL"].values)
        self.df['MONATSZAHL'] = le_category.transform(self.df["MONATSZAHL"].values)
        

        le_type = preprocessing.LabelEncoder()
        le_type.fit(self.df["AUSPRAEGUNG"].values)
        self.df['AUSPRAEGUNG'] = le_type.transform(self.df["AUSPRAEGUNG"].values)
        

if __name__ == "__main__":
    
    dm = DataModel("data/220511_monatszahlenmonatszahlen2204_verkehrsunfaelle.csv")
    foo = dm.df[dm.df["JAHR"] < 2021].reset_index(drop=True)

    print(foo)

