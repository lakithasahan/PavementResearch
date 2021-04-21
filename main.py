

import pandas as pd
from pycaret.datasets import get_data
from pycaret.regression import *

df=pd.read_excel('data/ANN Training dataset.xlsx')
print(df)

#Get column names
print(df.columns)


#Groupby chainage- Initial value-500

df=df[df['Chainage']==500]
print(df)

exp_name = setup(data = df,  target = 'IWP IRI')
lr = create_model('lr')
print(lr)
