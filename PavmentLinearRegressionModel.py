import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import PolynomialFeatures

df=pd.read_excel('data/ANN Training dataset.xlsx')
print(df)

#Get column names
print(df.columns)
print(df.dtypes)


#Groupby chainage- Initial value-500
df=df[df['Chainage']==500]
print(df)

years_np=np.array(df['Year']).reshape(-1,1)
IRI_np=np.array(df['IRI']).reshape(-1,1)
print(years_np)
print(IRI_np)

# polynomial_features = PolynomialFeatures(degree=0.5)
# years_np = polynomial_features.fit_transform(years_np)

regr = linear_model.LinearRegression()

regr.fit(years_np, IRI_np)

y_pred = regr.predict(years_np)

print(IRI_np.reshape(-1).tolist())
print(y_pred.reshape(-1).tolist())

# y_true = [0, 1, 2, 0, 1, 2]
# y_pred = [0, 2, 1, 0, 0, 1]
# print(f1_score(y_true, y_pred, average='macro'))

# print(accuracy_score(IRI_np.reshape(-1).tolist(), y_pred.reshape(-1).tolist()))

print('Mean squared error: %.2f'% mean_squared_error(IRI_np.reshape(-1).tolist(), y_pred.reshape(-1).tolist(),squared=False))

#PLotting
fig, (ax1, ax2) = plt.subplots(2)
fig.suptitle('IRI index variation over the years for Chainage=500')
ax1.plot(years_np, IRI_np)

ax2.plot(years_np, y_pred)
plt.xlabel('Date(Years)')
plt.ylabel('IRI')
plt.show()
