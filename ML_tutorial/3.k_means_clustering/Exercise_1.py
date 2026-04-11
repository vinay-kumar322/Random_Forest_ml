from sklearn.cluster import KMeans
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

df=pd.read_csv(r"C:\Users\DELL\Documents\income.csv")
print(df.head())

plt.scatter(df.Age,df['Income($)'])
plt.xlabel('Age')
plt.ylabel('Income($)')

km = KMeans(n_clusters=3)
y_predicted = km.fit_predict(df[['Age','Income($)']])
y_predicted

df['cluster'] = y_predicted
df.head()

km.cluster_centers_

df1= df[df.cluster == 0]
df2= df[df.cluster == 1]
df3= df[df.cluster == 3]
plt.scatter(df1.Age,df1['Income($)'],color = 'green')
plt.scatter(df2.Age,df1['Income($)'],color = 'red')
plt.scatter(df3.Age,df1['Income($)'],color = 'blue')
plt.show()
plt.legend()


### preprocessing using minmax scaler
Scaler = MinMaxScaler()                               # hear we converted the values in between order 0-1
Scaler.fit(df[['Income($)']])
df['Income($)'] = Scaler.transform(df[['Income($)']])

Scaler = MinMaxScaler()
Scaler.fit(df[['Age']])
df['Age'] = Scaler.transform(df[['Age']])
df.head()

plt.scatter(df.Age,df['Income($)'])
plt.show()

km = KMeans(n_clusters=3)
y_predicted = km.fit_predict(df[['Age','Income($)']])
y_predicted

df['cluster'] = y_predicted
df.head()

km.cluster_centers_

df1= df[df.cluster == 0]
df2= df[df.cluster == 1]
df3= df[df.cluster == 3]
plt.scatter(df1.Age,df1['Income($)'],color = 'green')
plt.scatter(df2.Age,df1['Income($)'],color = 'red')
plt.scatter(df3.Age,df1['Income($)'],color = 'blue')
plt.show()
plt.legend()





