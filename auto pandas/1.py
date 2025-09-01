import pandas as pd
import numpy as np

Auto = pd.read_csv("Auto.data", delim_whitespace=True)

#View dataset
#print( Auto.head(10) )

#Show column
print(Auto["horsepower"])

#Unique instances
print(np.unique(Auto["horsepower"]))

#Make ? -> np.na
Auto = pd.read_csv("Auto.data", delim_whitespace=True, na_values=['?'])
print(Auto['horsepower'].sum())

#Dataset's attributes
print(Auto.shape)

#Delete na's
Auto_new = Auto.dropna()
#Auto = Auto_new #Overwrite table
print(Auto.columns)

#Get 3 first rows, you can also use head()
print(Auto[:3])

#Get a boolean mask
idx_80 = Auto['year'] > 80
#Use it to display rows
print(Auto[idx_80] )

#View several columns
print(Auto[['mpg', 'horsepower']])

#See indexing
print(Auto.index)

#Rename the rows using values in name (instead of indexes)
#Change indexing to the "name" of rows
Auto = Auto.set_index("name")
print(Auto[:3])
print(Auto.columns) #Column name is no longer here

#Get row by index
rows = ['amc rebel sst', 'ford torino']
print( Auto.loc[rows] )

#Access index by position of rows/columns
print( Auto.iloc[[3, 4], [0, 2, 3]] )

#Get data by name of rows/columns
print( Auto.loc["ford galaxie 500", ['mpg', 'origin']] )

#Weight, origin with year > 80
Auto_buf = Auto.loc[ Auto['year'] > 80, ['weight', 'origin']]

#Use lambda to get the binary mask
print( Auto.loc[lambda df: (df['year'] > 80) & (df['mpg'] > 30), ['weight', 'origin']] )

#Custom lambda function checks each row with str.contains()
Auto.loc[lambda df: (df['displacement'] < 300)
    & (df.index.str.contains('ford')
    | df.index.str.contains('datsun')),
    ['weight', 'origin']
]
#SOME LOOPS!


print("LOOPS!")
for value, weight in zip([1, 2, 3], [1, 2, 3]):
    print(value, weight)