import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

rng = np.random.default_rng(1)
A = rng.standard_normal((127, 5)) #Create 127*5 list of random values from normal distribution
M = rng.choice([0, np.nan], p=[0.8,0.2], size=A.shape) #Creaete random table
A += M #Number + nan = nan
D = pd.DataFrame(A, columns=['food',
'bar',
'pickle',
'snack',
'popcorn'])


#print(D[:3]) # Print bc it's not ipynb :(

for col in D.columns:
    template = 'Column "{0}" has {1:.3%} missing values' #Display 3 numbers after the dot
    print(template.format(col, np.isnan(D[col]).mean()))


Auto = pd.read_csv("Auto.data", delim_whitespace=True, na_values=['?'])
Auto = Auto.dropna(subset=['horsepower', 'mpg'])

#print(Auto.columns)

fig, ax = plt.subplots(figsize=(8, 8)) #Subplots very convinient, many windows
ax.plot(Auto['horsepower'], Auto['mpg'], 'o')

ax = Auto.plot.scatter('horsepower', 'mpg') #Draw scatterplot
ax.set_title('Horsepower vs. MPG')

fig = ax.figure
fig.savefig('horsepower_mpg.png') #Save as png

fig, axes = plt.subplots(ncols=3, figsize=(15, 5)) #1X3 graph
Auto.plot.scatter('horsepower', 'mpg', ax=axes[1]) #A lot of graphs on one window, you can use shape=(2, 3) for 6 plots

#plt.show()

#Turn small quantitative data into qualitative
Auto.cylinders = pd.Series(Auto.cylinders, dtype='category')
print( "Type", Auto.cylinders.dtype ) #categorical variable

#Display qualitative data
fig, ax = plt.subplots(figsize=(8, 8))
Auto.boxplot('mpg', by='cylinders', ax=ax)

fig, ax = plt.subplots(figsize=(8, 8)) #Creating new plot(window)
Auto.hist('mpg', color='red', bins=12, ax=ax) #Make more columns!! (bins, red color)

#Visualise all relationship between any 2 variables
pd.plotting.scatter_matrix(Auto)

#Relationship between subset of features
pd.plotting.scatter_matrix(Auto[['mpg',
'displacement',
'weight']])

#plt.show()

#Numerical summary of columns
print( Auto[['mpg', 'weight']].describe() )

#GOOD JOB!