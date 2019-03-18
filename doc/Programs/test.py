import numpy as np
import pandas as pd
from IPython.display import display
np.random.seed(100)
# setting up a 9 x 4 matrix
rows = 9
cols = 4
a = np.random.randn(rows,cols)
df = pd.DataFrame(a)
display(df)
print(df.mean())
print(df.std())
display(df**2)


df.columns = ['First', 'Second', 'Third', 'Fourth']
df.index = np.arange(9)

display(df)
print(df['Second'].mean() )
print(df.info())
print(df.describe())

from pylab import plt, mpl
plt.style.use('seaborn')
mpl.rcParams['font.family'] = 'serif'

df.cumsum().plot(lw=2.0, figsize=(10,6))
#plt.show()


df.plot.bar(figsize=(10,6), rot=15)
#plt.show()


b = np.arange(16).reshape((4,4))
print(b)

df1 = pd.DataFrame(b)
