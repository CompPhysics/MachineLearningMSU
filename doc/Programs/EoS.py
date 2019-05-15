# Common imports
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
# Where to save the figures and data files
PROJECT_ROOT_DIR = "Results"
FIGURE_ID = "Results/FigureFiles"
DATA_ID = "DataFiles/"

if not os.path.exists(PROJECT_ROOT_DIR):
    os.mkdir(PROJECT_ROOT_DIR)

if not os.path.exists(FIGURE_ID):
    os.makedirs(FIGURE_ID)

if not os.path.exists(DATA_ID):
    os.makedirs(DATA_ID)

def image_path(fig_id):
    return os.path.join(FIGURE_ID, fig_id)

def data_path(dat_id):
    return os.path.join(DATA_ID, dat_id)

def save_fig(fig_id):
    plt.savefig(image_path(fig_id) + ".png", format='png')

def r_squared(y, y_hat):
    return 1 - np.sum((y - y_hat) ** 2) / np.sum((y - np.mean(y_hat)) ** 2)

infile = open(data_path("EoS.csv"),'r')

# Read the EoS data as  csv file
EoS = pd.read_csv(infile, names=('Density', 'Energy'))
EoS['Energy'] = pd.to_numeric(EoS['Energy'], errors='coerce')
EoS = EoS.dropna()
Energies = EoS['Energy']
Density = EoS['Density']
print(EoS)
X = np.zeros((len(Density),5))
X[:,4] = Density**(5.0/3.0)
X[:,3] = Density**(4.0/3.0)
X[:,2] = Density
X[:,1] = Density**(2.0/3.0)
X[:,0] = 1

X_train, X_test, y_train, y_test = train_test_split(X, Energies, test_size=0.5)
# matrix inversion to find beta
beta = np.linalg.inv(X_train.T.dot(X_train)).dot(X_train.T).dot(y_train)
# and then make the prediction
ytilde = X_train @ beta
y_predict = X_test @ beta
r_train = r_squared(ytilde, y_train)
r_test = r_squared(y_predict, y_test)
print(r_test, r_train)
"""
EoS['Eapprox']  = y_predict
#print(EoS)
#print(np.mean( (Energies-fity)**2))
# Generate a plot comparing the experimental with the fitted values values.
fig, ax = plt.subplots()
ax.set_xlabel(r'$\rho[\mathrm{fm}^{-3}$')
ax.set_ylabel(r'$E_\mathrm{bind}\,/A$')
ax.plot(EoS['Density'], EoS['Energy'], alpha=0.7, lw=2,
            label='Ame2016')
ax.plot(EoS['Density'], EoS['Eapprox'], alpha=0.7, lw=2, c='m',
            label='Fit')
ax.legend()
save_fig("EoS2016")
plt.show()
"""





