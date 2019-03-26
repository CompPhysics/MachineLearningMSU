# Common imports
import os

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

infile = open(data_path("MassEval2016.dat"),'r')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

"""
This is taken from the data file of the mass 2016 evaluation.
All files are 3436 lines long with 124 character per line.
       Headers are 39 lines long.
   col 1     :  Fortran character control: 1 = page feed  0 = line feed
   format    :  a1,i3,i5,i5,i5,1x,a3,a4,1x,f13.5,f11.5,f11.3,f9.3,1x,a2,f11.3,f9.3,1x,i3,1x,f12.5,f11.5
   These formats are reflected in the pandas widths variable below, see the statement
   widths=(1,3,5,5,5,1,3,4,1,13,11,11,9,1,2,11,9,1,3,1,12,11,1),
   Pandas has also a variable header, with length 39 in this case.
"""


# Read the experimental data into a Pandas DataFrame.
df = pd.read_fwf(infile, usecols=(2,3,4,11),
              names=('N', 'Z', 'A', 'avEbind'),
              widths=(1,3,5,5,5,1,3,4,1,13,11,11,9,1,2,11,9,1,3,1,12,11,1),
              header=39,
              index_col=False)

# Extrapolated values are indicated by '#' in place of the decimal place, so
# the avEbind column won't be numeric. Coerce to float and drop these entries.
df['avEbind'] = pd.to_numeric(df['avEbind'], errors='coerce')
df = df.dropna()
# Also convert from keV to MeV.
df['avEbind'] /= 1000

# Group the DataFrame by nucleon number, A.
gdf = df.groupby('A')
# Find the rows of the grouped DataFrame with the maximum binding energy.
maxavEbind = gdf.apply(lambda t: t[t.avEbind==t.avEbind.max()])

# Add a column of estimated binding energies calculated using the SEMF.
#maxavEbind['Eapprox'] = SEMF(maxavEbind['Z'], maxavEbind['N'])

# Generate a plot comparing the experimental with the SEMF values.
fig, ax = plt.subplots()
ax.plot(maxavEbind['A'], maxavEbind['avEbind'], alpha=0.7, lw=2,
            label='Ame2016')

ax.set_xlabel(r'$A = N + Z$')
ax.set_ylabel(r'$E_\mathrm{bind}\,/\mathrm{MeV}$')
ax.legend()
#ax.set_ylim(7,9)
save_fig("Masses2016")
plt.show()




#A = data[:,2]
#bexpt = data[:,3]
# The function we want to fit to, only two terms here
def func(A,a1, a2):
    return a1*A-a2*(A**(2.0/3.0))
# function to perform nonlinear least square with guess for a1 and a2
#popt, pcov = curve_fit(func, A, bexpt, p0 = (16.0, 18.0))
#a1  = popt[0]
#a2 = popt[1]
#liquiddrop = a1*A-a2*(A**(2.0/3.0))


