import numpy as np
import pandas as pd
from scipy.stats import qmc
import astropy.units as u
import astropy.constants as c

# 10 Sampled parameters
# 3 more that get attached from stellar temperature

T_eff = [3000, 7000] # K
Rad_p = [0.4, 2.0] # R_E
M_p = [0.4, 10] # M_E
T_eq = [100, 1000] # K
CH4 = [-13, -0.5] # log10(CH4)
H2O = [-13, -0.5] # log10(H2O)
CO2 = [-13, -0.5] # log10(CO2)
O2 = [-13, -0.5] # log10(O2)
O3 = [-13, -0.5] # log10(O3)
N20 = [-13, -0.5] # log10(N2O)

# Create the Bounds parameter
bounds = np.array([T_eff, Rad_p, M_p, T_eq, CH4, H2O, CO2, O2, O3, N20]).T

# Create the Latin Hypercube Sampling object
sampler = qmc.LatinHypercube(d=bounds.shape[1])

# Generate samples
n_samples = 1000000
samples = sampler.random(n_samples)

# Scale samples to the bounds
scaled_samples = qmc.scale(samples, bounds[0], bounds[1])

# Read in the stellar hosts data
stellar_hosts = pd.read_csv('../data/STELLARHOSTS.csv', skiprows=20)

# Some preprocessing
# if there is no mass or log g, remove the row
stellar_hosts = stellar_hosts[stellar_hosts['st_logg'].notna() | (stellar_hosts['st_mass'].notna() & stellar_hosts['st_rad'].notna())]

# Only beteen 3000 and 7000 K
stellar_hosts = stellar_hosts[(stellar_hosts['st_teff'] > 3000) & (stellar_hosts['st_teff'] < 7000)]

# if there is no log g, calculate it from the mass and radius
stellar_hosts['st_logg'] = stellar_hosts['st_logg'].fillna(
    np.log10(((stellar_hosts['st_mass'] * c.M_sun.cgs * c.G.cgs) / (stellar_hosts['st_rad'] * c.R_sun.cgs)**2))
    )

teff_sampled = scaled_samples[:, 0]
rads_sampled = np.zeros_like(teff_sampled)
logg_sampled = np.zeros_like(teff_sampled)
met_sampled = np.zeros_like(teff_sampled)

from tqdm import tqdm

# Find the closest stellar host for each sampled temperature
for i in tqdm(range(len(teff_sampled))):
    # Find the closest stellar host
    idx = (np.abs(stellar_hosts['st_teff'] - teff_sampled[i])).idxmin()
    rads_sampled[i] = stellar_hosts['st_rad'][idx]
    logg_sampled[i] = stellar_hosts['st_logg'][idx]
    met_sampled[i] = stellar_hosts['st_met'][idx]

# Insert the stellar parameters into the sampled data
scaled_samples = np.insert(scaled_samples, 1, met_sampled, axis=1)
scaled_samples = np.insert(scaled_samples, 1, logg_sampled, axis=1)
scaled_samples = np.insert(scaled_samples, 1, rads_sampled, axis=1)

# Array with the filename 'planet_indx.txt' and add all to a dataframe
planet_names = np.array([f'planet_{i}.txt' for i in range(n_samples)])
df = pd.DataFrame(scaled_samples, columns=['T_eff', 'Rad_s', 'logg', 'met', 'Rad_p', 'M_p', 'T_eq', 'CH4', 'H2O', 'CO2', 'O2', 'O3', 'N20'])
df.insert(0, 'file', planet_names)

# Save the DataFrame to a CSV file
df.to_csv('/glade/derecho/scratch/aidenz/data/HERMES_labels/sampled_parameters.csv', index=False, float_format='%.3f')

# Save the samples to a CSV file
# data_formats = ['%d'] + ['%.3f'] * (scaled_samples.shape[1] - 1)
# np.savetxt('/glade/derecho/scratch/aidenz/data/HERMES_labels/sampled_parameters.csv', scaled_samples, delimiter=',',\
#     header='file,T_eff,Rad_s,logg,met,Rad_p,M_p,T_eq,CH4,H2O,CO2,O2,O3,N20', fmt=data_formats)
