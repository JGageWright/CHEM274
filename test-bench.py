'''
File for testing code
'''

import numpy as np
import pandas as pd

def fake_EIS(E_DC: float, E_AC: float, freq: np.ndarray, Rm: float, samp_rate: int,
             extra_samps: int=6000, ai1_delay: float=8.5e-6) -> pd.DataFrame:

    # Create an empty dataframe to store data
    df = pd.DataFrame(columns=['E', 'iw', 't', 'f', 'Y', 'Z'])
    for loop_frequency in freq:
        '''
        For each frequency in the array freq, take determine Ecell, Y, Z and append to df
        '''
        num_samps = int(20 * samp_rate / loop_frequency)  # 20 periods at frequency of interest
        time = np.arange(0, (extra_samps + num_samps + extra_samps),
                         1) / samp_rate  # padded with extra samples of start and end
        # Set up potential profile
        program_pot = E_DC * np.ones(num_samps + 2 * extra_samps)  # DC program potential array to apply
        program_pot = program_pot + E_AC * np.sin(2 * np.pi * loop_frequency * time)  # Add on AC program potential


            # create and trim Ecell, iwRm, iw and time arrays
        Ecell = 1  # keep only the middle num_samps worth of Ecell
        iw = 1
        time = 1  # keep only the middle nums amps worth of data

        '''Calculate Admittance'''
        # Ecell inner products with sine and cosine bases
        Ecell_in = np.dot(Ecell, np.sin(2 * np.pi * loop_frequency * time)) / (0.5 * num_samps)
        Ecell_out = np.dot(Ecell, np.cos(2 * np.pi * loop_frequency * time)) / (0.5 * num_samps)

        # Ecell phase and mangnitude
        Ecell_phi = np.arctan(Ecell_out / Ecell_in)
        Ecell_mag = (Ecell_in ** 2 + Ecell_out ** 2) ** 0.5

        # iw inner product with sine and cosine bases
        iw_in = np.dot(iw, np.sin(2 * np.pi * loop_frequency * (time + ai1_delay) + Ecell_phi)) / (num_samps/2)
        iw_out = np.dot(iw, np.cos(2 * np.pi * loop_frequency * (time + ai1_delay) + Ecell_phi)) / (num_samps/2)

        # Admittance
        Y = iw_in / Ecell_mag + 1j*iw_out / Ecell_mag

        # Impedance (note Zout = -Im(Z) in complex impedance analysis)
        Z = Y**-1

        df.loc[len(df)] = Ecell, iw, time, loop_frequency, Y, Z
    return df

df = fake_EIS(1,1,np.array([10,100,1000]), 1000, 100000)
print(df, np.real(df['Y']))

