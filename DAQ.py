'''
Functions that run the DAQ
'''

import nidaqmx
from nidaqmx import Task
import numpy as np
import pandas as pd


def take_EIS(E_DC: float, E_AC: float, freq: np.ndarray, Rm: float, samp_rate: int,
             extra_samps: int=6000, ai1_delay: float=8.5e-6) -> pd.DataFrame:
    '''
    :rtype: tuple
    :param E_DC: Constant potential to hold during perturbation in V
    :param E_AC: Magnitude of the perturbation potential in V
    :param freq: Array of perturbation potential frequencies in Hz
    :param Rm: Resistance of the measurement resistor in Ω
    :param samp_rate: Sampling rate in Hz
    :param extra_samps: Samples to acquire before and after data used to determine Y and Z
    :param ai1_delay: Empirical delay between ai0 and ai1 acquisitions in s
    :return: DataFrame holding values of Ecell, iw, time, Y, Z for each frequency

    This function creates and writes the program potential array into the ao0 output of the DAQ
    and then initiates its output to Ein of the pstat.

    At the same time, potential acquisitions are made at the ai0 and ai1
    inputs of the DAQ: ai0 measures Ecell while ai1 measures iwRm.

    Complex Y and Z are determined from the middle num_samps
    data values, accounting for the delay of ai1 relative to ai0.
    '''
    # Create an empty dataframe to store data
    df = pd.DataFrame(columns=['E', 'iw', 't', 'f', 'Y', 'Z'])
    for loop_frequency in freq:
        '''
        For each frequency in the array freq, take determine Ecell, Y, Z and append to df
        '''
        # Set up potential program
        num_samps = int(20 * samp_rate / freq)  # 20 periods at frequency of interest
        time = np.arange(0, (extra_samps + num_samps + extra_samps),
                         1) / samp_rate  # padded with extra samples of start and end
        program_pot = E_DC * np.ones(num_samps + 2 * extra_samps)  # DC program potential array to apply
        program_pot = program_pot + E_AC * np.sin(2 * np.pi * freq * time)  # Add on AC program potential

        '''Get device name '''
        # get a list of all devices connected
        all_devices = list(nidaqmx.system.System.local().devices)
        # get name of first device
        dev_name = all_devices[0].name
        # print(dev_name)
        '''initialize potential profile and acquire data'''
        with Task() as task_o, Task() as task_i, Task():
            # add ai0 and ai1 input channel to read Ecell and iwRm
            task_i.ai_channels.add_ai_voltage_chan(dev_name + "/ai0:1", min_val=-10.0, max_val=10.0)
            # add ai0 output ao0 channel for setting the potential profile
            task_o.ao_channels.add_ao_voltage_chan(dev_name + "/ao0", min_val=-10.0, max_val=10.0)

            # Set sampling rates for input and output channels
            task_i.timing.cfg_samp_clk_timing(rate=samp_rate, samps_per_chan=(num_samps + 2 * extra_samps))
            task_o.timing.cfg_samp_clk_timing(rate=samp_rate, samps_per_chan=(num_samps + 2 * extra_samps))

            # set up a digital trigger for the output channel to set the potential
            task_o.triggers.start_trigger.cfg_dig_edge_start_trig('/' + dev_name + '/ai/StartTrigger')
            # define output channel task. Task will only execute when the output channel trigger is activated
            task_o.write(program_pot, auto_start=False)
            task_o.start()

            # Acquire data from input channels. This will trigger the potential profile to be set by the output channel
            acquired_data = task_i.read(number_of_samples_per_channel=(num_samps + 2 * extra_samps),
                                            timeout=nidaqmx.constants.WAIT_INFINITELY)
            # create and trim Ecell, iwRm, iw and time arrays
        Ecell = np.array(acquired_data[0])
        Ecell = Ecell[extra_samps:(extra_samps + num_samps)]  # keep only the middle num_samps worth of Ecell
        iwRm = np.array(acquired_data[1])
        iwRm = iwRm[extra_samps:(extra_samps + num_samps)]  # keep only the middle num_samps worth of iwRm
        iw = iwRm / Rm
        time = time[extra_samps:(extra_samps + num_samps)]  # keep only the middle nums amps worth of data

        '''Calculate Y and Z'''
        # Ecell inner products with sine and cosine bases
        Ecell_in = np.dot(Ecell, np.sin(2 * np.pi * freq * time)) / (0.5 * num_samps)
        Ecell_out = np.dot(Ecell, np.cos(2 * np.pi * freq * time)) / (0.5 * num_samps)

        # Ecell phase and mangnitude
        Ecell_phi = np.arctan(Ecell_out / Ecell_in)
        Ecell_mag = (Ecell_in ** 2 + Ecell_out ** 2) ** 0.5

        # iw inner product with sine and cosine bases
        iw_in = np.dot(iw, np.sin(2 * np.pi * freq * (time + ai1_delay) + Ecell_phi)) / (num_samps/2)
        iw_out = np.dot(iw, np.cos(2 * np.pi * freq * (time + ai1_delay) + Ecell_phi)) / (num_samps/2)

        # Y and Z (note Zout = -Im(Z) in complex impedance analysis)
        Y = iw_in / Ecell_mag + 1j*iw_out / Ecell_mag
        Z = Y**-1

        df.loc[len(df)] = Ecell, iw, time, loop_frequency, Y, Z
    return df

