'''
Functions that run the DAQ
'''

import nidaqmx
from nidaqmx import Task
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from cycler import cycler

'''EIS functions'''
def take_EIS_custom_array(E_DC: float, E_AC: float, freq: np.ndarray, Rm: float, Cm:float, samp_rate: int=100000,
             extra_samps: int=6000, ai1_delay: float=8.5e-6) -> tuple:
    '''
    :rtype: tuple
    :param E_DC: Constant potential to hold during perturbation in V
    :param E_AC: Magnitude of the perturbation potential in V
    :param freq: Array of perturbation potential frequencies in Hz
    :param Rm: Measurement resistance in Ω
    :param Cm: Measurement capacitance in F
    :param samp_rate: Sampling rate in Hz
    :param extra_samps: Samples to acquire before and after data used to determine Y and Z
    :param ai1_delay: Empirical delay between ai0 and ai1 acquisitions in s
    :return: Tuple of DataFrames holding values of data (f, Yre, Yim, Zre, Zim) and parameters,
     as well as list of dataframes holding raw data for each frequency.

    Creates and writes the program potential array into the ao0 output of the DAQ
    and then initiates its output to Ein of the pstat.

    At the same time, potential acquisitions are made at the ai0 and ai1
    inputs of the DAQ: ai0 measures Ecell while ai1 measures iwRm.

    Complex Y and Z are determined from the middle num_samps
    data values, accounting for the delay of ai1 relative to ai0.
    '''
    # Create an empty dataframe to store data
    data = pd.DataFrame(columns=['f', 'Yre', 'Yim', 'Zre', 'Zim'])

    params = pd.DataFrame({'parameter': ['E_DC', 'E_AC', 'freq_array', 'Rm', 'Cm', 'samp_rate', 'extra_samps', 'ai1_delay'],
                           'value': [E_DC, E_AC, freq, Rm, Cm, samp_rate, extra_samps, "{:e}".format(ai1_delay)]})
    opt = []  # hold raw Ecell, iw, t as optional dataframes
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

        # Admittance
        Y = iw_in / Ecell_mag + 1j * iw_out / Ecell_mag
        Yre = np.real(Y)
        Yim = np.imag(Y)

        # Impedance (note Zout = -Im(Z) in complex impedance analysis)
        Z = Y ** -1
        Zre = np.real(Z)
        Zim = np.imag(Z)

        # Append data to dataframe
        data.loc[len(data)] = freq, Yre, Yim, Zre, Zim
        raw = pd.DataFrame({'Ecell_mag': Ecell_mag, 'Ecell_phi': Ecell_phi,
                            'Ecell_re': Ecell_in, 'Ecell_im': Ecell_out,
                            'iw_re': iw_in, 'iw_im': iw_out,
                            't': time}, index=np.arange(time.size))
        opt.append(raw)
    return data, params, opt


def take_EIS(E_DC: float, E_AC: float, low_freq: int, Rm: float, Cm: float, samp_rate: int=100000,
             extra_samps: int=6000, ai1_delay: float=8.5e-6) -> tuple:
    '''
    :rtype: tuple
    :param E_DC: Constant potential to hold during perturbation in V
    :param E_AC: Magnitude of the perturbation potential in V
    :param low_freq: Lowest perturbation potential frequency in Hz. The highest possible is always measured.
    :param Rm: Measurement resistance in Ω
    :param Cm: Measurement capacitance in F
    :param samp_rate: Sampling rate in Hz
    :param extra_samps: Samples to acquire before and after data used to determine Y and Z
    :param ai1_delay: Empirical delay between ai0 and ai1 acquisitions in s
    :return: Tuple of DataFrames holding values of data (f, Yre, Yim, Zre, Zim) and parameters,
     as well as list of dataframes holding raw data for each frequency.

    Creates and writes the program potential array into the ao0 output of the DAQ
    and then initiates its output to Ein of the pstat.

    At the same time, potential acquisitions are made at the ai0 and ai1
    inputs of the DAQ: ai0 measures Ecell while ai1 measures iwRm.

    Complex Y and Z are determined from the middle num_samps
    data values, accounting for the delay of ai1 relative to ai0.

    Creates live plots along the way.
    '''
    # Create an empty dataframe to store data
    data = pd.DataFrame(columns=['f', 'Yre', 'Yim', 'Zre', 'Zim'])

    # Store parameters
    params = pd.DataFrame({'parameter': ['E_DC', 'E_AC', 'low_freq', 'Rm', 'Cm', 'samp_rate', 'extra_samps', 'ai1_delay'],
                           'value': [E_DC, E_AC, low_freq, Rm, Cm, samp_rate, extra_samps, ai1_delay]})
    opt = []  # hold raw Ecell, iw, t as optional dataframes

    # Set up live plots
    cc1 = (cycler(color=list('rgbcmy')) *
           cycler(linestyle=['-', '--']))
    cc2 = (cycler(color=list('rgbcmy')))

    # set up a 2 x 2 grid for the admittance plot
    grid = plt.GridSpec(2, 2, wspace=0.03, hspace=0.01)
    fig2 = plt.figure("Figure 2", figsize=(12, 7))
    ax1 = fig2.add_subplot(grid[0, 0])  # real admittance
    ax2 = fig2.add_subplot(grid[1, 0], sharex=ax1)  # imaginary admittance
    ax3 = fig2.add_subplot(grid[0, 1])  # potential vs time
    ax4 = fig2.add_subplot(grid[1, 1], sharex=ax3)  # current vs time
    fig2.subplots_adjust(left=0.1, bottom=0.25, right=0.75, top=0.9, wspace=0, hspace=0.03)
    plt.ion()
    fig2.show()
    fig2.canvas.draw()

    # manually add legend
    lines = Line2D([0], [0], color='red', label='Real', linewidth=1, linestyle='--')
    lines2 = Line2D([0], [0], color='red', label='Imag', linewidth=1, linestyle='-')

    # set up Nyquist plot
    fig3 = plt.figure("Figure 3", figsize=(5, 5))
    ax5 = fig3.add_subplot()
    fig3.subplots_adjust(left=0.15, bottom=0.25, right=0.8, top=0.85, wspace=0, hspace=0.03)

    freq = samp_rate / 16 # Always > 4 samples per period
    while freq >= low_freq:
        '''
        For each frequency, take determine Ecell, Y, Z and append to data with t and f
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

        # Admittance
        Y = iw_in / Ecell_mag + 1j * iw_out / Ecell_mag
        Yre = np.real(Y)
        Yim = np.imag(Y)

        # Impedance (note Zout = -Im(Z) in complex impedance analysis)
        Z = Y ** -1
        Zre = np.real(Z)
        Zim = np.imag(Z)

        # Append data to dataframe
        data.loc[len(data)] = freq, Yre, Yim, Zre, Zim
        raw = pd.DataFrame({'Ecell_mag': Ecell_mag, 'Ecell_phi': Ecell_phi,
                            'Ecell_re': Ecell_in, 'Ecell_im': Ecell_out,
                            'iw_re': iw_in, 'iw_im': iw_out,
                            't': time}, index=np.arange(time.size))

        opt.append(raw)

        '''Update live plots'''
        # subplot 1
        ax1.set_title('Electrochemical Impedance Spectra')
        ax1.tick_params(axis='both', which='both', direction='in', right=True, top=True)
        ax1.tick_params(labelbottom=False)
        ax1.set_prop_cycle(cc1)
        ax1.plot(data['f'], data['Yre'],
                 marker='.')  # , label = '$R_{ct}$=' + str(Rct0) +'$\Omega; R_{u}$=' + str(Ru) +'$\Omega$ Real')
        ax1.plot(data['f'], -data['Yim'],
                 marker='.')  # , label = '$R_{ct}$=' + str(Rct0) +'$\Omega; R_{u}$=' + str(Ru) +'$\Omega$ Imaginary')
        ax1.set_ylabel('Admittance / S')
        ax1.set_xscale('log')
        ax1.ticklabel_format(axis='y', style='sci', scilimits=(-2, 3))
        ax1.legend(handles=[lines, lines2])
        # ax1.legend()
        # ax1.set_yscale('log')

        # subplot 2
        ax2.tick_params(axis='both', which='both', direction='in', right=True, top=True)
        ax2.set_prop_cycle(cc1)
        ax2.plot(data['f'], data['Zre'], marker='.')
        ax2.plot(data['f'], -data['Zim'], marker='.')
        ax2.set_ylabel('Impedance / $\Omega$')
        ax2.set_xlabel('Periodic Frequency / Hz')
        ax2.set_xscale('log')
        # ax2.ticklabel_format(axis = 'y', style='sci', scilimits = (-2, 3))

        # clear subplots 3 and 4 to plot potential and current from most recent frequency
        ax3.clear()
        ax4.clear()

        # subplot 3
        ax3.set_title('$E_{cell}$ and $i_w$ vs time')
        ax3.tick_params(axis='both', which='both', direction='in')
        ax3.set_prop_cycle(cc1)
        ax3.plot(time, Ecell)
        ax3.set_ylabel('Potential / V')
        ax3.set_xlabel('Time / sec')
        ax3.yaxis.set_ticks_position("right")
        ax3.yaxis.set_label_position("right")
        ax3.ticklabel_format(axis='y', style='sci', scilimits=(-2, 2))
        # ax3.axes.xaxis.set_ticklabels([])

        # ax3.set_xscale('log')
        # ax3.ticklabel_format(axis = 'y', style='sci', scilimits = (-2, 3))

        # subplot 4
        ax4.tick_params(axis='both', which='both', direction='in')
        ax4.set_prop_cycle(cc1)
        ax4.plot(time, iw, color='blue')
        ax4.set_ylabel('Current / A')
        ax4.set_xlabel('Time / sec')
        ax4.ticklabel_format(axis='y', style='sci', scilimits=(-2, 2))
        ax4.yaxis.set_ticks_position("right")
        ax4.yaxis.set_label_position("right")

        # ax3.set_xscale('log')
        # ax3.ticklabel_format(axis = 'y', style='sci', scilimits = (-2, 3))

        # fig2.canvas.draw()

        # ax2.set_yscale('log')

        # subplot 3
        # fig3 = plt.figure("Figure 3", figsize = (4, 4))
        # fig3.subplots_adjust(left=0.15, bottom=0.1, right=0.95, top=0.9, wspace=0, hspace=0.03)
        # ax3 = fig3.add_subplot()

        ax5.set_title('Nyquist Plot')
        ax5.tick_params(axis='both', which='both', direction='in', right=True, top=True)
        ax5.set_prop_cycle(cc2)
        ax5.plot(data['Zre'], -data['Zim'], marker='.')
        ax5.set_ylabel('$-Z_{Im}$ / $\Omega$')
        ax5.set_xlabel('$Z_{Re}$ / $\Omega$')
        ax5.set_aspect('equal', 'box')

        if (max(data['Zre']) >= max(-data['Zim'])):
            axes_max = max(data['Zre'])
        else:
            axes_max = max(-data['Zim'])
        ax5.set_xlim([0, axes_max])
        ax5.set_ylim([0, axes_max])
        ax5.ticklabel_format(axis='y', style='sci', scilimits=(-2, 3))
        ax5.ticklabel_format(axis='x', style='sci', scilimits=(-2, 3))
        # ax3.legend()

        # update plots
        # ax3.clear()
        # ax4.clear()
        fig2.canvas.draw()
        fig3.canvas.draw()

        # Update the frequency and continue the loop
        freq = freq / 2  # Separating frequencies by factors of 2 gives a bit more than 3 values per decade
    return data, params, opt

'''CV and helper functions'''
def CV_potential_profile(f_start_pot: float, f_end_pot: float, samp_rate: int,
                         scan_rate: float, h_time: float, buffer_size: int=3600) -> tuple:
    '''
    :param f_start_pot: Initial potential in V
    :param f_end_pot: Vertex potential in V
    :param samp_rate: Sampling rate in samples/s; Use an integral multiple of 120/s and at least 3600 per volt
    :param scan_rate: Scan rate in V/s
    :param h_time: Hold time before forward sweep in s
    :param buffer_size: Samples stored in buffer before callback
    :return: Tuple of (Potential profile array, Total number of samples, Scan Rate)

    Returns the potential profile for linear sweeps from f_start_pot to f_end_pot and back.
    Initializes the profile on the DAQ, which sets it to begin holding at f_start_pot.
    Graphs the potential profile for user validation.
    '''

    # The return sweep is the reverse of the forward sweep
    r_start_pot = f_end_pot
    r_end_pot = f_start_pot

    '''Total time required for each section'''
    f_time = np.abs((f_end_pot - f_start_pot)/scan_rate)
    r_time = np.abs((r_end_pot - r_start_pot)/scan_rate)

    '''Voltage array to be set for each section'''
    h_profile = np.linspace(f_start_pot, f_start_pot, int(samp_rate*h_time) )
    f_profile = np.linspace(f_start_pot, f_end_pot, int(samp_rate*f_time) )
    r_profile = np.linspace(r_start_pot, r_end_pot, int(samp_rate*r_time) )
    return_profile = np.linspace(r_start_pot, r_end_pot, int(samp_rate*r_time))

    # potential profile is simply the individual potential profiles of each sections combined
    pot_profile = np.concatenate((h_profile, f_profile, r_profile))

    '''Add extra samples to round out pot_profile nicely. 
    buffer_size has been changed to an optional argument'''
    samp_num_tot = len(pot_profile) # total sample number
    # buffer size (must be a factor of samp_num_tot). Buffer size is set to the largest number that is at least
    # 15 times smaller than 'samp_num_tot'. This ensures we have at least 15 updates per CV run
    # buffer_size = calc_closest_factor(samp_num_tot, round(samp_num_tot/15))
    # buffer_size = 3600 # round(200/scan_rate)

    # round off sample size
    additional_hold_sample = 0
    # round off sample size to be a multiple of round(200/scan_rate)
    n = additional_hold_sample + samp_num_tot
    while n%buffer_size != 0:
        additional_hold_sample += 1
        n = additional_hold_sample + samp_num_tot

    # additional hold to keep total scan number a multiple of the buffer size
    h2_profile = np.linspace(f_start_pot, f_start_pot, additional_hold_sample)

    # recalculate potential profile and total sample number
    pot_profile = np.concatenate((h2_profile, pot_profile))
    samp_num_tot = len(pot_profile)

    '''Check potential profile to be set'''
    plt.title('CV Program Potential', fontsize = 16)
    plt.xlabel('Time / s', fontsize = 16)
    plt.ylabel('$E_{\mathrm{in}}$ / V', fontsize = 16)
    plt.tick_params(axis='both',which='both',direction='in',right=True, top=True)
    plt.plot(np.arange(0, len(pot_profile), 1)/samp_rate, pot_profile)

    '''Send the profile to the DAQ'''
    # get a list of all devices connected
    all_devices = list(nidaqmx.system.System.local().devices)
    # get name of first device
    dev_name = all_devices[0].name
    # print(dev_name)

    ''' add DAQ channels and define measurement parameters'''
    with nidaqmx.Task() as task_i, nidaqmx.Task() as task_o:
        # add ai0 & ai1 input channels for reading potentials. add ao0 output channels for setting potential
        task_i.ai_channels.add_ai_voltage_chan(dev_name + "/ai0:1")
        task_o.ao_channels.add_ao_voltage_chan(dev_name + "/ao0", min_val=-10.0, max_val=10.0)
        task_o.write(f_start_pot)
        task_o.start()

    return pot_profile, samp_num_tot, scan_rate

def calc_closest_factor(num, fac) -> int:
    '''
    :param num: number you want to find factors for
    :param fac: upper bound for factors searched
    :return: highest factor of 'num' less than or equal to 'fac'

    Calculates the highest factor of the number 'num' less than or equal to 'fac'
    '''
    while num % fac != 0:
        fac -= 1
    return int(fac)

def take_CV(pot_profile: np.ndarray, samp_num_tot: int, scan_rate: float, Rm: float,
            Cm: float, buffer_size: int = 3600, samp_rate: int = 3600) -> tuple:
    '''
    :param pot_profile: Array of potentials returned by CV_potential_profile
    :param samp_num_tot: Total number of samples returned by CV_potential_profile
    :param scan_rate: Scan rate in V/s
    :param Rm: Measurement resistance in Ω
    :param Cm: Measurement capacitance in F
    :param buffer_size: Samples stored in buffer before callback
    :param samp_rate: Sampling rate in samples/s; Use an integral multiple of 120/s and at least 3600 per volt
    :return: Tuple of DataFrames holding values of data (Ecell, iw, time, f, Y, Z) and parameters.

    This function inputs a potential profile into the ao0 output of the
    potentiostat to set Ein, which then sets Ecell when the counter electrode is connected.

    At the same time potential acquisitions are made on input channels ai0 and ai1 to measure
    Ecell and iwRm.

    The sweep rate, initial hold time and program potential initial value and vertices should be
    defined CV_potential_profile() and passed into this function.

    Internal collection parameters:
        total_data_WE = cell potential during program potential (ai0)
        total_data_RM = iwRm during program potential (ai1)
        np.array(total_data_RM)/Rm = iwRm during the program potential
        np.abs(np.arange(0, len(total_data_WE), 1)/samp_rate) = time array during the program potential
    '''

    '''Get device name '''
    # get a list of all devices connected
    all_devices = list(nidaqmx.system.System.local().devices)
    # get name of first device
    dev_name = all_devices[0].name
    # print(dev_name)

    ''' add DAQ channels and define measurement parameters'''
    with nidaqmx.Task() as task_i, nidaqmx.Task() as task_o:
        # add ai0 & ai1 input channels for reading potentials. add ao0 output channels for setting potential
        task_i.ai_channels.add_ai_voltage_chan(dev_name + "/ai0:1", min_val=-10.0, max_val=10.0)
        task_o.ao_channels.add_ao_voltage_chan(dev_name + "/ao0", min_val=-10.0, max_val=10.0)

        # define sampling rate and total samples acquired per channel for input & output channels
        task_i.timing.cfg_samp_clk_timing(rate=samp_rate, samps_per_chan=samp_num_tot)
        task_o.timing.cfg_samp_clk_timing(rate=samp_rate, samps_per_chan=samp_num_tot)

        # set up a digital trigger for the output channel to set the potential.  Should be commented out for myDAQs
        # task_o.triggers.start_trigger.cfg_dig_edge_start_trig('/'+ dev_name +'/ai/StartTrigger')

        # create empty lists to populate
        total_data_WE = []  # cell potential in V
        total_data_RM = []  # iwRm potential in V

        # define output channel task. Task will only execute when the output channel trigger is activated
        task_o.write(pot_profile)  # , auto_start = False)
        task_o.start()

        ''' Set up plot during data acquisition'''
        # set up a 2 x 2 grid for the plot
        grid = plt.GridSpec(2, 1, wspace=0.3, hspace=0.2)
        fig = plt.figure(figsize=(14, 12))

        # set right edge of plot to be at 80% of fig width and bottom to be at 20% of fig height to fit everything.
        plt.subplots_adjust(right=0.8)
        plt.subplots_adjust(bottom=0.2)

        # Define positions of 3 subplots
        ax1 = fig.add_subplot(grid[1, 0])
        ax2 = fig.add_subplot(grid[0, 0])
        # ax3 = fig.add_subplot(grid[1, 1])#, sharex = ax2)
        plt.ion()
        fig.show()
        fig.canvas.draw()

        '''Encapsulate buffer callback function'''
        def cont_read(task_handle, every_n_samples_event_type,
                      number_of_samples, callback_data):
            '''
            Define a 'callback' function to execute when the buffer is full

            When this funtion is called a subset of samples are acquired at ai0 and ai1 and then appended
            to the lists 'total_data_WE'and 'total_data_RM'. Then the CV plot is updated with this new data.
             '''

            # Acquire subset of samples and store data in a temporary list
            temp_samples = task_i.read(number_of_samples_per_channel=buffer_size)
            # add acquired data to list storing all data
            total_data_WE.extend(temp_samples[0])
            total_data_RM.extend(temp_samples[1])

            # calculate time profile (for plotting)
            total_time_profile = np.abs(np.arange(0, len(total_data_WE), 1) / samp_rate)
            # calculate current at Rm (for plotting)
            Rm_current = np.array(total_data_RM) / Rm

            # Return size of 'total_data' and update subplots every time buffer is full
            # print(len(total_data_RM))

            ax1.clear()
            ax1.set_title('Cyclic Voltammogram', fontsize=16)
            ax1.tick_params(axis='both', which='both', direction='in', right=True, top=True)
            ax1.set_xlabel('$E_{\mathrm{cell}}$ / V', fontsize=16)
            ax1.set_ylabel('$i_{\mathrm{w}}$ / A', fontsize=16)
            ax1.ticklabel_format(axis='y', style='sci', scilimits=(-2, 3))
            ax1.plot(total_data_WE, Rm_current)

            ax2.clear()
            ax2.set_title('$E_{\mathrm{cell}}$ and $i_{\mathrm{w}}R_{\mathrm{m}}$ vs Time', fontsize=16)
            ax2.tick_params(axis='both', which='both', direction='in', right=True, top=True)
            # ax2.tick_params(labelbottom=False)
            ax2.set_xlabel('Time / s', fontsize=16)
            ax2.set_ylabel('Potential / V', fontsize=16)
            ax2.plot(total_time_profile, total_data_WE, label='$E_{\mathrm{cell}}$')
            ax2.plot(total_time_profile, total_data_RM, label='$i_{\mathrm{w}}R_{\mathrm{m}}$')
            ax2.legend()

            # redrew plot with new data
            fig.canvas.draw()

            # callback function must return an integer
            return 5

        '''
        Define buffer size and callback function executed every time buffer is full. 

        Note that the buffer size includes samples from all channels. E.g. if you're collecting 100 samples
        over 5 channels then a buffer size of 100*5 = 500 will be filled once every channel acquires 100 
        samples. 
        '''
        task_i.register_every_n_samples_acquired_into_buffer_event(buffer_size, cont_read)

        # start task to read potential at inputs. This will trigger output to begin potenial sweep
        task_i.start()

        # need an input here for some reason. Press any key to end
        input('Must press Enter to end execution of code block')

        '''put everything in dataframes'''
        data = pd.DataFrame(columns=['E_program', 'Ecell', 'iw', 't'])
        data['E_program'] = pot_profile
        data['Ecell'] = total_data_WE
        data['iw'] = np.array(total_data_RM) / Rm
        data['t'] = np.abs(np.arange(0, len(total_data_WE), 1) / samp_rate)

        # Store parameters
        params = pd.DataFrame({'parameter': ['scan_rate', 'Rm', 'Cm', 'samp_num_total', 'buffer_size', 'samp_rate'],
                               'value': [scan_rate, Rm, Cm, samp_num_tot, buffer_size, samp_rate]})

    # return data
    # return total_data_WE, total_data_RM, np.array(total_data_RM) / Rm, np.abs(
    #     np.arange(0, len(total_data_WE), 1) / samp_rate)
    return data, params

'''CA and helper functions'''
def CA_potential_profile(pot_init: float, pot_step: float, pot_rest: float,
                         pot_init_time: float, pot_step_time: float, pot_rest_time: float,
                         buffer_size: int = 1200, samp_rate: int = 3600) -> tuple:
    """
    :param pot_init: Initial potential in V
    :param pot_step: Step potential in V
    :param pot_rest: Rest potential in V
    :param pot_init_time: Time to hold the initial potential in s
    This will be elongated as needed to round out the total sample number.
    :param pot_step_time: Time to hold the step potential in s
    :param pot_rest_time: Time to hold the resting potential in s
    :param buffer_size: Samples stored in buffer before callback
    :param samp_rate: Sampling rate in samples/s; Use an integral multiple of 120/s and at least 3600 per volt
    :return: pot_profile, samp_num_tot: An array holding potentials for each sample and the total sample number
    """
    # number of samples for each section
    samp_num_init = samp_rate * pot_init_time
    samp_num_step = samp_rate * pot_step_time
    samp_num_rest = samp_rate * pot_rest_time

    # create potential profile array for each section
    pot_profile_init = pot_init * np.repeat(1, samp_num_init)
    pot_profile_step = pot_step * np.repeat(1, samp_num_step)
    pot_profile_rest = pot_rest * np.repeat(1, samp_num_rest)

    '''Since the total sample number must be a multiple of the buffer_size, 
    add additional samples to the initial potential step until it is.'''

    # additional samples in the hold step to round off potential profile
    additional_hold_sample = 0

    # total sample size of the potential profile with extra samples as needed
    samp_num_tot = additional_hold_sample + len(pot_profile_init) + len(pot_profile_step) + len(pot_profile_rest)
    while samp_num_tot % buffer_size != 0:
        additional_hold_sample += 1
        samp_num_tot = additional_hold_sample + len(pot_profile_init) + len(pot_profile_step) + len(pot_profile_rest)

    # Calculate hold profile
    h_profile = np.linspace(pot_init, pot_init, int(additional_hold_sample))

    '''Construct the potential profile by combining each individual section'''
    pot_profile = np.concatenate((h_profile, pot_profile_init, pot_profile_step, pot_profile_rest))
    samp_num_tot = int(len(pot_profile))  # must be an integer

    '''Check potential profile to be set'''
    plt.title('CA Program Potential', fontsize=16)
    plt.xlabel('Time / s', fontsize=16)
    plt.ylabel('$E_{\mathrm{in}}$ / V', fontsize=16)
    plt.tick_params(axis='both', which='both', direction='in', right=True, top=True)
    plt.plot(np.arange(0, len(pot_profile), 1) / samp_rate, pot_profile)

    return pot_profile, samp_num_tot

def take_CA(pot_profile: np.ndarray, samp_num_tot: int, Rm: float,
            Cm: float, buffer_size: int = 3600, samp_rate: int = 3600) -> tuple:
    '''
    :param pot_profile: Array of potentials returned by CV_potential_profile
    :param samp_num_tot: Total number of samples returned by CV_potential_profile
    :param Rm: Measurement resistance in Ω
    :param Cm: Measurement capacitance in F
    :param buffer_size: Samples stored in buffer before callback
    :param samp_rate: Sampling rate in samples/s; Use an integral multiple of 120/s and at least 3600 per volt
    :return: Tuple of DataFrames holding values of data (Ecell, iw, time, f, Y, Z) and parameters.

    This function inputs a potential profile into the ao0 output of the
    potentiostat to set Ein, which then sets Ecell when the counter electrode is connected.

    At the same time potential acquisitions are made on input channels ai0 and ai1 to measure
    Ecell and iwRm.

    Internal collection parameters:
        total_data_WE = cell potential during program potential (ai0)
        total_data_RM = iwRm during program potential (ai1)
        np.array(total_data_RM)/Rm = iwRm during the program potential
        np.abs(np.arange(0, len(total_data_WE), 1)/samp_rate) = time array during the program potential
    '''

    '''Get device name '''
    # get a list of all devices connected
    all_devices = list(nidaqmx.system.System.local().devices)
    # get name of first device
    dev_name = all_devices[0].name
    # print(dev_name)

    ''' add DAQ channels and define measurement parameters'''
    with nidaqmx.Task() as task_i, nidaqmx.Task() as task_o:
        # add ai0 & ai1 input channels for reading potentials. add ao0 output channels for setting potential
        task_i.ai_channels.add_ai_voltage_chan(dev_name + "/ai0:1", min_val=-10.0, max_val=10.0)
        task_o.ao_channels.add_ao_voltage_chan(dev_name + "/ao0", min_val=-10.0, max_val=10.0)

        # define sampling rate and total samples acquired per channel for input & output channels
        task_i.timing.cfg_samp_clk_timing(rate=samp_rate, samps_per_chan=samp_num_tot)
        task_o.timing.cfg_samp_clk_timing(rate=samp_rate, samps_per_chan=samp_num_tot)

        # set up a digital trigger for the output channel to set the potential.  Should be commented out for myDAQs
        # task_o.triggers.start_trigger.cfg_dig_edge_start_trig('/'+ dev_name +'/ai/StartTrigger')

        # create empty lists to populate
        total_data_WE = []  # cell potential in V
        total_data_RM = []  # iwRm potential in V

        # define output channel task. Task will only execute when the output channel trigger is activated
        task_o.write(pot_profile)  # , auto_start = False)
        task_o.start()

        ''' Set up plot during data acquisition'''
        # set up a 2 x 2 grid for the plot
        grid = plt.GridSpec(2, 1, wspace=0.3, hspace=0.2)
        fig = plt.figure(figsize=(14, 12))

        # set right edge of plot to be at 80% of fig width and bottom to be at 20% of fig height to fit everything.
        plt.subplots_adjust(right=0.8)
        plt.subplots_adjust(bottom=0.2)

        # Define positions of 3 subplots
        ax1 = fig.add_subplot(grid[1, 0])
        ax2 = fig.add_subplot(grid[0, 0])
        # ax3 = fig.add_subplot(grid[1, 1])#, sharex = ax2)
        plt.ion()
        fig.show()
        fig.canvas.draw()

        '''Encapsulate buffer callback function'''
        def cont_read(task_handle, every_n_samples_event_type,
                      number_of_samples, callback_data):
            '''
            Define a 'callback' function to execute when the buffer is full

            When this funtion is called a subset of samples are acquired at ai0 and ai1 and then appended
            to the lists 'total_data_WE'and 'total_data_RM'. Then the CV plot is updated with this new data.
             '''

            # Acquire subset of samples and store data in a temporary list
            temp_samples = task_i.read(number_of_samples_per_channel=buffer_size)
            # add acquired data to list storing all data
            total_data_WE.extend(temp_samples[0])
            total_data_RM.extend(temp_samples[1])

            # calculate time profile (for plotting)
            total_time_profile = np.abs(np.arange(0, len(total_data_WE), 1) / samp_rate)
            # calculate current at Rm (for plotting)
            Rm_current = np.array(total_data_RM) / Rm

            # Return size of 'total_data' and update subplots every time buffer is full
            # print(len(total_data_RM))

            ax1.clear()
            ax1.set_title('$i_w$ vs $E_{cell}$', fontsize=16)
            ax1.tick_params(axis='both', which='both', direction='in', right=True, top=True)
            ax1.set_xlabel('$E_{cell}$ / V', fontsize=16)
            ax1.set_ylabel('$i_{w}$ / A', fontsize=16)
            ax1.ticklabel_format(axis='y', style='sci', scilimits=(-2, 3))
            ax1.plot(total_data_WE, Rm_current)

            ax2.clear()
            ax2.set_title('$E_{cell}$ and $i_{w}R_{m}$ vs Time', fontsize=16)
            ax2.tick_params(axis='both', which='both', direction='in', right=True, top=True)
            # ax2.tick_params(labelbottom=False)
            ax2.set_xlabel('Time / s', fontsize=16)
            ax2.set_ylabel('Potential / V', fontsize=16)
            ax2.plot(total_time_profile, total_data_WE, label='$E_{cell}$')
            ax2.plot(total_time_profile, total_data_RM, label='$i_{w}R_{m}$')
            ax2.legend()

            # redrew plot with new data
            fig.canvas.draw()

            # callback function must return an integer
            return 5

        '''
        Define buffer size and callback function executed every time buffer is full. 

        Note that the buffer size includes samples from all channels. E.g. if you're collecting 100 samples
        over 5 channels then a buffer size of 100*5 = 500 will be filled once every channel acquires 100 
        samples. 
        '''
        task_i.register_every_n_samples_acquired_into_buffer_event(buffer_size, cont_read)

        # start task to read potential at inputs. This will trigger output to begin potenial sweep
        task_i.start()

        # need an input here for some reason. Press any key to end
        input('Must press Enter to end execution of code block')

        '''put everything in dataframes'''
        data = pd.DataFrame(columns=['E_program', 'Ecell', 'iw', 't'])
        data['E_program'] = pot_profile
        data['Ecell'] = total_data_WE
        data['iw'] = np.array(total_data_RM) / Rm
        data['t'] = np.abs(np.arange(0, len(total_data_WE), 1) / samp_rate)

        # Store parameters
        params = pd.DataFrame({'parameter': ['Rm', 'Cm', 'samp_num_total', 'buffer_size', 'samp_rate'],
                               'value': [Rm, Cm, samp_num_tot, buffer_size, samp_rate]})

    # return data
    # return total_data_WE, total_data_RM, np.array(total_data_RM) / Rm, np.abs(
    #     np.arange(0, len(total_data_WE), 1) / samp_rate)
    return data, params