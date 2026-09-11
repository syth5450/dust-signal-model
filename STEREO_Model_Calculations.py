# -*- coding: utf-8 -*-
"""
Impact Charging Model for STEREO
Syd Thomas
Updated Sept 10 2026

"""

# Load packages
import STEREO_Model_Functions as sm
import os
import numpy as np
from scipy.interpolate import LinearNDInterpolator
import time
import matplotlib.pyplot as plt
import matplotlib.style as style 
style.use('tableau-colorblind10')

### Loading CST voltage data and calculating G function

# Data txt files listed are assumed to be stored in the same folder under
# current working directory:
    # 'CST_Efield.txt' - electric field data from CST used for ion dynamics
    # 'STEREO_CST_data.txt' - Geo. function data used for interpolation and
    # voltage calculation
    # 'surface_points.txt' - 3D model points with 1 or 0 G value

Vdata = np.loadtxt('STEREO_CST_data.txt', delimiter=',', skiprows=1)
Edata = np.loadtxt('CST_Efield.txt', delimiter=',', skiprows=1)

print('Calculating G')
# CST numerically calculated mutual capacitance matrix, units in Farads
# Formatted as: 
# [C_A1 ... C_SC]
# [...       ...]
# [C_SC ... C_SC]

Q_test = 1*10**-10          # [C] Test charge used in CST simulations

b_CST = np.array([[63.3677, -3.96813, -3.29582, -31.4145],
                  [-3.96813, 60.2400, -4.61724, -19.6257],
                  [-3.29582, -4.61724, 61.2253, -23.6587],
                  [-31.4145, -19.6257, -23.6587, 248.823]])*10**-12

Gdata = sm.calculate_G(b_CST, Vdata, Q_test)

# Load surface points where G=1 and G=0 for all other components
surface = np.loadtxt('surface_points.txt', delimiter=',')
Gdata = np.vstack((Gdata, surface))

# Build interpolator objects for G and E-field data
start_time = time.time()

Eobj = LinearNDInterpolator(Edata[:,0:3], Edata[:,3:])
Gobj = LinearNDInterpolator(Gdata[:,0:3], Gdata[:,3:])

end_time = time.time()
print('Time to build interpolator: ', end_time - start_time, 's')

# Constants and STEREO Info
peak_v_e = 10**6             # [m/s] estimated avg velocity of the electrons
m_e = 9.109*10**-31
# [kg] respective mass for Fe, C, and H ions:
m_species = np.array([9.273*10**-26, 1.994*10**-26, 1.673*10**-27]) - m_e


beta = 1                # power of cosine angular distribution

V_SC = 5                # [V] SC potential as set in CST
C_ant = 63*(10**-12)    # [F], 6m length physical capacitance
C_x = 67*(10**-12)      # [F], mutual capacitance including base + preamp
C_sc = 248.8*(10**-12)  # [F], geometry estimate

gain = 10**(7.6/20)     # 7.6 dB = 2.4 linear gain
lowcut = 10.6           # [Hz]

# Define the base capacitance matrix as written in Shen model
# [C_SC + 3C_x, -C_x, -C_x, -C_x]
# [-C_x, C_A1 + C_x, 0, 0]
# [-C_x, 0, C_A2 + C_x, 0]
# [-C_x, 0, 0, C_A3 + C_x]
# C_x is a mutual capacitance between the components, = C_stray in this case

b_STEREO = np.array([[C_sc + 3*C_x, -C_x, -C_x, -C_x],
                     [-C_x, C_ant + C_x, 0, 0],
                     [-C_x, 0, C_ant + C_x, 0],
                     [-C_x, 0, 0, C_ant + C_x]])

### Read in measured waveform data
# Measured signals have been pre-processed to set initial
# voltage to zero at impact (t=0)

signal_info = [{'data': 'SWAVES_Waveforms/03_Jan_00_47_16_436_A.txt', 
                'name': '2007 Jan 03 00:47:16.436'},
               {'data': 'SWAVES_Waveforms/01_Apr_2009_21_04_27_928_A.txt', 
                'name': '2009 Apr 01 21:04:27.928'},
               {'data': 'SWAVES_Waveforms/30_Oct_2013_00_24_52_136_A.txt', 
                'name': '2013 Oct 30 00:24:52.136'},
               {'data': 'SWAVES_Waveforms/01_Jan_2021_15_19_24_293_A.txt', 
                'name': '2021 Jan 01 15:19:24.293'},
               {'data': 'SWAVES_Waveforms/22_Jan_2021_00_14_05_269_A.txt', 
                'name': '2021 Jan 22 00:14:05.269'}]

# Define measurement from dictionary and load respective data file
signal = np.loadtxt(signal_info[2]['data'], delimiter=',')
signal_name = signal_info[2]['name']

# Set a range of time from measured signal sample rate
end_t   = 2000*10**(-6)                                 # [s]
sample_rate = np.round(signal[1,3] - signal[0,3], 6)    # Will be 4 or 8 μs
time_range = np.arange(0, end_t, sample_rate)

### Establish impact
# Simulating the ion plume is the longest running part and typically takes
# longer for a greater slow ion population
# Antihelion impact resolution improved for n > 500

# Basic Surface Constraints and Key Location Coordinates:
    # +z (Helion): 1.032 at center feature or 0.887 elsewhere
    # -z (Antihelion): -0.421 at center feature or -0.301 elsewhere
    # x (Ram/Anti-ram): +/- 0.63
    # y (Top/Bottom): -0.88 to 0.85. Box features at (y= 0.968 and -.960)
    # Between antenna: [0.352, 0.755, -0.355]
    # Panels 0.957 on -z or 0.974 on +z, out to 3.9 m in x
    # Main body region:
        # x: [-0.63, 0.63]
        # y: [-0.88, 0.85]
        # z: [-0.30, 0.88]
    
# If impact location has been selected inside one of the smaller SC features,
# this will be visible as spikes/noise at the immediate start of the signal

n = 500
m = 500

# Replace belows arrays with impact coordinates and orthogonal vector
imp = np.array([0, 0, 0])   
rhat = np.array([1, 0, 0])

# [slow species, fast species] ratios, medium speed = 1 - sum(vel_ratio)
vel_ratio = [0.2, 0.25]          

start = time.time()
print('Expanding Impact Ionization Plume')

v_i, v_e, m_i, T_e = sm.plume_vel_dist(n, m, rhat, beta, vel_ratio, peak_v_e,
                                       m_species, V_SC)
pos_i, ij_i = sm.ion_pos(n, v_i, imp, time_range, m_i, Eobj)
pos_e, ij_e = sm.electron_pos(m, v_e, imp, time_range)

end = time.time()
print( 'Run time: ', (end-start)/60, 'min')

print('Interpolating G function')
XYZG_i = sm.particle_G(pos_i, ij_i, Gobj)
XYZG_e = sm.particle_G(pos_e, ij_e, Gobj)
end_time = time.time()
print('Run Time for Test: ', (end_time - start_time)/60, 'min')

# %% ## Voltage Calculation
kappa = 0.5           # Expected value sufficient for most fits
R_base = 50*10**6     # [Ω]
Q_imp = 15*10**-12    # [C]
a1 = .5
a2 = .75              # 0.1-1
a3 = .5
a_sc = 0.6

# Calculate voltages of each component
V = sm.calculate_V(XYZG_e, XYZG_i, b_STEREO, Q_imp, kappa, a1, a2, a3, a_sc, 
  R_base, time_range, sample_rate, m, n, V_SC, T_e, ij_e, ij_i)

# Monopole (SC-ANT)
V_ant = np.array([V[1,:]-V[0,:], V[2,:]-V[0,:], V[3,:]-V[0,:]])

# Filter signal TDS highband: 4μs = 108kHz 8μs = 54kHz
if sample_rate == 8*10**-6:
    V_filt = sm.filter_V(V_ant, time_range, gain, lowcut, 54000)
elif sample_rate == 4*10**-6:
    V_filt = sm.filter_V(V_ant, time_range, gain, lowcut, 108000)

# For figure info display
def format_ratio(ratio):
    formatted_ratio = np.array([ratio[0], 1-np.sum(ratio), ratio[1]])
    formatted_ratio = '[' + ', '.join(map(str,list(formatted_ratio))) +']'
    return formatted_ratio

### Plot result against measured signal
endt = np.where(signal[:,3] > time_range[-1])[0][0] # Fix array length
signalfig = plt.figure(figsize=(9,7))
A1fig = signalfig.add_subplot(311)
plt.plot(time_range[:endt]*(10**3), signal[:endt,0], label = 'Measured ANTz')
plt.plot(time_range*(10**3), V_filt[0]*1000, label='Modeled ANTz')
plt.grid()
plt.legend(fontsize=12)
plt.title('Model Fit to S/WAVES Signal ' + signal_name, fontsize=14)
    
A2fig = signalfig.add_subplot(312)
plt.plot(time_range[:endt]*(10**3), signal[:endt,1], label = 'Measured ANTy')
plt.plot(time_range*(10**3), V_filt[1]*1000, label='Modeled ANTy')
plt.grid()
plt.legend(fontsize=12)
plt.ylabel('Voltage (mV)', fontsize=14)
    
A3fig = signalfig.add_subplot(313)
plt.plot(time_range[:endt]*(10**3), signal[:endt,2], label = 'Measured ANTx')
plt.plot(time_range*(10**3), V_filt[2]*1000, label='Modeled ANTx')
plt.grid()
plt.legend(fontsize=12)
plt.xlabel('Time (ms)', fontsize=14)

plt.gcf().text(0.92, 0.86, 'Location: ' + '(' + ', '.join(map(str,list(imp)))
               + ') ' + 'm', fontsize=12)
plt.gcf().text(0.92, 0.82, 'Q_imp: ' + str(np.round(Q_imp*10**12, 2)) + 
               ' pC,', fontsize=12)
plt.gcf().text(1.08, 0.82, 'Kappa: ' + str(np.round(kappa,2)), fontsize=12)
plt.gcf().text(0.92, 0.78, 'R_base: ' + str(R_base/10**6) + ' MΩ,' + 
               '  a_SC: ' + str(np.round(a_sc, 2)), fontsize=12)
plt.gcf().text(0.92, 0.74, 'a_z: ' + str(a1), fontsize=12)
plt.gcf().text(1.03, 0.74, 'a_y: ' + str(a2), fontsize=12)
plt.gcf().text(1.12, 0.74, 'a_x: ' + str(a3), fontsize=12)
plt.gcf().text(0.92, 0.7, 'Velocity Ratio: ' + format_ratio(vel_ratio), 
               fontsize=12)
