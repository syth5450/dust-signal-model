# -*- coding: utf-8 -*-
"""
Functions used for STEREO Impact Modeling
Syd Thomas
Updated 9/11/2026

"""

import numpy as np
import math
from math import pi
from scipy.linalg import null_space
import scipy.signal
import os

m_p = 1.67*10**-27     # [kg] mass of a proton
m_e = 9.109*10**-31    # [kg] mass of electron
q = 1.602*10**-19      # [C] fundamental charge

### For convenience
def sind(x):
    sin_degrees = np.sin(x*(np.pi/180))
    return sin_degrees

def cosd(x):
    cos_degrees = np.cos(x*(np.pi/180))
    return cos_degrees

### Calculate the geometric function of all data points in the set
def calculate_G(cap, data, Q_test):
    # cap: the capacitance matrix calculated by CST
    # data: the voltage data from CST
    # Q_test: value of the test charge used in CST sims
    
    # will be 1/n the length of data where n is the number of objects
    GData = np.empty((int(len(data)/4), 7)) # [x, y, z, G_ANT1, ..., G_SC]
    
    for i in range(1, int(len(data)/4)+1):
        Gmat = (1/Q_test)*np.matmul(cap, data[4*i-4:4*i, 4])

        coord = data[4*i-1, 1:4]
        GData[i-1] = np.hstack([coord, Gmat])
        
    # Check for negative values due to sim or dataset error
    if len(np.where(GData[:, 3:] < 0)[0]) > 0:
        print('WARNING: Negative G-value(s) found')
    return GData

### Generate the distribution of angles and velocities for the expanding plume
# Returns an array for both the ions and electrons, as well as the randomised
# ion mass and electron temperature for each particle
def plume_vel_dist(num_i, num_e, rhat_imp, beta, v_popRatio, const_vel_e,
                   m_species, V_SC):
    # num_i/num_e: cation / electron population size
    # rhat_imp: unit vector pointing in the direction of expansion post impact
    # orthogonal with the SC surface
    # beta: cone angle, power of cosine angular distribution, >= 1
    # const_vel_e: peak velocity for the electrons in boltzmann distribution
    # q: charge of particles
    # m_e: mass of electrons
    # m: mass of particle/particle species
    # V_SC: the SC potential in Volts
    
    # cosine angular ion distribution for arbitrary location
    # x and y in this case are not the coordinates of x and y used in CST
    xrange = np.arange(0, 90.1, 0.1)
    norm_max = np.max(2*pi*sind(xrange)*(cosd(xrange)**beta))
    i_angles = np.zeros(num_i)
    for i in range(0, len(i_angles)):
        # generate a random angle in the first dimension
        # 0-1 multiplier to get height in 2nd dimension
        x_cos = 90*np.random.rand()         
        y_cos = norm_max*np.random.rand()
        
        if y_cos <= (2*pi*sind(x_cos)*cosd(x_cos)**beta):
            i_angles[i] = x_cos
  
    # velocity distribution ions
    v_pops = np.array([3800, 14000, 36700])             # [m/s] Fe, C, H
    # Round up to nearest integers
    pop_num = np.empty(3, dtype=int)                    
    pop_num[0] = int(math.ceil(num_i*v_popRatio[0]))
    pop_num[1] = int(math.ceil(num_i*(1-sum(v_popRatio))))
    pop_num[2] = int(math.ceil(num_i*v_popRatio[1]))
    
    # column vectors for velocity and mass assigned to each particle
    n_vels = np.vstack((v_pops[0]*np.ones((pop_num[0], 1)), 
                        v_pops[1]*np.ones((pop_num[1], 1)),
                        v_pops[2]*np.ones((pop_num[2], 1))))
    m_i = np.vstack((m_species[0]*np.ones((pop_num[0], 1)),
                     m_species[1]*np.ones((pop_num[1], 1)),
                     m_species[2]*np.ones((pop_num[2], 1))))
    
    # isotropic distribution for electrons
    e_angles = 90*np.random.rand(num_e, 1)
    
    # velocity and direction for both species
    two_ortho = null_space([rhat_imp])
    rhat_imp = np.array([[rhat_imp[0]], [rhat_imp[1]], [rhat_imp[2]]])
    new_ortho = np.hstack((two_ortho, rhat_imp))
    rot_i = 360*np.random.rand(num_i, 1)            # [degrees]
    rot_e = 360*np.random.rand(num_e, 1)
    
    # ions
    v_dir_i = np.empty((3, num_i))
    v_i = np.empty((3, num_i))
    for i in range(0, num_i):
        v_dir_i[:,i] = (new_ortho[:,0]*cosd(rot_i[i])*sind(i_angles[i]) 
                        + new_ortho[:,1]*sind(rot_i[i])*sind(i_angles[i]) 
                        + new_ortho[:,2]*cosd(i_angles[i]))
        v_i[0:3,i] = n_vels[i]*v_dir_i[:,i]
        
    # electrons
    v_dir_e = np.empty((3, num_e))
    v_e = np.empty((3, num_e))
    T_e = np.random.normal(2.2, 0.4, num_e) # [eV]
    # Boltzmann distribution for electrons speeds affected by SC potential
    vel_e = const_vel_e*math.e**(-V_SC/T_e)
    # randomly assign speeds to each electron
    np.random.shuffle(vel_e)                

    for i in range(0, num_e):
        v_dir_e[:,i] = (new_ortho[:,0]*cosd(rot_e[i])*sind(e_angles[i]) 
                        + new_ortho[:,1]*(sind(rot_e[i])*sind(e_angles[i]))
                        + new_ortho[:,2]*cosd(e_angles[i]))
        v_e[0:3, i] = vel_e[i]*v_dir_e[:,i]
        
    return v_i, v_e, m_i, T_e

### Calculate electron position at each time step. Lorentz force not included
# Also returns an array of indexes for particles that re-collide with the SC
def electron_pos(num_e, v, r_imp, time):
    # num_e: electron population size
    # v: velocity vector for electrons
    # r_imp: vector pointing in direction of expansion post impact orthogonal
    # to SC surface
    # time: time range
    
    pos_electrons = np.zeros((len(time), 3, num_e))
    # list to store indexes of impact with SC
    ij = []                                         
    for i in range(0, num_e):
        p = np.zeros((len(time), 3))                
        for j in range(0, len(time)):
            x = r_imp[0] + v[0,i]*time[j]
            y = r_imp[1] + v[1,i]*time[j]
            z = r_imp[2] + v[2,i]*time[j]
            p[j,:] = np.array([x, y, z])
            
            # if p within SC, set remainder of this particle's array to
            # previous position
            # approximate dimensions of main body, boom, and panels
            if ((-0.631 < p[j,0] < 0.631) and (-0.88 < p[j,1] < 0.85)
                and (-0.301 < p[j,2] < 0.886)
            or ((-0.725 < p[j,0] < -0.497) and (-0.979 < p[j,1] < -0.865)
                and p[j,2] < 0.911)
            or ((-3.93 < p[j,0] < -0.672) and (0.016 < p[j,1] < 0.698)
                and (0.957 < p[j,2] < 0.974))
            or ((0.671 < p[j,0] < 3.93) and (-0.41 < p[j,1] < 0.268)
                and (0.957 < p[j,2] < 0.974))): 
                    p[j:,:] = p[j-1,:]
                    ij.append([i, j]) # save particle index and time index
                    break
        pos_electrons[:,:,i] = p
        
    np.delete(pos_electrons, 0, axis=0)
    return pos_electrons, np.asarray(ij)

### Same position calculation for ions, Lorentz force included
def ion_pos(num_i, v0, r_imp, time, m, Eobj):
    # num_i: cation population size
    # v0: initial velocity vector for each particle from plume_vel_dist
    # r_imp: vector pointing in direction of expansion post impact
    # time: time range
    # q, m: ion charge and mass
    # Eobj: Interpolator object built from E field data in main script
    
    pos_particles = np.zeros((len(time), 3, num_i))
    # list to store indexes of impact with SC
    ij = []                                            
    for i in range(0, num_i):
        p = np.zeros((len(time), 3))
        p[0,:] = r_imp
        v = np.vstack((v0[:,i].reshape(-1), np.zeros((len(time), 3))))
        dv = np.zeros((len(time),3))
        for j in range(0, len(time)-1):
            # calculate change in v by interpolating e-field from SC for
            # current position using charge and mass of respective particle
            # hard adjusted spacing to speed up simulation, multiple of nt:
            if j < 50: 
                dv[j,:] = (q*Eobj(p[j,:]))/m[i]
                v[j+1,:] = v[j,:] + dv[j,:]*time[j+1]    
                      
                x = r_imp[0] + v[j,0]*time[j+1] + 0.5*dv[j,0]*(time[j+1]**2)
                y = r_imp[1] + v[j,1]*time[j+1] + 0.5*dv[j,1]*(time[j+1]**2)
                z = r_imp[2] + v[j,2]*time[j+1] + 0.5*dv[j,2]*(time[j+1]**2)
                p[j+1,:] = np.array([x, y, z])
                
            nt = 25
            # Interpolate E every n time steps. Assumes time divisble by nt
            if (50 <= j < len(time) - nt) and j%nt == 0:
                
                dv[j:j+nt] = (q*Eobj(p[j-1,:]))/m[i]
                v[j+1:j+nt+1,:] = v[j,:] + dv[j,:]*time[j+1]
                
                # calculate position for next n time steps using same E value
                for k in range(0, nt):
                    x = r_imp[0] + v[j+k,0]*time[j+k+1] +0.5*dv[j+k,0]*(time[j+k+1]**2)
                    y = r_imp[1] + v[j+k,1]*time[j+k+1] + 0.5*dv[j+k,1]*(time[j+k+1]**2)
                    z = r_imp[2] + v[j+k,2]*time[j+k+1] + 0.5*dv[j+k,2]*(time[j+k+1]**2)
                    p[j+1+k,:] = np.array([x, y, z])
                    
            # if particle exceeds Edata range, set velocity from previous
            # calculation to avoid nan's
            if np.isnan(dv[j,:]).any() == True:
                v[j+1,:] = v[j,:]
                p[j+1,:] = r_imp + v[j,:]*time[j]
            
            # if a within SC, set remainder of this particle's array to
            # previous position
            # approximate dimensions of main body, boom, and panels
            if ((-0.631 < p[j,0] < 0.631) and (-0.88 < p[j,1] < 0.85)
                and (-0.301 < p[j,2] < 0.886)
            or ((-0.725 < p[j,0] < -0.497) and (-0.979 < p[j,1] < -0.865)
                and p[j,2] < 0.911)
            or ((-3.93 < p[j,0] < -0.672) and (0.016 < p[j,1] < 0.698)
                and (0.957 < p[j,2] < 0.974))
            or ((0.671 < p[j,0] < 3.93) and (-0.41 < p[j,1] < 0.268)
                and (0.957 < p[j,2] < 0.974))): 
                    p[j:,:] = p[j-1,:]
                    ij.append([i, j])    # save particle index and time index
                    break
    
        pos_particles[:,:,i] = p
    return pos_particles, np.asarray(ij)


### Calculate the geometric function for every particle position
def particle_G(partPos, ij, objG, returnNaN = False):    
    # partPos: particle positions calculated by previous two functions
    # ij: collision indexes
    # objG: interpolator object built in main script
    # returnNaN: used for debugging in case of particles outside of sim region
    
    # add columns to the positions array for G values, order xyz, G_A1 ...G_SC
    partXYZG = np.concatenate((partPos.copy(), np.empty((len(partPos[:,0,0]),
             4,len(partPos[0,0,:])))), 1) 
    
    for i in range(0,len(partXYZG[0,0,:])):
        partXYZG[:,3:,i] = objG(partPos[:,:,i])
        
        # Find any nans in case of high speed particle escaping range of data points
        nanList = []
        if np.isnan(partXYZG[:,3:7,i]).any() == True:
            nanIdx = np.asarray(np.where(np.isnan(partXYZG[:,6,i])))
            nanList.append(nanIdx[:,0])
            # assume antenna G already near zero
            partXYZG[:,3:7,i] = np.nan_to_num(partXYZG[:,3:7,i], nan=0.0001)
            partXYZG[:,6,i] = np.nan_to_num(partXYZG[:,6,i], nan=0.001)
        
        # Set G values below an assumed computational error to zero
        err = np.where(partXYZG[:,3,i] < 0.0001)
        partXYZG[err[0][0::4], 3, i] = 0
        partXYZG[err[0][1::4], 4, i] = 0
        partXYZG[err[0][2::4], 5, i] = 0
        partXYZG[err[0][3::4], 6, i] = 0

    # Set G=0 after collision with spacecraft body occurs
    if len(ij>0):
        for t in range(0, len(ij[:,1])):      
            partXYZG[ij[t,1]:, 3:7, ij[t,0]] = 0  # correct assignment
            partXYZG[ij[t,1], 6, ij[t,0]] = 1     # override G_SC = 1 at hit

    if returnNaN == True:
        return partXYZG, nanList  
    else:       
        return partXYZG
    
### objectwise voltage calculation of the transient impact signal
def calculate_V(XYZGe, XYZGi, b, Q_imp, kappa, a1, a2, a3, asc, R_base,
                time, tstep, m, n, V_SC, T_e, ij_e, ij_i):
    # b: capacitance matrix built from actual SC components
    # Q_imp: Total charge of impacting dust particle
    # aSC/a1/a2/a3: discharge efficiency parameter
    # R_base: resistance of the SC base electronics
    # V_SC: floating potential of the SC
    
    # Electron and ion separation
    # Q_coll is the collected charge, Q_esc the escaping charge and a boltzmann distribution
    T_eavg = np.sum(T_e)/len(T_e)
    Qe_esc = -Q_imp*kappa*math.e**(-V_SC/T_eavg)
    Qe_coll = -Q_imp - Qe_esc
    Qi_esc = Q_imp
    Qi_coll = Q_imp - Qi_esc
    Q_coll = Qe_coll + Qi_coll
    
    dQ_ind = np.zeros((4, len(time)))   # Plume-induced charging
    dQ_base = np.zeros((4, len(time)))  # initial conditions assumed zero
    dQ_pl = np.zeros((4,len(time)))     # discharge through plasma environment
    V = np.zeros((4, len(time)))
    
    for i in range(1,len(time)):            
        # Correct size of n and m for each time step in case of plume collision with SC
        # i > 1 condition so that first time step includes all electrons even though some are absorbed immediately
        if len(ij_i) > 0 and i > 1: 
            nn = n - len(ij_i[np.where(ij_i[:,1] < i)])
        else:
            nn = n
        if len(ij_e) > 0 and i > 1:    
            mm = m - len(ij_e[np.where(ij_e[:,1] < i)]) 
        else:
            mm = m
        
        # Discharge through plasma environment    
        phi_SC = 5.5            # Expected equilibrium voltages [V]
        phi_ANT = 6.4
        C_SC = 248*10**-12      # SC capacitance (numerical calc.) [F]
        C_ANT = 61*10**-12      # ANT capacitance (measured) [F]
        S_SC = 11.46            # [m^2] Surface area of SC (from CST model)
        S_ANT = 0.45            # [m^2] Surface area of all three ANT 
        T_ph = 2                # photoelecton temperature [eV]
        T_e = 8                 # solar wind electron temperature [eV]             
        
        e0 = 1.602*10**-19    # [C]
        m_e = 9.109*10**-31   # [kg]
        n_e = 5000000         # [m^-3]
        w_e = ((e0*T_e)/(2*np.pi*m_e))**0.5
        
        # Discharge of SC into ANTs through base resistor
        dQ_base[0,i] = np.sum((3*V[0,0:i-1]-V[1,0:i-1]-V[2,0:i-1]-V[3,0:i-1])/R_base*tstep)
        dQ_base[1:,i] = np.sum((V[1:,0:i-1]-V[0,0:i-1])/R_base*tstep)
        
        # Recalculate time constants with more accurate surface areas
        tau_SC = (C_SC*T_ph)/(e0*n_e*w_e*S_SC)*T_e/(T_e + phi_SC +T_ph)     # 58 μs
        tau_ANT = (C_ANT*T_ph)/(e0*n_e*w_e*S_ANT)*(1+(phi_ANT/T_e))**-0.5   # 533 μs
        
        R_disSC = tau_SC/C_SC       # ~ 230 kOhm
        R_disANT = tau_ANT/C_ANT    # ~ 8.7 MOhm
    
        # Discharge through ambient plasma
        dQ_pl[0,i] = asc*np.sum(V[0,0:i-1]/R_disSC*tstep)
        dQ_pl[1,i] = a1*np.sum((V[1,0:i-1])/(R_disANT)*tstep)
        dQ_pl[2,i] = a2*np.sum((V[2,0:i-1])/(R_disANT)*tstep)
        dQ_pl[3,i] = a3*np.sum((V[3,0:i-1])/(R_disANT)*tstep)
        
        dQ_ind[0,i] = Qe_esc*np.sum(XYZGe[i,6,:])/mm + Qi_esc*np.sum(XYZGi[i,6,:])/nn
        dQ_ind[1,i] = Qe_esc*np.sum(XYZGe[i,3,:])/mm + Qi_esc*np.sum(XYZGi[i,3,:])/nn
        dQ_ind[2,i] = Qe_esc*np.sum(XYZGe[i,4,:])/mm + Qi_esc*np.sum(XYZGi[i,4,:])/nn
        dQ_ind[3,i] = Qe_esc*np.sum(XYZGe[i,5,:])/mm + Qi_esc*np.sum(XYZGi[i,5,:])/nn
        
        # dQ matrix normalized over all particles
        # initial collected charge on antennas assumed zero (impact on SC)  
        dQ = np.array([Q_coll + dQ_ind[0,i] - dQ_base[0,i] - dQ_pl[0,i], 
                         dQ_ind[1,i] - dQ_base[1,i] - dQ_pl[1,i],
                         dQ_ind[2,i] - dQ_base[2,i] - dQ_pl[2,i], 
                         dQ_ind[3,i] - dQ_base[3,i] - dQ_pl[3,i]]) 
        
        # Voltage perturbation per time step
        dV = np.matmul(np.linalg.inv(b), dQ)
        V[:,i] = dV
        
    return V

### Filter calculated voltage signals
def filter_V(V, time_range, gain, lowcut, highcut):
    # V: array of the mutual voltages between antennas and SC
    # low/highcut: the low and high frequencies of the bandwidth
    fs = 1/(time_range[1]-time_range[0]) # sampling rate
    
    fc = np.array([lowcut, highcut])
    wc = 2*fc/fs
    
    b1, a1 = scipy.signal.butter(1, lowcut, btype='highpass', fs=fs)
    b2, a2 = scipy.signal.butter(1, highcut, btype='lowpass', fs=fs)
    
    V_filtered = np.empty(len(time_range))
    
    V_filtered = scipy.signal.lfilter(b1, a1, V)          # Apply highpass
    V_filtered = scipy.signal.lfilter(b2, a2, V_filtered) # Apply lowpass
    V_filtered = gain*V_filtered
    return V_filtered

### Data reading from raw CST data which is formatted uniquely
# This version set up for STEREO which has SC and 3 antenna data keys
# input 'mainDirectory' is address of a data folder within working directory
def read_and_sort(mainDirectory):
    directory = []
    for foldername in os.listdir(mainDirectory):
        # d is a string of the folder name
        d = os.path.join(mainDirectory, foldername)
        # skips dataless folders from mac transfer
        if d.find('DS') > -1:
            continue
        else:
            directory.append(d)
    data = np.empty((1,5))
    
    # access data
    for i in range(0,len(directory)):
         # save z value from folder name
         z = os.path.basename(directory[i])

         # for each file in each z value
         for filename in os.listdir(directory[i]):
             fpath = os.path.join(directory[i], filename) # path to each file
             
             # Skipping the files that can't be decoded/no data read
             if filename.find('._') > -1:
                 print('WARNING: file skipped: ', filename)
                 continue
             
             # if statements for string containing object
             # capacitance matrixes for assigning these in the correct order
             if filename.find('ANT1') > -1 or filename.find('A1') > -1:
                 objName = 0
             elif filename.find('ANT2') > -1 or filename.find('A2') > -1:
                 objName = 1
             elif filename.find('ANT3') > -1 or filename.find('A3') > -1:
                 objName = 2
             #elif filename.find('ANT4') > -1:
                 #objName = 4
             elif filename.find('SC') > -1:
                 objName = 3
             else:
                 print('Error: object not found')
    
             # read file into temporary array
             with open(fpath, 'r', encoding = 'utf-8', errors='ignore') as f:
                lines = f.readlines()
             fileData = [line.strip() for line in lines]
             
             # set empty arrays for storing variables
             yVals = []
             x = []
             V = []
             y = []
             j = 0
             for j in range(0,len(fileData)):
                 #print(fileData[j])
                 
                 # find y values in the file data
                 if fileData[j].find('y=') != -1: # if string contains y=
                    # use string manipulation to isolate y
                    start = fileData[j].find('y=') + 2
                    end = fileData[j].rfind(')')
                    #print('start/end: ', start, end)
                    yVals.append(float(fileData[j][start:end]))
                    
                 elif fileData[j].find('ycp=') != -1:
                     start = fileData[j].find('ycp=') + 4
                     end = fileData[j].rfind(')')
                     yVals.append(float(fileData[j][start:end]))
                     
                 elif fileData[j].find('y_cp=') != -1:
                     start = fileData[j].find('y_cp=') + 5
                     end = fileData[j].rfind(')')
                     yVals.append(float(fileData[j][start:end]))
                     
                 elif fileData[j].find('y_prime') != -1:
                     start = fileData[j].find('y_prime=') + 8
                     end = fileData[j].rfind(')')
                     yVals.append(float(fileData[j][start:end]))
                     
                 elif fileData[j].find('yprime') != -1:
                     start = fileData[j].find('yprime=') + 7
                     end = fileData[j].rfind(')')
                     yVals.append(float(fileData[j][start:end]))
                         
                 # pull all x and V values, fill in y values
                 elif fileData[j].find('\t') != -1:
                    start = fileData[j].find('\t')
                    end = fileData[j].find('\t') + 1
                    x.append(float(fileData[j][0:start]))
                    V.append(float(fileData[j][end:len(fileData[j])]))
                    
                    # append most recent y value to data  
                    y.append(float(yVals[len(yVals)-1]))               

             fileData2 = np.zeros((len(x),5))
             j = 0
             for j in range(0,len(x)):
                 fileData2[j][0] = objName
                 fileData2[j][1] = x[j]
                 fileData2[j][2] = y[j]
                 fileData2[j][3] = z
                 fileData2[j][4] = V[j]
                 
             # add on to the larger data set
             if len(fileData2) > 0:
                 data = np.vstack((data, fileData2))
             else:
                 print ('WARNING: empty file data')             
    
    # clear empty row
    data = np.delete(data, 0, axis=0)
    
    # keep unique data points
    uniques = np.unique(data[:,:], axis=0)
    #uniques = data[idx[1], :]
    
    # Sort data by positions and object for G calculation
    sort_indexes = np.lexsort((uniques[:,0], uniques[:,3], uniques[:,2], uniques[:,1]))
    dataSorted = uniques[sort_indexes]
            
    return dataSorted