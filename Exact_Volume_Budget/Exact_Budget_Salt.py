# -*- coding: utf-8 -*-
"""
Created on Mon Oct  5 15:49:58 2020

@author: Jasen
"""
import os
os.chdir('/home/jjacob2/python/Salt_Budget/SalinityVarianceBudget/Exact_Volume_Budget/')

from netCDF4 import Dataset as nc4
import numpy as np
import Exact_Budget_Salt_Terms as ebt
import obs_depth_JJ as dep
import matplotlib.pyplot as plt


#load files (hourly output)
FilePath = '/home/cae/runs/jasen/wc15.a01.b03.hourlywindWT.windmcurrent.diags/out.jasen3/'
HistFile = FilePath + 'ocean_his_2014.nc'
Hist = nc4(HistFile, 'r')
AvgFile = FilePath + 'ocean_avg_2014.nc'
Avg = nc4(AvgFile, 'r')
Diag = nc4(FilePath + 'ocean_dia_2014.nc', 'r')

#bounds of control volume (points in counter clockwise order)
#north west corner of domain for mask testing
#Vertices = np.array([[35.7, -124.215], [35.7, -124.0], [35.8, -124.0], [35.8, -124.215]])

#shelf & slope offshore of Pt. Ano Nuevo
#Vertices = np.array([[37.0, -123.0], [37.0, -122.5], [37.2, -122.5], [37.2, -123.0]])

#Carmel Bay
#Vertices = np.array([[36.522, -121.948], [36.522, -121.926], [36.567, -121.926], [36.567, -121.948]])

#Monterey Bay
Vertices = np.array([[36.585, -121.976], [36.585, -121.784], [36.981, -121.784], [36.981, -121.976]])

#create masks
Masks = ebt.CreateMasks(Avg, Vertices)

#load time for steping
time = Avg.variables['ocean_time'][:]

#load dx and dy for gradients
dx = np.repeat(1/np.array(Avg.variables['pm'][:])[np.newaxis, :, :], \
               Avg.variables['salt'][0,:,0,0].size, axis = 0)
dy = np.repeat(1/np.array(Avg.variables['pn'][:])[np.newaxis, :, :], \
               Avg.variables['salt'][0,:,0,0].size, axis = 0)

#Prealocation 
dsdt_dV = np.empty(time.shape)
dsdt_dV.fill(np.nan)
AdvFlux = np.empty(time.shape)
AdvFlux.fill(np.nan)
DifFlux = np.empty(time.shape)
DifFlux.fill(np.nan)
IntMix = np.empty(time.shape)
IntMix.fill(np.nan)
HuonFlux = np.empty(time.shape)
HuonFlux.fill(np.nan)

##Loop on time step
for tstep in range(time.shape[0]) :
    #compute variance at time step
    var = Avg.variables['salt'][tstep, :, :, :]
    
    #compute depth at averg points
    dz = np.diff(dep._set_depth(AvgFile, None, 'w',\
                                Avg.variables['h'][:],\
                                Avg.variables['zeta'][tstep, :, :]), \
                 n = 1, axis = 0)
    
    #compute cell areas
    Areas = ebt.CellAreas(dx, dy, dz, Masks)
        
    #time derivative of salinity variance squared
    dsdt_dV[tstep] = ebt.TimeDeriv(tstep, var, Hist, HistFile, Avg, AvgFile, Diag, Areas['Axy'], Masks)
    
    #Advective flux
    AdvFlux[tstep] = ebt.Adv_Flux_west(tstep, var, Avg, Areas, Masks)
    
    HuonFlux[tstep] = np.sum(Avg.variables['Huon_salt'][tstep,:,:,:][Masks['WFace']])
    
    #Diffusive Flux
    DifFlux[tstep] = ebt.Diff_Flux_west(Avg, var, dx, Areas, Masks)
    
    #Internal Mixing, FIX THIS
    IntMix[tstep] = ebt.Int_Mixing(tstep, var, Avg, dx, dy, dz, Masks)

#plotting
Total = dsdt_dV + AdvFlux - DifFlux + IntMix

plt.figure()
line0, = plt.plot(dsdt_dV, label = 'd/dt')
line1, = plt.plot(AdvFlux, label = 'Advective Flux')
line2, = plt.plot(DifFlux, label = 'Diffusive Flux')
line3, = plt.plot(IntMix, label = 'Internal Mixing')
line4, = plt.plot(Total, label = 'Sum of Terms')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
plt.title('Monterey Bay Salt Budget Time Step (<u>*<s>)')

Total1 = dsdt - HuonFlux - DifFlux + IntMix
plt.figure()
line0, = plt.plot(dsdt, label = 'd/dt')
line1, = plt.plot(-1*HuonFlux, label = 'Huon_salt')
line2, = plt.plot(DifFlux, label = 'Diffusive Flux')
line3, = plt.plot(IntMix, label = 'Internal Mixing')
line4, = plt.plot(Total1, label = 'Sum of Terms')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
plt.title('Monterey Bay Salt Budget Time Step (<u*s>)')

chk = np.abs(AdvFlux/HuonFlux)
plt.figure()
plt.plot(chk)
plt.grid()
plt.title('abs ( Adv/Huon ) ')
plt.ylim(0.999, 1.001)
