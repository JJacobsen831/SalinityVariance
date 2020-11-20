# -*- coding: utf-8 -*-
"""
Created on Mon Oct  5 15:49:58 2020

@author: Jasen
"""
import os
os.chdir('/home/jjacob2/python/Salt_Budget/SalinityVarianceBudget/Exact_Volume_Budget/')

from netCDF4 import Dataset as nc4
import numpy as np
import Exact_Budget_Terms as ebt
import obs_depth_JJ as dep
import matplotlib.pyplot as plt

#load files
#time step
FilePath = '/home/cae/runs/jasen/wc15.a01.b03.hourlywindWT.windmcurrent.diags/out.jasen3/'
HistFile = FilePath + 'ocean_his_2014.nc'
Hist = nc4(HistFile, 'r')
AvgFile = FilePath + 'ocean_avg_2014.nc'
Avg = nc4(AvgFile, 'r')
Diag = nc4(FilePath + 'ocean_dia_2014.nc', 'r')

#bounds of control volume (points in counterclockwise order)
#Shelf and slope
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

##Loop on time step
for tstep in range(time.shape[0]-1) :
    #compute variance at time step
    var = Avg.variables['salt'][tstep, :, :, :]
    
    #increase percision for multiplication
    vprime = np.array((var - var[Masks['RhoMask']].mean()), dtype = np.float64)*Masks['LandMask']
    
    #then return to float 32 -> DON'T APPLY MASK, do after calc
    vprime2 = np.array(vprime*vprime, dtype = np.float64)
        
    #compute depth at averg points
    dz = np.diff(dep._set_depth(AvgFile, None, 'w',\
                                Avg.variables['h'][:],\
                                Avg.variables['zeta'][tstep, :, :]), \
                 n = 1, axis = 0)
    
    #compute cell areas
    Areas = ebt.CellAreas(dx, dy, dz, Masks)
        
    #time derivative of salinity variance squared
    dsdt_dV[tstep] = ebt.TimeDeriv(tstep, vprime, Hist, HistFile, Avg, AvgFile, Diag, Areas['Axy'], Masks)
    
    #Advective flux
    AdvFlux[tstep] = ebt.Adv_Flux_west(tstep, vprime2, Avg, Areas, Masks)
    
    #Diffusive Flux
    DifFlux[tstep] = ebt.Diff_Flux_west(Avg, vprime2, dx, Areas, Masks)
    
    #Internal Mixing
    IntMix[tstep] = ebt.Int_Mixing(tstep, vprime, Avg, dx, dy, dz, Masks)

#plotting
Resid = dsdt_dV + AdvFlux - DifFlux + IntMix

pred = -AdvFlux + DifFlux - IntMix
diff = pred - dsdt_dV
err = np.abs(diff/dsdt_dV*100)

Mix_Res = np.abs(Resid/IntMix)

plt.figure()
line0, = plt.plot(dsdt_dV, label = 'd/dt')
line1, = plt.plot(AdvFlux, label = 'Advective Flux')
line2, = plt.plot(DifFlux, label = 'Diffusive Flux')
line3, = plt.plot(IntMix, label = 'Internal Mixing')
line4, = plt.plot(Resid, label = 'Residual')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
plt.title('Monterey Bay Salinity Variance Budget (model time step)')
#plt.savefig('ExactBudget_Coastal_03Nov2020')

plt.figure()
plt.scatter(range(Mix_Res.size), Mix_Res)
plt.plot(Mix_Res)
plt.title('abs( residual/mixing )')
plt.xlim(-1, 24)
plt.ylim(0, 1)

plt.figure()
line0, = plt.plot(dsdt_dV, label = 'd/dt')
line1, = plt.plot(pred, label = 'predicted')
line2, = plt.plot(diff, label = 'difference')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

plt.figure()
plt.scatter(range(0,err.size), err)
plt.xlim(-1,err.size)
plt.title('% Error')