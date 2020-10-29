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

#bounds of control volume (points in clockwise order)
Vertices = np.array([[37.0, -123.0], [37.0, -122.5], [37.2, -122.5], [37.2, -123.0]])


#load files
FilePath = '/home/cae/runs/jasen/wc15.a01.b03.hourlywindWT.windmcurrent.diags/out/'
HistFile = FilePath + 'ocean_his_2014_0005.nc'
Hist = nc4(HistFile, 'r')
AvgFile = FilePath + 'ocean_avg_2014_0005.nc'
Avg = nc4(AvgFile, 'r')
Diag = nc4(FilePath + 'ocean_dia_2014_0005.nc', 'r')
GridFile = '/home/ablowe/runs/ncfiles/grids/wc15.a01.b03_grd.nc'

#create masks
Masks = ebt.CreateMasks(Avg, Vertices)

#load time for steping
time = Avg.variables['ocean_time'][:]

#Prealocation 
dsdt_dV = np.empty(time.shape)
dsdt_dV.fill(np.nan)
AdvFlux = np.empty(time.shape)
AdvFlux.fill(np.nan)
DifFlux = np.empty(time.shape)
DifFlux.fill(np.nan)
IntMix = np.empty(time.shape)
DifFlux.fill(np.nan)

tstep = 0



##Loop on time step
for t in range(time) :
    #compute variance at time step
    var = Avg.variables['salt'][tstep, :, :, :]
    
    #increase percision for multiplication
    v_prime = np.array((var - var[Masks['RhoMask']].mean())*Masks['RhoMask'], dtype = np.float128)
    
    #then return to float 32
    v_prime2 = np.array(v_prime*v_prime, dtype = np.float32)
    v_prime = np.array(v_prime, dtype = np.float32)
   
    #compute depth at avg points and hist points
    depthAvg = dep._set_depth(AvgFile, None, 'w',\
                                    Avg.variables['h'][:],\
                                    Avg.variables['zeta'][tstep, :, :])
       
    #compute cell areas
    Areas = ebt.CellAreas(tstep, AvgFile, Avg, Masks)
    
    #time derivative of salinity variance squared
    dsdt_dV[tstep] = ebt.TimeDeriv(tstep, v_prime2, Hist, HistFile, Avg, AvgFile, Diag, Areas['Axy'], Masks)
    
    #Advective flux
    AdvFlux[tstep] = ebt.Adv_Flux(tstep, v_prime2, Avg, Areas, Masks)
    
    #Diffusive Flux
    DifFlux[t] = ebt.Diff_Flux(t, v_prime2, Avg, Masks, Areas)
    
    #Internal Mixing
    IntMix[t] = ebt.Int_Mixing(t, v_prime, Avg, AvgFile, GridFile, Masks)
