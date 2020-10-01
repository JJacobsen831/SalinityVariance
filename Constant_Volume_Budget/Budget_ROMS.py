# -*- coding: utf-8 -*-
"""
Created on Thu Jul 30 15:16:12 2020

@author: Jasen
"""
import os
os.chdir('/Users/Jasen/Documents/GitHub/ROMS_Budget/')

import numpy.ma as ma
import Budget_Terms as bud
import ROMS_Tools_Mask as rt

import matplotlib.pyplot as plt

# files
RomsFile = '/Users/Jasen/Documents/Data/ROMS_TestRuns/wc12_his_5day.nc'
RomsGrd = '/Users/Jasen/Documents/Data/ROMS_TestRuns/wc12_grd.nc.0'

# variable
varname = 'salt'

#bounds of control volume
latbounds = [35, 37]
lonbounds = [-126, -125]

# define mask
RhoMask, _, _ = rt.RhoUV_Mask(RomsFile, latbounds, lonbounds)

#load and subset variable
salt = rt.ROMS_CV(varname, RomsFile, RhoMask) 

#deviation from volume mean in each time step
_prime = bud.Prime(salt)
S_prime = ma.array(_prime, mask = RhoMask)

#variance squared
S_var2 = S_prime*S_prime   
       
#Term 1: change in variance within control volume
dVar_dt = bud.TermOne(RomsFile, RhoMask, S_var2)

#Term 2: Flux of variance across boundary
Flux = bud.TermTwo(RomsFile, RomsGrd, varname, latbounds, lonbounds)
Flux_pt = 0.5*(Flux[0:Flux.shape[0]-1] + Flux[1:Flux.shape[0]]) #dt points

#Term 4: Internal mixing - destruction of variance
Mixing = bud.TermFour(RomsFile, RomsGrd, S_prime)
Mixing_pt = 0.5*(Mixing[0:Mixing.shape[0]-1] + Mixing[1:Mixing.shape[0]])

#Term 3: Diffusion across boundaries
Diff = bud.TermThree(RomsFile, RomsGrd, varname, latbounds, lonbounds)
Diff_pt = -0.5*(Diff[0:Diff.shape[0]-1] + Diff[1:Diff.shape[0]])

#total
Total = dVar_dt + Flux_pt + Mixing_pt + Diff_pt

#Plotting
line0, = plt.plot(dVar_dt, label = 'd/dt')
line1, = plt.plot(Flux_pt, label = 'Flux')
line2, = plt.plot(Mixing_pt, label = 'Mixing')
line3, = plt.plot(Diff_pt, label = 'Diffusion')
line4, = plt.plot(Total, label = 'Total')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
#plt.savefig('Budget_18Aug2020.png')
