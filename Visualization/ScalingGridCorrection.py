# -*- coding: utf-8 -*-
"""
Created on Thu Jul 30 12:12:20 2020

@author: Jasen
"""
import os
os.chdir('/Users/Jasen/Documents/GitHub/ROMS_Budget/')
import numpy as np
import numpy.ma as ma
import Gradients as gr
import matplotlib.pyplot as plt
from netCDF4 import Dataset as nc4
import ROMS_Tools_Mask as rt
import obs_depth_JJ as dep

#files
RomsFile = '/Users/Jasen/Documents/Data/ROMS_TestRuns/wc12_his_5day.nc'
RomsGrd = '/Users/Jasen/Documents/Data/ROMS_TestRuns/wc12_grd.nc.0'



#load bathymetry
RomsNC = nc4(RomsFile, 'r')
bathy = RomsNC.variables['h'][:]

#select variable
Salt = RomsNC.variables['salt'][:]
Prime2 = (Salt - ma.mean(Salt))**2
varname = Prime2

#horizontal gradients
ds_dx = gr.x_grad_rho(RomsFile, RomsGrd, varname)
ds_dy = gr.y_grad_rho(RomsFile, RomsGrd, varname)

#grid corrections
xCor = gr.x_grad_GridCor_Rho(RomsFile, RomsGrd, varname)
yCor = gr.y_grad_GridCor_Rho(RomsFile, RomsGrd, varname)

#ratio
x_rat = ma.array(ma.abs(xCor)/ma.abs(ds_dx))
y_rat = ma.array(ma.abs(yCor)/ma.abs(ds_dy))

#depth median ratio
DM_ratx = ma.median(x_rat[0,:,:,:], axis = 0)
DM_raty = ma.median(y_rat[0,:,:,:], axis = 0)

#mask land values
Land = ma.getmask(varname)
Landdt = rt.AddDepthTime(RomsFile, Land)


Xratio = ma.array(x_rat, mask = Land).flatten()
Yratio = ma.array(y_rat, mask = Land).flatten()

#remove mask
Xratio = Xratio[~Xratio.mask]
Yratio = Yratio[~Yratio.mask]

X_good = Xratio !=0
Y_good = Yratio !=0

xlog = ma.log10(Xratio[X_good])
ylog = ma.log10(Yratio[Y_good])

#depth
romsvars = {'h' : RomsNC.variables['h'][:], \
            'zeta' : RomsNC.variables['zeta'][:]}
    
#compute depth at rho points
depth = dep._set_depth_T(RomsFile, None, 'rho', romsvars['h'], romsvars['zeta'])
depth = ma.array(depth, mask = Land).flatten()

_dep = depth[X_good]
depthLR = _dep[xlog > 1]
depthLR = depthLR[~depthLR.mask]


#log median ration, skipping zeros
for n in range(DM_ratx.shape[0]) :
    ind = DM_ratx[n,:] != 0
    DM_ratx[n,ind] = ma.log10(DM_ratx[n,ind])

for n in range(DM_raty.shape[0]) :
    ind = DM_raty[n,:] != 0
    DM_raty[n,ind] = np.log10(DM_raty[n, ind])


#plotting
#histograms of magnitude of correction
fig0, (ax0, ax1) = plt. subplots(nrows = 1, ncols = 2, constrained_layout = True)
C0 = ax0.hist(xlog[~xlog.mask], bins = 40)
ax0.set_title('Magnitude of x correction ratio')
C1 = ax1.hist(ylog[~ylog.mask], bins = 40)
ax1.set_title('Magnitude of y correction ratio')
fig0.savefig('DepthResolvedMagnitudeCorrection.png')

#depth of large correction ratio
plt.hist(depthLR, bins = 40)
plt.title('Depth of log(ratio) > 1')
plt.xlabel('Depth [m]')
plt.text(-200, 30, ["median = " + str(round(ma.median(depthLR), 2))])
plt.savefig('Depth_log_1.png')

depthVLR = _dep[xlog > 0.5]
depthVLR = depthVLR[~depthVLR.mask]
plt.hist(depthVLR, bins = 40)
plt.title('Depth of log(ratio) > 0.5')
plt.xlabel('Depth [m]')
plt.text(-1800, 20000, ["median = " + str(round(ma.median(depthVLR), 2))])
plt.savefig('Depth_log_05.png')

#spatial map of median correction ratio
fig1, (ax1, ax2) = plt.subplots(nrows = 1, ncols = 2, constrained_layout=True)
#subplot 1
CS = ax1.contourf(DM_ratx, \
                  levels = [-2, -1.75, -1.5, -1.25, -1,  -0.75, -0.5, -0.25, 0, 0.5], cmap = plt.cm.inferno)
CS1 = ax1.contour(bathy, cmap = plt.cm.gray)
ax1.set_title('GridCor:ds_dx')
#subplot 2
CS = ax2.contourf(DM_raty, \
                  levels = [-2, -1.75, -1.5, -1.25, -1,  -0.75, -0.5, -0.25, 0, 0.5], cmap = plt.cm.inferno)
CS1 = ax2.contour(bathy, cmap = plt.cm.gray) 
ax2.set_title('GridCor:ds_dy')
fig1.colorbar(CS)
fig1.suptitle('Magnitude of Depth Median of Grid Corrention')

plt.figure(figsize=(20,10))

fig1.savefig('GridCor_Map.jpg')