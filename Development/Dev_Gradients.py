# -*- coding: utf-8 -*-
"""
Created on Wed Jul 29 13:16:20 2020
>>>>>>>>>>>>>MOST OF THIS IS EITHER WRONG OR DOES NOT WORK<<<<<<<<<<<<<<<<<<<
>>>>>>>>>>>>>     THIS IS A REPOSITORY FOR TESTING CODE   <<<<<<<<<<<<<<<<<<<
@author: Jasen
"""
import numpy as np
from netCDF4 import Dataset as dt
import obs_depth_JJ as dep
from ROMS_Tools_Mask import rho_dist_grd as dist
import Gradients as gr

dsdx = gr.x_grad(RomsFile, RomsGrd, 'salt')

x_cor = gr.x_grad_GridCor(RomsFile, RomsGrd, 'salt')

dsdy = gr.y_grad(RomsFile, RomsGrd, 'salt')

y_cor = gr.y_grad_GridCor(RomsFile, RomsGrd, 'salt')

x_ratio = x_cor/dsdx
y_ratio = y_cor/dsdy

xr = np.array(x_ratio).flatten()
yr = np.array(y_ratio).flatten()


def x_grad_GridCor00(RomsFile, RomsGrd, varname) :
    """
    compute gradient in x (lon) direction
    """
    #load roms file
    RomsNC = dt(RomsFile, 'r')
    
    #load variable and compute differential
    var = RomsNC.variables[varname][:]
    dvar_x = np.diff(var, n = 1, axis = 3)
    dvar_z = np.diff(var, n = 1, axis = 1)
    
    #compute depth at rho points
    depth = dep._set_depth_T(RomsFile, None, 'rho', RomsNC.variables['h'],RomsNC.variables['zeta'])
       
    #distance between rho points in x and y directions
    x_dist = dist(RomsGrd)[0]
    
    #repeat over depth and time space
    _DX = np.repeat(np.array(x_dist)[np.newaxis, :, :], depth.shape[1], axis = 0)
    dx = np.repeat(np.array(_DX)[np.newaxis, :, :, :], depth.shape[0], axis = 0)
    
    #depth difference between adjacent rho points
    dz_x = np.diff(depth, n = 1, axis = 3)
    
    #vertical derivative
    dz_z = np.diff(depth, n = 1, axis = 1)
    dp_dz0 = dvar_z/dz_z
    dp_dz = 0.5*(dp_dz0[:,:,:,0:_dp_dz.shape[3]-1] + dp_dz0[:,:,:, 1:dp_dz0.shape[3]])
    
    #distance between adjacent rho points
    dl = np.sqrt(dx*dx + dz*dz)
    
    #correction for roms grid
    dp_dl = dvar/dl*dl/dx
    
    #gradient
    dp_dx = dvar/dx
    
    rat = np.abs(dp_dl[0,:,:,:])/np.abs(dp_dx[0,:,:,:])
    
    grat = dl[0,:,:,:]/dx[0,:,:,:]
    
    stat = np.array(rat).flatten()
    
