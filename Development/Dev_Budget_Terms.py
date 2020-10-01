# -*- coding: utf-8 -*-
"""
Created on Wed Aug  5 11:02:37 2020

@author: Jasen
"""
import os
os.chdir('/Users/Jasen/Documents/GitHub/ROMS_Budget/')

import numpy as np
import numpy.ma as ma
import ROMS_Tools_Mask as rt
from netCDF4 import Dataset as nc4
import Differential as df
import GridShift
import Gradients as gr
import Filters as flt
import Budget_Terms as bud

RomsFile = '/Users/Jasen/Documents/Data/ROMS_TestRuns/wc12_his_5day.nc'
RomsGrd = '/Users/Jasen/Documents/Data/ROMS_TestRuns/wc12_grd.nc.0'

# variable
varname = 'salt'

#bounds of control volume
latbounds = [35, 37]
lonbounds = [-126, -125]

RhoMask, _, _ = rt.RhoUV_Mask(RomsFile, latbounds, lonbounds)

#load and subset variable
salt = rt.ROMS_CV(varname, RomsFile, RhoMask) 

#deviation from volume mean in each time step
_prime = bud.Prime(salt)
variance = ma.array(_prime, mask = RhoMask)

## term 4 mixing
def TermFour(RomsFile, RomsGrd, variance) :
    """
    Internal mixing
    """
    RomsNC = nc4(RomsFile, 'r')
    
    #vertical viscosity coefficient K
    _Kv_w = RomsNC.variables['AKv'][:]
    
    #average to rho points
    Kv_rho = 0.5*(_Kv_w[:,0:_Kv_w.shape[1]-1, :, :] + _Kv_w[:, 1:_Kv_w.shape[1], :, :])
    
    #apply mask
    Kv = ma.array(Kv_rho, mask = ma.getmask(variance))
    #horizontal viscosity coefficient
    #??
    
    #gradients squared
    #xgrad = ma.array(gr.x_grad(RomsFile, RomsGrd, variance)**2, \
    #                 mask = ma.getmask(variance))
    #ygrad = ma.array(gr.y_grad(RomsFile, RomsGrd, variance)**2, \
    #                 mask = ma.getmask(variance))
    zgrad = ma.array(gr.z_grad(RomsFile, variance)**2, \
                     mask = ma.getmask(variance))
    
    #
    #differentail volume
    dV = ma.array(df.dV(RomsFile), mask = ma.getmask(variance))
    
    #integrad
    z_m = 2*Kv*zgrad*dV
    
    #integrate
    mixing = np.empty(z_m.shape[0])
    mixing.fill(np.nan)
    for n in range(mixing.shape[0]) :
        mixing[n] = np.sum(z_m[n,:,:,:])
    
    return mixing

#term 3 diffusion
def TermThree(RomsFile, RomsGrd, varname, latbounds, lonbounds) :
    """
    Diffusion of variance across open boundaries
    """
    RomsNC = nc4(RomsFile, 'r')
    var = RomsNC.variables[varname][:]
    
    #shift variable from rho points to u and v points
    _var_u = 0.5*(var[:,:, :, 0:var.shape[3]-1] + var[:,:, :, 1:var.shape[3]])
    _var_v = 0.5*(var[:,:, 0:var.shape[2]-1, :] + var[:,:, 1:var.shape[2], :])
    
    #define masks for u and v points
    _, U_Mask, V_Mask = rt.RhoUV_Mask(RomsFile, latbounds, lonbounds)
    
    #compute variance 
    var_u = ma.array(_var_u, mask = U_Mask)
    var_v = ma.array(_var_v, mask = V_Mask)
    
    #variance squared
    prime2_u = (var_u - ma.mean(var_u))**2
    prime2_v = (var_v - ma.mean(var_v))**2
    
    #define face masks
    NorthFace, WestFace, SouthFace, EastFace = rt.FaceMask(RomsFile,\
                                                       latbounds, lonbounds)
    
    #apply face masks to variance squared
    North_var = ma.array(prime2_v, mask = NorthFace)
    South_var = ma.array(prime2_v, mask = SouthFace)
    West_var = ma.array(prime2_u, mask = WestFace)
    East_var = ma.array(prime2_u, mask = EastFace)
    
    #compute differential areas
    Ax_norm, Ay_norm = df.dA(RomsFile, RomsGrd)
    
    #subset areas
    North_Ay = ma.array(Ay_norm, mask = NorthFace)
    West_Ax = ma.array(Ax_norm, mask = WestFace)
    South_Ay = ma.array(Ay_norm, mask = SouthFace)
    East_Ax = ma.array(Ax_norm, mask = EastFace)
    
    #load diffusion coef
    _Ks_w = RomsNC.variables['AKs'][:]
    
    #shift to u points
    Ks_u = GridShift.Wpt_to_Upt(_Ks_w)
    
    #shift to v points
    Ks_v = GridShift.Wpt_to_Vpt(_Ks_w)
    
    #apply face mask to east and west faces
    East_Ks = ma.array(Ks_u, mask = EastFace)
    West_Ks = ma.array(Ks_u, mask = WestFace)
    
    #apply face mask to north and south faces
    North_Ks = ma.array(Ks_v, mask = NorthFace)
    South_Ks = ma.array(Ks_v, mask = SouthFace)
    
    #integrad
    North = -1*North_Ks*North_var*North_Ay
    South = South_Ks*South_var*South_Ay
    East = -1*East_Ks*East_var*East_Ax
    West = West_Ks*West_var*West_Ax
    
    #integrate
    Diff = np.empty(var.shape[0])
    Diff.fill(np.nan)
    for n in range(Diff.shape[0]):
        Diff[n] = ma.sum(North[n,:,:,:]) + ma.sum(South[n,:,:,:]) + ma.sum(West[n,:,:,:]) + ma.sum(East[n,:,:,:])
    
    return Diff
    


def Term2(RomsFile, RomsGrd, varname, latbounds, lonbounds) :
    """
    Flux of variance across open boundaries
    """
    RomsNC = nc4(RomsFile, 'r')
    var = RomsNC.variables[varname][:]
    
    #shift variable from rho points to u and v points
    _var_u = 0.5*(var[:,:, :, 0:var.shape[3]-1] + var[:,:, :, 1:var.shape[3]])
    _var_v = 0.5*(var[:,:, 0:var.shape[2]-1, :] + var[:,:, 1:var.shape[2], :])
    
    #define masks for u and v points
    _, U_Mask, V_Mask = rt.RhoUV_Mask(RomsFile, latbounds, lonbounds)
    
    #compute variance 
    var_u = ma.array(_var_u, mask = U_Mask)
    var_v = ma.array(_var_v, mask = V_Mask)
    
    #variance squared
    prime2_u = (var_u - ma.mean(var_u))**2
    prime2_v = (var_v - ma.mean(var_v))**2
    
    #define face masks
    NorthFace, WestFace, SouthFace, EastFace = rt.FaceMask(RomsFile,\
                                                       latbounds, lonbounds)
    
    #apply face masks to variance squared
    North_var = ma.array(prime2_v, mask = NorthFace)
    South_var = ma.array(prime2_v, mask = SouthFace)
    West_var = ma.array(prime2_u, mask = WestFace)
    East_var = ma.array(prime2_u, mask = EastFace)
    
    #compute differential areas
    Ax_norm, Ay_norm = df.dA(RomsFile, RomsGrd)
    
    #subset areas
    North_Ay = ma.array(Ay_norm, mask = NorthFace)
    West_Ax = ma.array(Ax_norm, mask = WestFace)
    South_Ay = ma.array(Ay_norm, mask = SouthFace)
    East_Ax = ma.array(Ax_norm, mask = EastFace)
    
    #velocities
    North_V = ma.array(RomsNC.variables['v'][:], mask =  NorthFace)
    West_U = ma.array(RomsNC.variables['u'][:], mask = WestFace)
    South_V = ma.array(RomsNC.variables['v'][:], mask = SouthFace)
    East_U = ma.array(RomsNC.variables['u'][:], mask = EastFace)
    
    #multiply to get integrad
    North = North_V*North_Ay*North_var
    West = West_U*West_Ax*West_var
    South = South_V*South_Ay*South_var
    East = East_U*East_Ax*East_var
    
    #sum/integrate each time step
    Flux = np.empty(North.shape[0])
    Flux.fill(np.nan)
    for n in range(North.shape[0]) :
        Flux[n] = ma.sum(North[n,:,:,:]) + ma.sum(South[n,:,:,:]) + \
                  ma.sum(East[n,:,:,:]) + ma.sum(West[n,:,:,:])

    return Flux
    

def RhoPsiIndex_Mask(RomsFile, latbounds, lonbounds):
    """Locates indices of lat and lon within ROMS Output File with logical mask"""
    #load roms file
    RomsNC = nc4(RomsFile, 'r')
    var = RomsNC.variables['salt'][:]
    
    Rholats = (RomsNC.variables['lat_rho'][:] >= latbounds[0])*(RomsNC.variables['lat_rho'][:] <= latbounds[1])
    Rholons = (RomsNC.variables['lon_rho'][:] >= lonbounds[0])*(RomsNC.variables['lon_rho'][:] <= lonbounds[1])
    RHOMASK = np.invert(Rholats*Rholons)
    
    Psilats = (RomsNC.variables['lat_psi'][:] >= latbounds[0])*(RomsNC.variables['lat_psi'][:] <= latbounds[1])
    Psilons = (RomsNC.variables['lon_psi'][:] >= lonbounds[0])*(RomsNC.variables['lon_psi'][:] <= lonbounds[1])
    PSIMASK = np.invert(Psilats*Psilons)
    
    #repeat masks over depth and time dimensions
    _RM = np.repeat(np.array(RHOMASK)[np.newaxis, :, :], var.shape[1], axis = 0)
    RhoMask = np.repeat(np.array(_RM)[np.newaxis, :, :, :], var.shape[0], axis = 0)
    
    _PM = np.repeat(np.array(PSIMASK)[np.newaxis, :, :], var.shape[1], axis = 0)
    PsiMask = np.repeat(np.array(_PM)[np.newaxis, :, :, :], var.shape[0], axis = 0)
    
    
    return RhoMask, PsiMask
