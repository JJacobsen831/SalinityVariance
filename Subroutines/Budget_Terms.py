# -*- coding: utf-8 -*-
"""
Created on Mon Aug  3 15:27:30 2020

Compute each term in variance budget

@author: Jasen
"""
import numpy as np
import numpy.ma as ma
from netCDF4 import Dataset as nc4
import Differential as df
import ROMS_Tools_Mask as rt
import Gradients as gr
import GridShift

def Prime(var_4dim) :
    """
    Compute anomaly of variable for each time step
    var_4dim = var_4dim[time, depth, lat, lon]
    """
    var_prime = np.empty(var_4dim.shape)
    var_prime.fill(np.nan)
    for n in range(var_4dim.shape[0]) :
        var_prime[n, :, :, :] = var_4dim[n,:,:,:] - ma.mean(var_4dim[n, :, :, :])
    
    return var_prime

def TermOne(RomsFile, Mask, Variance) :
    """
    Compute the change of variance within a control volume
    """
    #ocean time
    RomsNC = nc4(RomsFile, 'r')
    time = RomsNC.variables['ocean_time']
    
    #differential volume
    dv = df.dV(RomsFile)
    dv = ma.array(dv, mask = Mask)
    
    #multiply to get integrad
    S_m = Variance*dv
    
    #integrate volume at each time step
    S_int = np.empty(S_m.shape[0])
    S_int.fill(np.nan)
    for n in range(S_int.shape[0]) :
        S_int[n] = np.sum(S_m[n, :, :, :])
        
    #time derivative
    dVar_dt = np.diff(S_int)/np.diff(time)
    
    return dVar_dt

def TermTwo(RomsFile, RomsGrd, varname, latbounds, lonbounds) :
    """
    Advective Flux of variance across open boundaries
    """
    RomsNC = nc4(RomsFile, 'r')
    var = RomsNC.variables[varname][:]
    
    #shift variable from rho points to u and v points
    _var_u = GridShift.Rho_to_Upt(var)
    _var_v = GridShift.Rho_to_Vpt(var)
    
    #define masks for u and v points
    _, U_Mask, V_Mask = rt.RhoUV_Mask(RomsFile, latbounds, lonbounds)
    
    #compute variance 
    var_u = ma.array(_var_u, mask = U_Mask)
    var_v = ma.array(_var_v, mask = V_Mask)
    
    #variance squared
    prime2_u = Prime(var_u)**2
    
    prime2_v = Prime(var_v)**2
    
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
    
    #velocities oriented into the control volume
    North_V = ma.array(-1*RomsNC.variables['v'][:], mask =  NorthFace)
    West_U = ma.array(RomsNC.variables['u'][:], mask = WestFace)
    South_V = ma.array(RomsNC.variables['v'][:], mask = SouthFace)
    East_U = ma.array(-1*RomsNC.variables['u'][:], mask = EastFace)
    
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

def TermThree(RomsFile, RomsGrd, varname, latbounds, lonbounds) :
    """
    Diffusive Flux of variance across open boundaries
    """
    RomsNC = nc4(RomsFile, 'r')
    var = RomsNC.variables[varname][:]
    
    #shift variable from rho points to u and v points
    _var_u = 0.5*(var[:,:, :, 0:var.shape[3]-1] + var[:,:, :, 1:var.shape[3]])
    _var_v = 0.5*(var[:,:, 0:var.shape[2]-1, :] + var[:,:, 1:var.shape[2], :])
    
    #define masks for u and v points
    _, U_Mask, V_Mask = rt.RhoUV_Mask(RomsFile, latbounds, lonbounds)
    
    #mask variable 
    var_u = ma.array(_var_u, mask = U_Mask)
    var_v = ma.array(_var_v, mask = V_Mask)
    
    #variance squared
    prime2_u = (var_u - ma.mean(var_u))**2
    prime2_v = (var_v - ma.mean(var_v))**2
    
    #gradients of variance squared
    dprime_u_dx = gr.x_grad_u(RomsFile, RomsGrd, prime2_u)
    dprime_v_dy = gr.y_grad_v(RomsFile, RomsGrd, prime2_v)
    
    #define face masks
    NorthFace, WestFace, SouthFace, EastFace = rt.FaceMask(RomsFile,\
                                                       latbounds, lonbounds)
    
    #apply face masks to gradient of variance squared
    North_var = ma.array(dprime_v_dy, mask = NorthFace)
    South_var = ma.array(dprime_v_dy, mask = SouthFace)
    West_var = ma.array(dprime_u_dx, mask = WestFace)
    East_var = ma.array(dprime_u_dx, mask = EastFace)
    
    #compute differential areas
    Ax_norm, Ay_norm = df.dA(RomsFile, RomsGrd)
    
    #subset areas
    North_Ay = ma.array(Ay_norm, mask = NorthFace)
    West_Ax = ma.array(Ax_norm, mask = WestFace)
    South_Ay = ma.array(Ay_norm, mask = SouthFace)
    East_Ax = ma.array(Ax_norm, mask = EastFace)
    
    #load horizontal diffusion coef
    Ks = RomsNC.variables['nl_tnu2'][0]
    
    #integrad
    North = -1*Ks*North_var*North_Ay
    South = Ks*South_var*South_Ay
    East = -1*Ks*East_var*East_Ax
    West = Ks*West_var*West_Ax
    
    #integrate
    Diff = np.empty(var.shape[0])
    Diff.fill(np.nan)
    for n in range(Diff.shape[0]):
        Diff[n] = ma.sum(North[n,:,:,:]) + ma.sum(South[n,:,:,:]) + ma.sum(West[n,:,:,:]) + ma.sum(East[n,:,:,:])
    
    return Diff


def TermFour(RomsFile, RomsGrd, variance) :
    """
    Internal mixing
    """
    RomsNC = nc4(RomsFile, 'r')
    
    #vertical viscosity coefficient K
    _Kv_w = RomsNC.variables['AKs'][:]
    
    #average to rho points
    Kv_rho = 0.5*(_Kv_w[:,0:_Kv_w.shape[1]-1, :, :] + _Kv_w[:, 1:_Kv_w.shape[1], :, :])
    
    #apply mask
    Kv = ma.array(Kv_rho, mask = ma.getmask(variance))
    
    #horizontal viscosity coefficient (constant)
    Kh = RomsNC.variables['nl_tnu2'][0]
    
    Mask = ma.getmask(variance)
    
    #gradients squared
    xgrad = ma.array(gr.x_grad_rho(RomsFile, RomsGrd, variance)**2, \
                     mask = Mask)
    ygrad = ma.array(gr.y_grad_rho(RomsFile, RomsGrd, variance)**2, \
                     mask = Mask)
    zgrad = ma.array(gr.z_grad(RomsFile, variance)**2, \
                     mask = Mask)
    
    #
    #differentail volume
    dV = ma.array(df.dV(RomsFile), mask = ma.getmask(variance))
    
    #integrad
    x_m = 2*Kh*xgrad*dV
    y_m = 2*Kh*ygrad*dV
    z_m = 2*Kv*zgrad*dV
    
    #integrate
    mixing = np.empty(z_m.shape[0])
    mixing.fill(np.nan)
    for n in range(mixing.shape[0]) :
        mixing[n] = ma.sum(z_m[n,:,:,:]) + ma.sum(x_m[n,:,:,:]) + ma.sum(y_m[n,:,:,:])
    
    return mixing
