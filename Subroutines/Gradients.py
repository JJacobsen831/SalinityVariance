# -*- coding: utf-8 -*-
"""
Created on Tue Jul 28 14:27:51 2020

@author: Jasen
Tools to compute gradients within a control volume defined in roms output
"""
import numpy as np
import numpy.ma as ma
from netCDF4 import Dataset as nc4
import obs_depth_JJ as dep
import ROMS_Tools_Mask as rt
import GridShift

def x_grad_u(RomsFile, RomsGrd, varname) :
    """
    compute x-gradient on u points
    """
    
    if type(varname) == str :
        #load roms file
        RomsNC = nc4(RomsFile, 'r')
        #load variable
        _var = RomsNC.variables[varname][:]
        
    else:
        _var = varname #[u points]
    
    #get mask
    Mask = ma.getmask(_var)
    
    #gradient in x direction [rho points]
    dvar_rho = ma.diff(_var, n = 1, axis = 3)
    
    #pad 
    mask_pad = ma.notmasked_edges(dvar_rho, axis = 3)
    dvar_ = dvar_rho
    dvar_[:,:,:, mask_pad[0][3][0]-1] = dvar_rho[:,:,:,mask_pad[0][3][0]]
    dvar_[:,:,:,mask_pad[1][3][0]+1] = dvar_rho[:,:,:,mask_pad[1][3][0]]
    dvar_pad = ma.concatenate((dvar_rho[:,:,:,0:1], dvar_rho, \
                               dvar_rho[:,:,:,-2:-1]), axis = 3)
    
    #shift to u points
    dvar_u = GridShift.Rho_to_Upt(dvar_pad)
    
    #dx [rho points]
    x_dist = rt.rho_dist_grd(RomsGrd)[0]
    
    #repeat over depth and time and apply mask
    _dx = rt.AddDepthTime(RomsFile, x_dist)
    
    dx = ma.array(GridShift.Rho_to_Upt(_dx), mask = Mask)
    
    #gradient
    dvar_dx = dvar_u/dx
    
    return dvar_dx


def x_grad_rho(RomsFile, RomsGrd, varname) :
    """
    Compute x-gradient assuming rectangular grid cells
    """
    
    
    if type(varname) == str :
        #load roms file
        RomsNC = nc4(RomsFile, 'r')
        
        #load variable
        _var = RomsNC.variables[varname][:]
        
    else:
        _var = varname
    
    #get mask
    Mask = ma.getmask(_var)
    
    #gradient in x direction on u points
    dvar_u = np.diff(_var, n = 1, axis = 3)
    
    #shift u points to rho points
    dvar = GridShift.Upt_to_Rho(dvar_u)
    
    #lon positions [meters]
    x_dist = rt.rho_dist_grd(RomsGrd)[0]
    
    #repeat over depth and time space and apply mask
    dx = ma.array(rt.AddDepthTime(RomsFile, x_dist), mask = Mask)
    
    #compute gradient
    dvar_dx = dvar/dx
    
    return dvar_dx

def y_grad_rho(RomsFile, RomsGrd, varname):
    """
    Compute y-gradient assuming rectangular grid cells
    """
    if type(varname) == str :
        #load roms file
        RomsNC = nc4(RomsFile, 'r')
        #load variable
        _var = RomsNC.variables[varname][:]
        
    else:
        _var = varname
    
    #get mask
    Mask = ma.getmask(_var)
    
    #compute difference
    dvar_v = np.diff(_var, n = 1, axis = 2)
    
    #shift from v points to rho points
    dvar = GridShift.Vpt_to_Rho(dvar_v)
    
    #lon positions [meters]
    y_dist = rt.rho_dist_grd(RomsGrd)[1]
    
    #repeat over depth and time space and apply mask
    dy = ma.array(rt.AddDepthTime(RomsFile, y_dist), mask = Mask)
    
    #compute gradient
    dvar_dy = dvar/dy
    
    return dvar_dy

def y_grad_v(RomsFile, RomsGrd, varname):
    """
    Compute y-gradient on v points
    """
    if type(varname) == str :
        #load roms file
        RomsNC = nc4(RomsFile, 'r')
        #load variable
        _var = RomsNC.variables[varname][:]
        
    else:
        _var = varname #[v points]
    
    #get mask 
    Mask = ma.getmask(_var)
    
    #compute difference [rho points]
    dvar_rho = ma.diff(_var, n = 1, axis = 2)
    
    #pad
    mask_pad = ma.notmasked_edges(dvar_rho, axis = 2)
    dvar_ = dvar_rho
    dvar_[:,:,mask_pad[0][2][0]-1, :] = dvar_rho[:,:,mask_pad[0][2][0], :]
    dvar_[:,:,mask_pad[1][2][0]+1, :] = dvar_rho[:,:,mask_pad[1][2][0], :]
    
    dvar_pad = ma.concatenate((dvar_[:,:,0:1, :],\
                               dvar_, \
                               dvar_[:,:,-2:-1, :]), axis = 2)
    
    #shift to V points
    dvar_v = GridShift.Rho_to_Vpt(dvar_pad)
    
    #dy
    y_dist = rt.rho_dist_grd(RomsGrd)[1]
    dy =rt.AddDepthTime(RomsFile, y_dist)
    dy_v =  ma.array(GridShift.Rho_to_Vpt(dy), mask = Mask)
    
    #compute gradient
    dvar_dy = dvar_v/dy_v
    
    return dvar_dy
    

def z_grad(RomsFile, varname) :
    """
    Compute z-gradient assuming rectangular grid cells
    """
    #load roms file
    RomsNC = nc4(RomsFile, 'r')
    
    if type(varname) == str :
        #load variable
        var = RomsNC.variables[varname][:]
        
    else:
        var = varname
    
    #vertical difference
    dvar_w = np.diff(var, n = 1, axis = 1)
    
    #shift w points to rho points
    dvar = GridShift.Wpt_to_Rho(dvar_w)
    
    #depth on w points
    depth = dep._set_depth_T(RomsFile, None, 'w', \
                             RomsNC.variables['h'], RomsNC.variables['zeta'])
    
    #difference in depth on rho points
    d_dep = np.diff(depth, n = 1, axis = 1)
    
    #compute gradient
    dvar_dz = dvar/d_dep
    
    return dvar_dz



def x_grad_GridCor_Rho(RomsFile, RomsGrd, varname) :
    """
    Compute Grid Correction for gradients in x direction on rho points
    
    """
    #load roms file
    RomsNC = nc4(RomsFile, 'r')
    
    #check if variable is suppled or need to be loaded 
    if type(varname) == str :
        _var = RomsNC.variables[varname][:]
    else :
        _var = varname
    
    #get land mask
    Mask_rho = ma.getmask(_var)
    Mask_w = ma.concatenate((Mask_rho, Mask_rho[:,-2:-1, :,:]), axis = 1)
    
    #compute depth at rho and w points
    _rhodepth = ma.array(dep._set_depth_T(RomsFile, None, 'rho', \
                                          RomsNC.variables['h'][:], \
                                          RomsNC.variables['zeta'][:]), mask = Mask_rho)
    
    _wdepth = ma.array(dep._set_depth_T(RomsFile, None, 'w', \
                                        RomsNC.variables['h'][:], \
                                        RomsNC.variables['zeta'][:]), mask = Mask_w)
    
    #depth difference in vertical [rho points]
    dz_z = np.diff(_wdepth, n = 1, axis = 1)
    
    #depth difference of adjacent rho points [u points]
    _dz_x = np.diff(_rhodepth, n = 1, axis = 3)
    
    #shift to rho points
    dz_x = GridShift.Upt_to_Rho(_dz_x)
        
    # compute vertical differential [on w points]
    _dvar_z = np.diff(_var, n = 1, axis = 1)
    
    # shift to rho points
    dvar_z = GridShift.Wpt_to_Rho(_dvar_z)
    
    #distance between rho points in x direction
    _x_dist = rt.rho_dist_grd(RomsGrd)[0]
    
    #repeat over depth and time space & apply mask
    dx = ma.array(rt.AddDepthTime(RomsFile, _x_dist), mask = Mask_rho)
    
    #vertical gradient on rho points
    dvar_dz = dvar_z/dz_z
    
    #correction for roms grid
    dv_dxCor = dvar_dz*(dz_x/dx)
    
    return dv_dxCor

def y_grad_GridCor_Rho(RomsFile, RomsGrd, varname) :
    """
    Compute Grid Correction for gradients in y direction
    
    """
    #load roms file
    RomsNC = nc4(RomsFile, 'r')
    
    #check if variable supplied or needs to load
    if type(varname) == str :
        _var = RomsNC.variables[varname][:]
    else :
        _var = varname
    
    #get land mask
    Mask_rho = ma.getmask(_var)
    Mask_w = ma.concatenate((Mask_rho, Mask_rho[:,-2:-1, :,:]), axis = 1)
    
    #compute depth at rho points
    _rhodepth = ma.array(dep._set_depth_T(RomsFile, None, 'rho', \
                                          RomsNC.variables['h'][:], \
                                          RomsNC.variables['zeta'][:]), mask = Mask_rho)
    
    #compute depth at w points 
    _wdepth = ma.array(dep._set_depth_T(RomsFile, None, 'w', \
                                        RomsNC.variables['h'][:], \
                                        RomsNC.variables['zeta'][:]), mask = Mask_w)
    
    #depth difference in vertical [rho points]
    dz_z = np.diff(_wdepth, n = 1, axis = 1)
    
    #depth difference between adjacent rho points [v points]
    _dz_y = np.diff(_rhodepth, n = 1, axis = 2)
    
    #shift to rho points
    dz_y = GridShift.Vpt_to_Rho(_dz_y)
        
    #compute difference [w points]
    _dvar_z = ma.diff(_var, n = 1, axis = 1)
    
    #shift to rho points
    dvar_z = GridShift.Wpt_to_Rho(_dvar_z)
    
    #distance between rho points in x and y directions
    _y_dist = rt.rho_dist_grd(RomsGrd)[1]
    
    #repeat over depth and time space and add mask
    dy = ma.array(rt.AddDepthTime(RomsFile, _y_dist), mask = Mask_rho)
    
    #vertical gradient [rho points]
    dvar_dz = dvar_z/dz_z
    
    #correction for roms grid
    dv_dyCor = dvar_dz*(dz_y/dy)
    
    return dv_dyCor
