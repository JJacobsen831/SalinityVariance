# -*- coding: utf-8 -*-
"""
Created on Tue Oct  6 12:26:00 2020

Tools to compute gradients within a control volume

@author: Jasen
"""

import numpy as np
import numpy.ma as ma
from netCDF4 import Dataset as nc4
import obs_depth_JJ as dep
import ROMS_Tools_Mask as rt
import GridShift3D as GridShift

#distance terms for gradients
def x_dist(RomsNC) :
    """
    compute dx for gradients 
    Cell widths centered on rho points
    """
    dx = 1/np.array(RomsNC.variables['pm'][:])
    
    return dx

def y_dist(RomsNC) :
    """
    compute dy for gradients
    Cell widths centered on rho points
    """
    dy = 1/np.array(RomsNC.variables['pn'][:])
    
    return dy

def x_grad_u(RomsNC, RomsGrd, var) :
    """
    compute x-gradient on u points
    """
    #get mask
    Mask = ma.getmask(var)
    
    #gradient in x direction [u points]
    dvar_rho = ma.diff(var, n = 1, axis = 2)
    
    #pad edges
    mask_pad = ma.notmasked_edges(dvar_rho, axis = 2)
    dvar_ = dvar_rho
    dvar_[:,:, mask_pad[0][3][0]-1] = dvar_rho[:,:,mask_pad[0][3][0]]
    dvar_[:,:,mask_pad[1][3][0]+1] = dvar_rho[:,:,mask_pad[1][3][0]]
    dvar_pad = ma.concatenate((dvar_rho[:,:,0:1], dvar_rho, \
                               dvar_rho[:,:,-2:-1]), axis = 2)
    
    #shift to u points
    dvar_u = GridShift.Rho_to_Upt(dvar_pad)
    
    #dx on rho points
    _dx = x_dist(RomsNC)
    
    #dx on u points
    dx = ma.array(GridShift.Rho_to_Upt(_dx), mask = Mask)
    
    #gradient
    dvar_dx = dvar_u/dx
    
    return dvar_dx


def x_grad_rho(RomsFile, RomsGrd, var) :
    """
    Compute x-gradient assuming rectangular grid cells
    """
    #get mask
    Mask = ma.getmask(var)
    
    #gradient in x direction on u points
    dvar_u = np.diff(var, n = 1, axis = 2)
    
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
