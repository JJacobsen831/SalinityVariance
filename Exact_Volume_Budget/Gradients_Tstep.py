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
import GridShift3D as GridShift

#distance terms for gradients
def xPad(Var) :
    """
    Insert copy of array along x dimension at mask edges
    """
    mask_pad = np.ma.notmasked_edges(Var, axis = 2)
    dvar_ = Var
    dvar_[:,:, mask_pad[0][2][0]-1] = Var[:,:,mask_pad[0][2][0]]
    dvar_[:,:,mask_pad[1][2][0]+1] = Var[:,:,mask_pad[1][2][0]]
    dvar_pad = np.ma.concatenate((dvar_[:,:,0:1], Var, \
                               dvar_[:,:,-2:-1]), axis = 2)
    return dvar_pad

def yPad(Var) :
    """
    Insert copy of array along y dimension at mask edges
    """
    mask_pad = np.ma.notmasked_edges(Var, axis = 1)
    dvar_ = Var
    dvar_[:,mask_pad[0][1][0]-1, :] = Var[:,mask_pad[0][1][0], :]
    dvar_[:,mask_pad[1][1][0]+1, :] = Var[:,mask_pad[1][1][0], :]
    
    dvar_pad = np.ma.concatenate((dvar_[:,0:1, :],\
                               Var, \
                               dvar_[:,-2:-1, :]), axis = 1)
    return dvar_pad

def x_grad_u(RomsNC, var) :
    """
    compute x-gradient on u points shifted to rho grid
    """
    #get mask
    Mask = ma.getmask(var)
    
    #gradient in x direction [u points]
    dvar_rho = ma.diff(var, n = 1, axis = 2)
    
    #pad edges
    dvar_pad = xPad(dvar_rho)
    
    #shift to u points
    dvar_u = GridShift.Rho_to_Upt(dvar_pad)
    
    #dx on rho points (identical to u points shifted; regular horizontal grid)
    dx = ma.array(np.repeat(1/np.array(RomsNC.variables['pm'][:])[np.newaxis, :, :], \
                            var.shape[0], axis = 0), mask = Mask)
    #gradient
    dvar_dx = dvar_u/dx
    
    return dvar_dx


def x_grad_rho(RomsNC, var) :
    """
    Compute x-gradient on rho points with rho dimension
    """
    #get mask
    Mask = ma.getmask(var)
    
    #shift rho points to u points
    var_pad = xPad(var)
    var_u = GridShift.Rho_to_Upt(var_pad)
    
    #gradient in x direction on rho points
    dvar_rho = np.diff(var_u, n = 1, axis = 2)
    
    #dx on rho points
    dx = ma.array(np.repeat(1/np.array(RomsNC.variables['pm'][:])[np.newaxis, :, :], \
                            var.shape[0], axis = 0), mask = Mask)
    
    #compute gradient
    dvar_dx = dvar_rho/dx
    
    return dvar_dx

def y_grad_rho(RomsNC, var):
    """
    Compute y-gradient on rho points
    """
    #get mask
    Mask = ma.getmask(var)
    
    #shift rho points to v points
    var_pad = yPad(var)
    var_v = GridShift.Rho_to_Vpt(var_pad)
    
    #compute difference [rho points]
    dvar_rho = np.diff(var_v, n = 1, axis = 1)
    
    #dy on rho points
    dy = ma.array(np.repeat(1/np.array(RomsNC.variables['pn'][:])[np.newaxis, :, :], \
                            var.shape[0], axis = 0), mask = Mask)
    
    #compute gradient
    dvar_dy = dvar_rho/dy
    
    return dvar_dy

def y_grad_v(RomsNC, var):
    """
    Compute y-gradient on v points that are shifted to rho grid
    """
    #get mask 
    Mask = ma.getmask(var)
    
    #compute difference [v points]
    dvar_rho = ma.diff(var, n = 1, axis = 1)
    
    #pad edges of mask
    dvar_pad = yPad(dvar_rho)
    
    #shift V points to rho grid
    dvar_v = GridShift.Rho_to_Vpt(dvar_pad)
    
    #dy on rho points
    dy = ma.array(np.repeat(1/np.array(RomsNC.variables['pn'][:])[np.newaxis, :, :], \
                            var.shape[0], axis = 0), mask = Mask)
        
    #compute gradient on v
    dvar_dy = dvar_v/dy
    
    return dvar_dy
    

def z_grad(tstep, RomsFile, var) :
    """
    Compute z-gradient on rho points
    """
    #load roms file
    RomsNC = nc4(RomsFile, 'r')
    
    #get Mask
    Mask = ma.getmask(var)
    
    #vertical difference of variable
    dvar_w = ma.diff(var, n = 1, axis = 0)
    
    #shift w points to rho points (padding in subroutine)
    dvar = GridShift.Wpt_to_Rho(dvar_w)
    
    #depth on w points
    depth = dep._set_depth(RomsFile, None, 'w', \
                           RomsNC.variables['h'], \
                           RomsNC.variables['zeta'][tstep, :, :])
    
    
    #difference in depth on rho points
    d_dep = ma.array(np.diff(depth, n = 1, axis = 0), mask = Mask)
    
    #compute gradient
    dvar_dz = dvar/d_dep
    
    return dvar_dz
