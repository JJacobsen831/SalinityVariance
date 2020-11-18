# -*- coding: utf-8 -*-
"""
Created on Tue Oct  6 12:26:00 2020

Tools to compute gradients within a control volume

@author: Jasen
"""

import numpy as np
import GridShift_3D as GridShift

def x_grad_u(var, dx, Masks) :
    """
    compute x-gradient on u points 
    """
    #convert to masked array
    var = np.ma.array(var, mask = np.invert(Masks['LandMask']))
    
    #gradient in x direction [u points]
    dvar_u = np.ma.diff(var, n = 1, axis = 2)
    
    #mask from shifted array
    land_u = np.ma.getmask(dvar_u)
       
    #dx on u points
    dx_u = GridShift.Rho_to_Upt(dx)
    
    #gradient
    dvar_dx = np.ma.array(dvar_u/dx_u, mask = land_u)
    
        
    return dvar_dx


def x_grad_rho(var, dx, Masks) :
    """
    Compute x-gradient on rho points with rho dimension
    """
    #convert to masked array
    var = np.ma.array(var, mask = np.invert(Masks['LandMask']))
    
    #gradient in x direction on u points
    dvar_u = np.ma.diff(var, n = 1, axis = 2)
    
    #land mask on u points    
    land_u = np.ma.getmask(dvar_u)
    land_rho = np.ma.concatenate((land_u, land_u[:,:,-2:-1]), axis = 2)
    
    #shift u points to rho points
    dvar_rho = np.ma.array(GridShift.Upt_to_Rho(dvar_u), mask =land_rho)
    
    #compute gradient
    dvar_dx = dvar_rho/dx
    
        
    return dvar_dx

def y_grad_rho(var, dy, Masks):
    """
    Compute y-gradient on rho points
    """
    #convert in to masked array
    var = np.ma.array(var, mask = np.invert(Masks['LandMask']))
    
    #compute difference [v points]
    dvar_v = np.ma.diff(var, n = 1, axis = 1)
    
    #land mask on v points
    land_v = np.ma.getmask(dvar_v)
    land_rho = np.ma.concatenate((land_v, land_v[:,-2:-1,:]), axis = 1)
    
    #shift v points to rho points
    dvar_rho = np.ma.array(GridShift.Vpt_to_Rho(dvar_v), mask = land_rho)
    
    #compute gradient
    dvar_dy = dvar_rho/dy
    
        
    return dvar_dy

def y_grad_v(var, dy, Masks):
    """
    Compute y-gradient on v points that are shifted to rho grid
    """
    #convert in to masked array
    var = np.ma.array(var, mask = np.invert(Masks['LandMask']))
    
    #gradient on v points
    dvar_v = np.diff(var, n = 1, axis = 1)
    
    #mask on v points
    land_v = np.ma.getmask(dvar_v)
    
    #shift V points to rho grid
    dy_v = GridShift.Rho_to_Vpt(dy)
       
    #compute gradient on v
    dvar_dy = np.ma.array(dvar_v/dy_v, mask = land_v)
    
       
    return dvar_dy
    

def z_grad(var, dz) :
    """
    Compute z-gradient on rho points
    """
    #vertical difference of variable
    dvar_w = np.diff(var, n = 1, axis = 0)
    
    #shift w points to rho points
    dvar_rho = GridShift.Wpt_to_Rho_pad(dvar_w)
    
    #compute gradient
    dvar_dz = dvar_rho/dz
    
    return dvar_dz
