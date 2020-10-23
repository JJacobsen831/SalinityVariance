# -*- coding: utf-8 -*-
"""
Created on Fri Aug  7 16:54:11 2020

@author: Jasen
"""
import numpy as np


def Upt_to_Rho(Upt_variable) :
    """
    Converts variables on u points to rho points
    """
    if np.ma.is_masked(Upt_variable) == True :
        
        import numpy.ma as ma
        
    else:
        
        import numpy as ma
    
    
    _dx_pad = ma.concatenate((Upt_variable[:,0:1], Upt_variable,\
                              Upt_variable[:,-2:-1]), axis = 1)
    
    d_x = 0.5*(_dx_pad[:, 0:_dx_pad.shape[1]-1] + \
                _dx_pad[:, 1:_dx_pad.shape[1]])
    
    return d_x

def Vpt_to_Rho(Vpt_variable) :
    """
    Convert variable on v point to rho point
    """
    if np.ma.is_masked(Vpt_variable) == True :
        
        import numpy.ma as ma
        
    else:
        
        import numpy as ma
    
    _dy_pad = ma.concatenate((Vpt_variable[0:1,:], Vpt_variable,\
                              Vpt_variable[-2:-1,:]), axis = 0)
    
    
    d_y = 0.5*(_dy_pad[0:_dy_pad.shape[0]-1, :] + \
               _dy_pad[1:_dy_pad.shape[0], :])
    
    return d_y

def Rho_to_Upt(Rho_variable) :
    """
    Convert Rho point to U point
    """
    #horzontal average to U points
    Upt = 0.5*(Rho_variable[:, 0:Rho_variable.shape[1]-1] + \
                     Rho_variable[:, 1:Rho_variable.shape[1]])
    
    return Upt

def Rho_to_Vpt(Rho_variable) :
    """
    Convert Rho point to V point
    """
    Vpt = 0.5*(Rho_variable[0:Rho_variable.shape[0]-1, :] + \
                           Rho_variable[1:Rho_variable.shape[0], :])
    
    return Vpt
