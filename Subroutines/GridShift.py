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
    
    
    _dx_pad = ma.concatenate((Upt_variable[:,:,:,0:1], Upt_variable,\
                              Upt_variable[:, :,:,-2:-1]), axis = 3)
    
    d_x = 0.5*(_dx_pad[:,:,:, 0:_dx_pad.shape[3]-1] + \
                _dx_pad[:,:,:, 1:_dx_pad.shape[3]])
    
    return d_x

def Vpt_to_Rho(Vpt_variable) :
    """
    Convert variable on v point to rho point
    """
    if np.ma.is_masked(Vpt_variable) == True :
        import numpy.ma as ma
    else :
        import numpy as ma
    
    _dy_pad = ma.concatenate((Vpt_variable[:,:,0:1, :], Vpt_variable, \
                              Vpt_variable[:,:,-2:-1, :]), axis = 2)
    
    d_y = 0.5*(_dy_pad[:,:, 0:_dy_pad.shape[2]-1] + \
               _dy_pad[:,:,1:_dy_pad.shape[2]])
    
    return d_y

def Wpt_to_Rho(Wpt_variable) :
    """
    convert variable on w point to rho point
    """
    if np.ma.is_masked(Wpt_variable) == True :
        import numpy.ma as ma
    else :
        import numpy as ma
    
    dvar_pad = ma.concatenate((Wpt_variable[:,0:1, :, :], Wpt_variable, \
                               Wpt_variable[:,-2:-1, :, :]), axis = 1)
    
    #average to rho points
    d_z = 0.5*(dvar_pad[:,0:dvar_pad.shape[1]-1,:, :] + \
                dvar_pad[:,1:dvar_pad.shape[1], :, :])
    
    return d_z
    
    


def Wpt_to_Upt(Wpt_variable) :
    """
    convert 'W-point' from vertical difference to U point
    """
    if np.ma.is_masked(Wpt_variable) == True :
        import numpy.ma as ma
    else:
        import numpy as ma
    
    #horzontal average to box points (psi like points)
    BoxVar = 0.5*(Wpt_variable[:,:,:, 0:Wpt_variable.shape[3]-1] + \
                  Wpt_variable[:,:,:, 1:Wpt_variable.shape[3]])
    
    
    #average box points onto X points
    Upt = 0.5*(BoxVar[:, 0:BoxVar.shape[1]-1, :, :] + \
                     BoxVar[:, 1:BoxVar.shape[1], :, :])
    
    return Upt

def Wpt_to_Vpt(Wpt_variable) :
    """
    Convert W point variable to V point
    """
    if np.ma.is_masked(Wpt_variable) == True :
        import numpy.ma as ma
    else:
        import numpy as ma
    
    #horizontal average to box points
    BoxVar = 0.5*(Wpt_variable[:,:, 0:Wpt_variable.shape[2]-1, :] + \
                           Wpt_variable[:,:, 1:Wpt_variable.shape[2], :])
    
       
    #averagge points to V points
    Vpt = 0.5*(BoxVar[:, 0:BoxVar.shape[1]-1, :, :] + \
                     BoxVar[:, 1:BoxVar.shape[1], :, :])
    
    return Vpt

def Rho_to_Upt(Rho_variable) :
    """
    Convert Rho point to U point
    """
    #horzontal average to U points
    Upt = 0.5*(Rho_variable[:, :, :, 0:Rho_variable.shape[3]-1] + \
                     Rho_variable[:, :, :, 1:Rho_variable.shape[3]])
    
    return Upt

def Rho_to_Vpt(Rho_variable) :
    """
    Convert Rho point to V point
    """
    Vpt = 0.5*(Rho_variable[:,:, 0:Rho_variable.shape[2]-1, :] + \
                           Rho_variable[:,:, 1:Rho_variable.shape[2], :])
    
    return Vpt