# -*- coding: utf-8 -*-
"""
Created on Fri Aug  7 16:54:11 2020

@author: Jasen
"""
import numpy as np

def Upad(Upt_variable) :
    """
    Pad edges of array with copies of edges on x/u/dim2 edge
    """
    PadArray = np.concatenate((Upt_variable[:,:,0:1], Upt_variable,\
                               Upt_variable[:,:,-2:-1]), axis = 2)
    return PadArray

    
def Vpad(Vpt_variable) :
    """
    Pad edges of array with copys of edges on y/u/dim1 edge
    """
    PadArray = np.concatenate((Vpt_variable[:,0:1,:], Vpt_variable, \
                               Vpt_variable[:,-2:-1,:]), axis = 1)
    return PadArray


def Wpad(Wpt_variable) :
    """
    Pad edges of array with copies of edges on z/w/dim0 edge
    """
    PadArray = np.concatenate((Wpt_variable[0:1,:,:], Wpt_variable, \
                               Wpt_variable[-2:-1,:,:]), axis = 0)
    return PadArray

def Upt_to_Rho(Upt_variable) :
    """
    Converts variables on u points to rho points
    """
    #copy edges before average
    _pad = Upad(Upt_variable)
    
    #average to rho points
    RhoVar = 0.5*(_pad[:,:, 0:_pad.shape[2]-1] + \
                _pad[:,:, 1:_pad.shape[2]])
    
    return RhoVar


def Vpt_to_Rho(Vpt_variable) :
    """
    Convert variable on v point to rho point
    """
    #copy edges
    _pad = Vpad(Vpt_variable)
    
    #average to rho points
    d_y = 0.5*(_pad[:, 0:_pad.shape[1]-1, :] + \
               _pad[:,1:_pad.shape[1], :])
    
    return d_y


def Rho_to_Upt(Rho_variable) :
    """
    Convert Rho point to U point
    """
    #horzontal average to U points
    Upt = 0.5*(Rho_variable[:, :, 0:Rho_variable.shape[2]-1] + \
                     Rho_variable[:, :, 1:Rho_variable.shape[2]])
    
    return Upt


def Rho_to_Vpt(Rho_variable) :
    """
    Convert Rho point to V point
    """
    Vpt = 0.5*(Rho_variable[:, 0:Rho_variable.shape[1]-1, :] + \
                           Rho_variable[:, 1:Rho_variable.shape[1], :])
    
    return Vpt

    
def Wpt_to_Rho(Wpt_variable) :
    """
    convert variable on w point to rho point
    """
    #average to rho points
    d_z = 0.5*(Wpt_variable[0:Wpt_variable.shape[0]-1,:, :] + \
                Wpt_variable[1:Wpt_variable.shape[0], :, :])
    
    return d_z

def Wpt_to_Rho_pad(Wpt_variable) :
    """
    Convert variable on w points with pad to maintain dimension size
    """
    _pad = Wpad(Wpt_variable)
    
    RhoPt = Wpt_to_Rho(_pad)
    
    return RhoPt
    

def Wpt_to_Upt(Wpt_variable) :
    """
    convert 'W-point' from vertical difference to U point
    """
    #horzontal average to box points (psi like points)
    BoxVar = 0.5*(Wpt_variable[:,:, 0:Wpt_variable.shape[3]-1] + \
                  Wpt_variable[:,:, 1:Wpt_variable.shape[3]])
    
    
    #average box points onto X points
    Upt = 0.5*(BoxVar[0:BoxVar.shape[1]-1, :, :] + \
                     BoxVar[1:BoxVar.shape[1], :, :])
    
    return Upt

def Wpt_to_Vpt(Wpt_variable) :
    """
    Convert W point variable to V point
    """
    #horizontal average to box points
    BoxVar = 0.5*(Wpt_variable[:, 0:Wpt_variable.shape[2]-1, :] + \
                           Wpt_variable[:, 1:Wpt_variable.shape[2], :])
    
       
    #averagge points to V points
    Vpt = 0.5*(BoxVar[0:BoxVar.shape[1]-1, :, :] + \
                     BoxVar[1:BoxVar.shape[1], :, :])
    
    return Vpt
