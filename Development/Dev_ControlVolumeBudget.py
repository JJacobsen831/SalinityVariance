# -*- coding: utf-8 -*-
"""
Created on Sat Jun 27 11:22:20 2020

@author: Jasen
"""
import os
os.chdir('/Users/Jasen/Documents/GitHub/ROMS_Budget/')
import ROMS_Tools as RT


#Locate file
RomsFile = '/Users/Jasen/Documents/Data/ROMS_TestRuns/wc12_his_5day.nc'
RomsGrd = '/Users/Jasen/Documents/Data/ROMS_TestRuns/wc12_grd.nc.0'

#select variable
varname = 'salt'

#coordinates of desired control volume
latbounds = [35, 37]
lonbounds = [-126, -125]

#create control volume
IndBounds = RT.RhoPsiIndex(RomsFile, latbounds, lonbounds)

# Extract Data at mask
ROMS_CV = RT.ROMS_CV_Load(RomsFile, 'salt', IndBounds)

#compute depth
ROMS_CV['depth_w'] = RT.ModelDepth(RomsFile, 'w', latbounds, lonbounds)

#Variance budget
#Terms 1 & 2: define scalar values of time average and purturbation
Salt = RT.Reynolds_Avg(ROMS_CV['salt'])

#Term 3: change in volume integrated purturbation
#   square prime in each cell

#   multiply each by cell size dx dy dz

#   sum each term over total volume

#   recompute for each time step

#Term 4: internal variance diffusion
#   Correct for tilted sigma-surface

#   d prime/dx, d prime/dy, d prime/dz

#   d/dx(Kh*d prime/dx), d/dy(Kh* d prime/dy)

#   compute Kv by interpolating to rho point

#   d/dz(Kv*d prime/dz)

#Term 5: advective flux through boundary
#   offshore face: u*prime^2*cell size in each cell, sum

#   inshore face: u*prime^2*cell size in each cell, sum

#   north face: v*prime^2*cell size in each cell, sum

#   south face: v*prime^2*cell size in each cell, sum

#Term 6: diffusive flux through boundary
#   offshore face

#   inshore face

#   north face

#   south face

