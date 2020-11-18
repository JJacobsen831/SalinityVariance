# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 08:39:44 2020

@author: jjacob2
"""
import os
os.chdir('/home/jjacob2/python/Salt_Budget/SalinityVarianceBudget/Exact_Volume_Budget/')

#paths to different runs
#
#daily output
FilePath = '/home/cae/runs/jasen/wc15.a01.b03.hourlywindWT.windmcurrent.diags/out/'
HistFile = FilePath + 'ocean_his_2014_0005.nc'
Hist = nc4(HistFile, 'r')
AvgFile = FilePath + 'ocean_avg_2014_0005.nc'
Avg = nc4(AvgFile, 'r')
Diag = nc4(FilePath + 'ocean_dia_2014_0005.nc', 'r')


#single time step output
FilePath = '/home/cae/runs/jasen/wc15.a01.b03.hourlywindWT.windmcurrent.diags/out.jasen2/'
HistFile = FilePath + 'ocean_his_2014.nc'
Hist = nc4(HistFile, 'r')
AvgFile = FilePath + 'ocean_avg_2014.nc'
Avg = nc4(AvgFile, 'r')
Diag = nc4(FilePath + 'ocean_dia_2014.nc', 'r')

#ourly time step output
FilePath = '/home/cae/runs/jasen/wc15.a01.b03.hourlywindWT.windmcurrent.diags/out.jasen3/'
HistFile = FilePath + 'ocean_his_2014.nc'
Hist = nc4(HistFile, 'r')
AvgFile = FilePath + 'ocean_avg_2014.nc'
Avg = nc4(AvgFile, 'r')
Diag = nc4(FilePath + 'ocean_dia_2014.nc', 'r')

#control volume vertices
#
#south east corner for quick mask varification
#Vertices = np.array([[35.7, -124.215], [35.7, -124.0], [35.8, -124.0], [35.8, -124.215]])
