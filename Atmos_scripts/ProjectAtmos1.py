# -*- coding: utf-8 -*-
"""
Created on Tue May 31 17:36:15 2022

@author: guria
"""

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import ticker as mticker
import cartopy.crs as ccrs
from cartopy.util import add_cyclic_point
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from matplotlib.colors import DivergingNorm
import warnings
import os
import datetime
warnings.filterwarnings('ignore')

#y=int(input('year? '))
#mo=int(input('month? '))
def valid_date(y,m,d):
        correctDate = None
        try:
            newDate = datetime.datetime(y,m,d)
            correctDate = True
        except ValueError:
            correctDate = False
        return correctDate
    
diri=r"C:\Users\guria\trmm"
valid_paths = next(os.walk(diri), (None, None, []))[2]   
#print(valid_paths)   
    
def precipitation(y,mo):
      # [] if no file
    global valid_paths
    nc_files=[]
    for date in range (1,32):
        if(valid_date(y,mo,date)==True):
            dd=datetime.datetime(y,mo,date).strftime("%Y%m%d")
        
            path=diri+r'\3B42_Daily.'+dd+r'.7.nc4'
            
            if '3B42_Daily.'+dd+'.7.nc4' in valid_paths:
                
                nc_files.append(diri+r'\3B42_Daily.'+dd+r'.7.nc4')
    print(nc_files)
    ds = xr.open_mfdataset(nc_files,engine='netcdf4',combine="nested",concat_dim="BeginDate")
   # print(ds)
    precip=[]
    for i in range(len(ds)):
        precip.append(ds['precipitation'][i])
    
    return (precip)

def precipitation_daily(y,mo,date):
      # [] if no file
    global valid_paths
    nc_files=[]
    
    if(valid_date(y,mo,date)==True):
        dd=datetime.datetime(y,mo,date).strftime("%Y%m%d")
        
        path=diri+r'\3B42_Daily.'+dd+r'.7.nc4'
            
        if r'3B42_Daily.'+dd+r'.7.nc4' in valid_paths:
            #print(r'3B42_Daily.'+dd+r'.7.nc4')
            ds = xr.open_dataset(path,engine='netcdf4')   
            return (ds.precipitation)
    return []

for year in range(2009,2010):
    for month in range(4,9):   
        for date in range(1,32):
            precip=precipitation_daily(year,month,date)
            if (precip.any()!=0):
                proj=ccrs.PlateCarree(central_longitude=0)           
                #print("here")
                fig = plt.figure(figsize=(8,7))
                ax1 = plt.subplot(111, projection=proj)
                plt.title("Year="+str(year)+" Month="+str(month)+"Date="+str(date))
                ax1.set_yticks([-90, -60, -30, 0, 30, 60, 90], crs=ccrs.PlateCarree())
                ax1.set_xticks([0, 60, 120, 180, 240, 300, 359.99], crs=ccrs.PlateCarree())
                lon_formatter = LongitudeFormatter(zero_direction_label=True,dateline_direction_label=False,number_format='.0f')
                ax1.xaxis.set_major_formatter(lon_formatter)
                lat_formatter = LatitudeFormatter()
                ax1.yaxis.set_major_formatter(lat_formatter)
                ax1.coastlines(linewidth=1, alpha=0.5)
                # ax1.set_xlim(70,90)
                # ax1.set_ylim(0,35)
                con_levs = np.arange(0, 150, 10)
                
                
                im1 = ax1.contourf(precip['lon'],precip['lat'],precip.T,levels=con_levs,cmap=plt.cm.twilight,transform=ccrs.PlateCarree())
                cbar_ax = fig.add_axes([0.95, 0.3, 0.015, 0.4])
                cbar = fig.colorbar(im1, cax=cbar_ax, orientation='vertical', extendrect=False)
                cbar.ax.tick_params(labelsize=14)
                cbar.ax.set_xlabel('mm', fontsize=14)
                filenomen="Year="+str(year)+" Month="+str(month)+"Date="+str(date)
                plt.savefig(filenomen)



'''
def anim(i):
    global month
    global year
    month+=1
    year+=month//12
    month=month%12
    ax1.clear()
    precip=precipitation(year,month)
    
   
    plt.text(0,0,"Year="+str(year)+" Month"+str(month))
    ax1.set_yticks([-90, -60, -30, 0, 30, 60, 90], crs=ccrs.PlateCarree())
    ax1.set_xticks([0, 60, 120, 180, 240, 300, 359.99], crs=ccrs.PlateCarree())
    lon_formatter = LongitudeFormatter(zero_direction_label=True,dateline_direction_label=False,number_format='.0f')
    ax1.xaxis.set_major_formatter(lon_formatter)
    lat_formatter = LatitudeFormatter()
    ax1.yaxis.set_major_formatter(lat_formatter)
    ax1.coastlines(linewidth=1, alpha=0.5)
    # ax1.set_xlim(70,90)
    # ax1.set_ylim(0,35)
    con_levs = np.arange(0, 200, 10)
    #precip1=precipitation(y, mo)
     
    im1 = ax1.contourf(precip['lon'],precip['lat'],precip.T,levels=con_levs,cmap=plt.cm.twilight,transform=ccrs.PlateCarree())
    cbar_ax = fig.add_axes([0.95, 0.3, 0.015, 0.4])
    cbar = fig.colorbar(im1, cax=cbar_ax, orientation='vertical', extendrect=False)
    cbar.ax.tick_params(labelsize=14)
    cbar.ax.set_xlabel('mm', fontsize=14)
anim=animation.FuncAnimation(fig,anim, frames=10,interval=0.5)
'''
#plt.show()