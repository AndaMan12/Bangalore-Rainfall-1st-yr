# -*- coding: utf-8 -*-
"""
Created on Mon Jul 11 19:10:39 2022

@author: guria
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Jun 30 19:02:21 2022

@author: guria
"""
import matplotlib.pyplot as plt
import imdlib as imd
import numpy as np
import scipy.integrate as integrate
import scipy.interpolate as interpolate
import warnings
warnings.filterwarnings('ignore')

start_yr =1901
end_yr=2018
def net_rain(start_yr,end_yr):
    variable = 'rain' # other options are ('tmin'/ 'tmax')
    file_format = 'yearwise' # other option (None), which will assume deafult imd naming convention
    file_dir = 'imd_base' # other option keep it blank
    data = imd.open_data(variable, start_yr, end_yr,'yearwise', file_dir)
    ds=data.get_xarray()
    ds=ds.where(ds['rain']!=-999)
    rainy=ds.rain
    rain=rainy.values
    index_lat= lambda lat: int((lat-6.5)/0.25)
    index_lon= lambda lon: int((lon-66.5)/0.25)
    lat= lambda index_lat: 6.5+index_lat*0.25
    lon= lambda index_lon: 66.5+index_lon*0.25
    Bangalore_lat_index=[index_lat(12.5),index_lat(13.5)]
    #print(Bangalore_lat_index)
    Bangalore_lon_index=[index_lon(77),index_lon(78)]
    #print(Bangalore_lon_index)
    
    
    days=rainy['time'].values.shape[0]
    
    
    monthly_avg_rain=np.zeros(shape=(end_yr-start_yr+2,12,Bangalore_lon_index[1]-Bangalore_lon_index[0]+1, 
                                     Bangalore_lat_index[1]-Bangalore_lat_index[0]+1))
    
    m=0
    y=0
    mon='01'
    yr='1901'
    day=0
    for i in range(days):
       
        if (mon!=str(rainy['time'].values[i])[5:7]): 
            print("month_processed:",mon)
            monthly_avg_rain[y,m,:,:]/day
            m+=1
            mon=str(rainy['time'].values[i])[5:7]
            day=0
            if (yr!=str(rainy['time'].values[i])[:4]): 
                print("year_processed:",yr)            
                y+=1
                yr=str(rainy['time'].values[i])[:4]
                m=0
        
        monthly_avg_rain[y,m,:,:]+=rain[i,Bangalore_lon_index[0]:Bangalore_lon_index[1]+1,Bangalore_lat_index[0]:Bangalore_lat_index[1]+1]
        day+=1
    
    
    
    
    
    
    x=[lon(i) for i in range(Bangalore_lon_index[0],Bangalore_lon_index[1]+1)]
    y=[lat(i) for i in range(Bangalore_lat_index[0],Bangalore_lat_index[1]+1)]
    X,Y=np.meshgrid(x,y)
    
    dia=np.array(monthly_avg_rain[6,:,:])
    #First we interpolate the grid points. Then generate a continuous curve through them
    def monthly_interpolated(year,month):
       
        mask=[~np.isnan(monthly_avg_rain[year,month,:,:])]
        
        x_mask = X[mask].reshape(-1)
        y_mask = Y[mask].reshape(-1)
        
        points_grid = np.array([x_mask,y_mask]).T
        values_grid = monthly_avg_rain[year,month,:,:][mask].reshape(-1) 
        
        interp_grid = interpolate.griddata(points_grid, values_grid, (X, Y), method='nearest')
        monthly_avg_rain[year,month,:,:]=interp_grid
        # print(monthly_avg_rain[month,:,:])
        #return(monthly_avg_rain[year,month,:,:])
    
    interpol_yr=[]      
    for y in range(end_yr-start_yr+1):
        for m in range(12):
            monthly_interpolated(y, m)
    '''    
        interpol_month=np.array([monthly_interpolated(y,m) for m in range(12)])
        interpol_yr.append(interpol_month)
    interpol_yr=np.array(interpol_yr)
    '''
    #PLOTTING SECTION
    '''
    
    X1 = np.linspace(min(x), max(x),num=5000)
    Y1 = np.linspace(min(y), max(y),num=5000)
        
    X1,Y1=np.meshgrid(X1,Y1) 
    '''
    net_rain_vol2=np.sum(monthly_avg_rain,(2,3))
    return(net_rain_vol2)
net_rain_vol1=net_rain(1901, 1950)

net_rain_vol2=net_rain(1951,2018)
net_rain_vol2=np.concatenate((net_rain_vol1[:-1,:],net_rain_vol2[:-1,:]),axis=0)

#Fill in the blank for monthly time series in a year
rectified=[]
for m in range(12):
    var_x=[]
    var_y=[]
    failed_yr=[]
    for y in range(net_rain_vol2.shape[0]):
        if net_rain_vol2[y,m]!=0.00000000e+00:
            var_x.append(y)
            var_y.append(net_rain_vol2[y,m])
        else:
            failed_yr.append(y)
    var_x=np.array(var_x)
    var_y=np.array(var_y)
    print(var_x,var_y)
    for i in range(1,4):
        f = interpolate.InterpolatedUnivariateSpline(var_x, var_y, k=1)
        rectified.append([])
        rectified[-1]=net_rain_vol2
        for i in failed_yr:
            rectified[-1][i,m]=f(i)
rectified_normalised=[]
for rec_rain in rectified:
    rectified_normalised.append(np.array([rec_rain[y,:]/np.sum(rec_rain[y,:]) for y in range(rec_rain.shape[0])]))
print(rectified_normalised)
#nomalised_rain=np.array([net_rain_vol2[y,:]/np.sum(net_rain_vol2[y,:]) for y in range(net_rain_vol2.shape[0])])
