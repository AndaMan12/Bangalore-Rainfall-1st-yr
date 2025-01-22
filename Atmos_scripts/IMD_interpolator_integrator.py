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
month=4
start_yr =1998
end_yr=2018
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
monthly_avg_rain=np.zeros(shape=(days//30,Bangalore_lon_index[1]-Bangalore_lon_index[0]+1, 
                                 Bangalore_lat_index[1]-Bangalore_lat_index[0]+1))
#print (monthly_avg_rain.shape)
m=0
mon='01'
for i in range(days):
   
    if (mon!=str(rainy['time'].values[i])[5:7]): 
        print("month_processed:",m)
        monthly_avg_rain[m,:,:]/30
        m+=1
        mon=str(rainy['time'].values[i])[5:7]
    monthly_avg_rain[m,:,:]+=rain[i,Bangalore_lon_index[0]:Bangalore_lon_index[1]+1,Bangalore_lat_index[0]:Bangalore_lat_index[1]+1]

x=[lon(i) for i in range(Bangalore_lon_index[0],Bangalore_lon_index[1]+1)]
y=[lat(i) for i in range(Bangalore_lat_index[0],Bangalore_lat_index[1]+1)]
X,Y=np.meshgrid(x,y)
def monthly_rain(month,point):
    x_ind=index_lon(point[0])-Bangalore_lon_index[0]
    y_ind=index_lat(point[1])-Bangalore_lat_index[0]
    
    return(30*monthly_avg_rain[month,x_ind,y_ind])
dia=np.array(monthly_avg_rain[6,:,:])
#First we interpolate the grid points. Then generate a continuous curve through them
def monthly_interpolated(month):
   
    mask=[~np.isnan(monthly_avg_rain[month,:,:])]
    
    x_mask = X[mask].reshape(-1)
    y_mask = Y[mask].reshape(-1)
    
    points_grid = np.array([x_mask,y_mask]).T
    values_grid = monthly_avg_rain[month,:,:][mask].reshape(-1) 
    
    interp_grid = interpolate.griddata(points_grid, values_grid, (X, Y), method='nearest')
    monthly_avg_rain[month,:,:]=interp_grid
   # print(monthly_avg_rain[month,:,:])
    
    points=[]
    values = []
    
    for px in x:
        for py in y:
            points.append([px,py])
            values.append(monthly_rain(month,[px,py])) 
            
    interp = interpolate.CloughTocher2DInterpolator(points,values ) 
    return(interp)
      

  
#PLOTTING SECTION

X1 = np.linspace(min(x), max(x),num=5000)
Y1 = np.linspace(min(y), max(y),num=5000)
    
X1,Y1=np.meshgrid(X1,Y1) 
net_rain_vol=[]
'''
for month in range(m):
    
    
    INT=integrate.nquad(monthly_interpolated(month),ranges=([77,78],[12.5,13.5]))
        #print(INT)
    net_rain_vol.append(INT[0])
    print('done ',month)
'''   
'''
fig = plt.figure(figsize=(8,7))
ax=plt.axes(projection='3d')
Z=monthly_interpolated(month)(X1,Y1)    
#im1 = ax.scatter3D(X,Y,30*monthly_avg_rain[5,:,:],color='b')
im2 = ax.plot_surface(X1,Y1,Z,color='r')
plt.show()
'''


fig = plt.figure(figsize=(8,7))
ax=plt.axes(projection='3d')
Z=monthly_interpolated(6)(X1,Y1)    
im1 = ax.scatter3D(Y,X,monthly_avg_rain[246,:,:],color='b')
#im2 = ax.plot_surface(X1,Y1,Z,color='r')
plt.show()
'''
print(net_rain_vol)
plt.plot([i+1 for i in range(m)],net_rain_vol)
'''