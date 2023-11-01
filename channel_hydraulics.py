#Depth optimization module
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from scipy.optimize import minimize

class ChannelSection():
    def __init__(self,  width: np.array, hzslope = 0):
        self.width = width
        self.hzslope = hzslope

    def updateArea(self, update_depth: np.array):
        return self.width * update_depth + self.hzslope * pow(update_depth,2)
    
    def updateWPerimeter(self, update_depth: np.array):
        return self.width + 2* update_depth * np.sqrt(1+pow(self.hzslope,2))
    
    def updateCentroidHeight(self, update_depth: np.array):
        return update_depth/2
    

def strickler_formula(d50):
    return pow(d50,1/6)/21

def critical_depth(Q,XSection):
    q= Q/XSection.width
    return (pow(pow(q,2)/9.81,1/3))

def critical_depth_allsec(Q,XSection: ChannelSection,initial_guess = 1,max_iter = 500, tol = 1e-18):
    def Fobj(depth:float):
        return abs(((Q/((XSection.width * depth + XSection.hzslope * pow(depth,2))*pow((9.81*depth),1/2))) - 1))

    def constraint1(depth:float):
        return depth-0.00001
    
    constraints = {'type':'ineq','fun':constraint1}
    options={'maxiter':max_iter}

    result = minimize(Fobj, initial_guess,constraints= constraints,method='SLSQP',options= options,tol=tol)
    print(f"CRITICAL DEPTH: {result.message}")
    return result.x


def make_grid(length_of_channel,del_X):
    return np.append(np.arange(0,length_of_channel,del_X,dtype= float),length_of_channel)

def assign_slope(slope_breaks, long_slopes, long_grid):
    slope_array = np.full_like(long_grid,0)
    for n_break in range(len(slope_breaks)-1):
        mask = (long_grid <= slope_breaks[n_break]) & (long_grid >= slope_breaks[n_break+1])
        slope_array[mask] = long_slopes[n_break]
    return slope_array

def normal_depth_calculator(Q,width, slope, n, hc,max_iter = 500, tol = 1e-18):
    def Fobj(depth):
        return abs(Q- (1/n)*((pow(width,5/3)*pow(depth,5/3))/(pow(width+2*depth,2/3)))*pow(slope,1/2))
    
    def constaint1(depth):
        return depth-0.00001
    
    initial_guess = hc
    constraint = {'type':'ineq','fun':constaint1}
    options = {'maxiter':max_iter}
    result = minimize(Fobj,initial_guess, constraints=constraint,method= 'SLSQP',options= options, tol= tol)
    print(f"NORMAL DEPTH [{slope}]: {result.message}")
    return result.x

def assign_normal_depth(Q, width,n, hc, slope_breaks,long_slopes, long_grid):
    normal_depth_array = np.full_like(long_grid,0)
    for n_break in range(len(slope_breaks)-1):
        normal_depth = normal_depth_calculator(Q,width, long_slopes[n_break],n,hc)
        mask = (long_grid <= slope_breaks[n_break]) & (long_grid >= slope_breaks[n_break+1])
        normal_depth_array[mask] = normal_depth[0]
    return normal_depth_array

def elevation_calculator(slope_breaks,long_slopes,long_grid,ds_elev):
    all_elev = []
    for n_break in range(len(slope_breaks)-1):
        elev_betn_breaks = []
        if n_break != 0:
            ds_elev = all_elev[n_break-1][0]

        for i in range(len(long_grid)):
            if long_grid[i] <= slope_breaks[n_break] and long_grid[i] >= slope_breaks[n_break+1]:
                elev_1 = ds_elev + long_slopes[n_break] * (slope_breaks[n_break]-long_grid[i])
                elev_betn_breaks.append(elev_1)
        all_elev.append(elev_betn_breaks)
    return all_elev

def Exner1D(long_grids:np.array, del_X,del_T, qs:np.array,  zo: np.array,qo = 0 ,lam = 0):
#Exner equation 1D
    nx = len(long_grids)
    zn = []
    for j in range(nx):
        if j == 0:
            znew = zo[j]+ (qo*del_T -(del_T/del_X)*(qs[j+1]-qs[j]))/(1-lam)
        elif j == nx-1:
            znew = zo[j]+ (qo*del_T-(del_T/del_X)*(qs[j]-qs[j-1]))/(1-lam)
        else:
            znew = zo[j]+ (qo*del_T-(del_T/(2*del_X))*(qs[j+1]-qs[j-1]))/(1-lam)
        zn.append(znew)
    return np.array(zn)


class BackwaterAnalysis():
    def __init__(self, length_of_channel:float,
                        del_X:float,
                        ds_elev: float,
                        slope_breaks: np.array,
                        long_slopes: np.array,
                        Q: float,
                        d50: float,
                        width: float,
                        initial_depth: float,
                        hzslope:float,
                        boundary: np.array,
                        spgr = 2.65,
                        subcritical = True):
        self.Xsec = ChannelSection(width = width, hzslope=hzslope)

        self.subcritical = subcritical
        self.boundary = boundary
        self.initial_depth = initial_depth
        self.d50 = d50
        self.spgr = spgr
        
        self.n = strickler_formula(d50=d50)
        self.hc = critical_depth(Q, XSection=self.Xsec)

        self.long_grid = make_grid(length_of_channel= length_of_channel, del_X= del_X)
        self.discharge = np.full_like(self.long_grid,Q)
        self.roughness = np.full_like(self.long_grid,self.n)
        self.hcritical = np.full_like(self.long_grid,self.hc)
        self.slopes = assign_slope(slope_breaks=slope_breaks,long_slopes= long_slopes, long_grid= self.long_grid)
        self.width = np.full_like(self.long_grid, width)

        self.normal_depth = assign_normal_depth(Q=Q,width=width,n=self.n,hc=self.hc,slope_breaks=slope_breaks,
                                              long_slopes=long_slopes,long_grid=self.long_grid)


        self.bed_elevation = elevation_calculator(slope_breaks= slope_breaks,long_slopes= long_slopes, 
                                                  long_grid= self.long_grid, ds_elev= ds_elev) # return 2d array with each array representing elevation between two slope breaks
        self.bed_elevation = self.bed_elevation[::-1]# reversing the order of array
        self.bed_elevation = np.flip(np.unique([element for sublist in self.bed_elevation for element in sublist])) #flatten the 2d array, remove overlap and make series from us to ds

    def depth_solver(self,max_iter = 500,tol = 1e-10,updated_bed_elevation = None, exner = False):
        if self.subcritical:
            if self.boundary[0] == 'default':
                d_bc = self.hc
            else:
                d_bc = self.boundary[0]
            
        else:
            if self.boundary[0] == 'default':
                u_bc = self.hc
            else:
                u_bc = self.boundary[0]

        def objective_function(flow_depth: np.array):
            if self.subcritical:
                flow_depth = np.append(flow_depth,d_bc)
            else:
                flow_depth = np.insert(flow_depth,0,u_bc)

            area = self.Xsec.updateArea(flow_depth)
            wettedPerimeter = self.Xsec.updateWPerimeter(flow_depth)
            Rh = area/wettedPerimeter
            velocity = self.discharge/area
            kinetic_head = pow(velocity,2)/ (2*9.81)
            specific_energy = flow_depth + kinetic_head
            energy_slope = pow(self.roughness,2) * pow(self.discharge,2) / (pow(area,2)*pow(Rh,4/3))
            if exner:
                total_head = specific_energy + updated_bed_elevation
            else:
                total_head = specific_energy + self.bed_elevation

            sq_tot_cost = []
            for i in range(len(total_head)-1):
                cost = total_head[i]-total_head[i+1]- (self.long_grid[i+1] - self.long_grid[i]) * (energy_slope[i]+energy_slope[i+1])/2
                sq_cost = pow(cost,2)
                sq_tot_cost.append(sq_cost)
            if self.subcritical:
                error = sum(sq_tot_cost[:])
            else:
                error = sum(sq_tot_cost[1:])
            return error

        def constraints_fn1(flow_depth):
            if self.subcritical:
                flow_depth= np.append(flow_depth,d_bc)
            else:
                flow_depth = np.insert(flow_depth,0,u_bc)
            return flow_depth-0.00001

        def constraints_fn2(flow_depth):
            if self.subcritical:
                flow_depth = np.append(flow_depth,d_bc)
            else:
                flow_depth = np.insert(flow_depth,0,u_bc)
                
            velocity = self.discharge / self.Xsec.updateArea(flow_depth)
            Fr = velocity / pow(flow_depth * 9.81, 1/2)

            if self.subcritical:
                error = 1-Fr
            else:
                error = Fr-1
            return error
        
        initial_depths = np.full_like(self.long_grid,self.initial_depth)
        if self.subcritical:
            initial_depths = initial_depths[:-1]
        else:
            initial_depths = initial_depths[1:]

        bounds = [(0.00001, None)]  # flow_depth should be >= 0.00001

        constraints = [{'type':'ineq','fun':constraints_fn1},
                    {'type':'ineq','fun': constraints_fn2}]

        options = {'maxiter': max_iter}

        result = minimize(objective_function,initial_depths,bounds= bounds, constraints= constraints, method='SLSQP',tol=tol, options= options)

        print(f"FLOW DEPTH: {result.message}")
        if self.subcritical:
            final_depth = np.append(result.x,d_bc)
        else:
            final_depth = np.insert(result.x,0,u_bc)
        
        return final_depth

    def momentum_solver(self, flow_depth:np.array):
        return 9.81* self.Xsec.updateCentroidHeight(flow_depth)* self.Xsec.updateArea(flow_depth) + pow(self.discharge,2)/self.Xsec.updateArea(flow_depth)
    
    def bed_shear_stress(self,flow_depth:np.array):
        Rh = self.Xsec.updateArea(flow_depth)/self.Xsec.updateWPerimeter(flow_depth)
        Sf = pow(self.roughness,2) *pow(self.discharge,2) / (pow(self.Xsec.updateArea(flow_depth),2)
                                                              *pow(self.Xsec.updateArea(flow_depth)/self.Xsec.updateWPerimeter(flow_depth),4/3))
        return 9810 * Rh * Sf
    
    def shields_parameter(self,bed_shear: np.array):
        return bed_shear/(9810* (self.spgr-1)*self.d50)
    
    def Einstein_parameter(self,shields_parameter:np.array):
        return 40* pow(shields_parameter,1.5)
    
    def sediment_transport_rate(self,q_star:np.array):
        return q_star* math.sqrt(9.81*(self.spgr-1)*self.d50)*self.d50
    
    def display_plot(self,values:np.array,combined= False):
        plt.plot(self.long_grid, self.bed_elevation,label = 'z', color = 'black')
        plt.plot(self.long_grid,self.bed_elevation + self.hcritical, label = 'z+hc',color = 'red')
        plt.plot(self.long_grid,self.bed_elevation+self.normal_depth,color= 'green',label = 'z+hn')
        plt.plot(self.long_grid, self.bed_elevation + values,label ='z+h',color = 'blue')
        plt.legend()
        plt.grid()
        plt.xlabel('Chainage (meters)')
        plt.ylabel('Depth (meters)')
        if combined:
            plt.title('Hydraulic jump profile')
        else:
            plt.title('Sub-critical profile' if self.subcritical else 'Super-critical profile')
        
    
 




