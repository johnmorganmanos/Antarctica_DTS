# from heat import *
from tqdm import tqdm
import numpy as np
import copy
from pathos.multiprocessing import ProcessPool
import time
import pickle as pkl

#Parameters
restarts = 10
iterations = 5
nt = 100

top_trend = 182 #optical distance (m), This is what we think the start of the long term signal is.
bottom_trend = 312.5 #optical distance (m), This is the bottom of the trend just above the warm bottom anomaly..
start_of_borehole = 162 #optical distance (m), Top of the borehole.


tmax = 2023
tmin = 1923
zmax = bottom_trend - start_of_borehole#150
dTdz = 0 #0.046
nz = 591 #len(mean_tot.sel(x=slice(start_of_borehole,bottom_trend))) -1
nt = nt
diffusivity = 35 * ((tmax - tmin) / (nt)) #originally was just 35
accumulation = 0

# t_surf = np.eye(nt)

# A = np.zeros((nt+1,nz+1))
def heat(t_surf,
         tmax = 1000,
         tmin = 0,
         zmax = 100,
         dTdz = 0.02,
         nz = 39,
         nt = 99,
         alpha = 35,
         accumulation = 0):
    
    '''
    Solves the advection/diffusion equation with mixed temperature/heat flux boundary conditions
    '''
    
    z0=0    
    dz = zmax/(nz+1)
    dt = (tmax - tmin)/nt
    t = np.linspace(tmin,tmax, nt+1)
    z = np.linspace(dz,zmax, nz+1)

    cfl = alpha*dt/(dz**2)
    Azz = np.diag([1+2*cfl] * (nz+1)) + np.diag([-cfl] * (nz),k=1)\
        + np.diag([-cfl] * (nz),k=-1)

    w = - accumulation * np.ones(nz)
    abc = w*dt/(2*dz)
    Az = np.diag(abc,k=1) - np.diag(abc,k=-1)
    Az[0,:] =0
    Az[-1,:]=0
    A = Azz - Az


    # Neumann boundary condition
    A[nz,nz-1] = -2*cfl
    b= np.zeros((nz+1,1))
    b[nz] =  2*cfl*dz * dTdz

    # Initial condition: gradient equal to basal gradient and equal to surface temp.
    U=np.zeros((nz+1,nt+1))
    U[:,0] = t_surf[0] + z*dTdz
    print('here heat')
    for k in range(nt):
        b[0] = cfl*t_surf[k]    #  Dirichlet boundary condition
        print(nt)
        c = U[:,k] + b.flatten()
        U[:,k+1] = np.linalg.solve(A,c)

    return U,t,z

def hastings_met_brute(m, iterations=100):
    
    m_accepted = [m]
    m_not_accepted = []
    m_accepted_all = []
    best_of_restart = []
    predicted_data = []
    temperature_histories = []
    

    for i in range(iterations):
        print('iteration '+str(i))
        m_next_copy = copy.deepcopy(m_accepted[-1])
        val_gauss = np.random.normal(loc=0,scale=1, size=nt)
        change = (np.cumsum(val_gauss) - np.mean(np.cumsum(val_gauss))) 
        this_t_surf = (m_next_copy + change) * .02
    #     if i == 20: this_t_surf = 10*this_t_surf
        temperature_histories.append(this_t_surf)

    #         plt.figure()
    #         plt.plot(range(m_next_copy.shape[0]), this_t_surf)
        print(this_t_surf)
        U,t,z = heat(this_t_surf,
             tmax = tmax,
             tmin = tmin,
             zmax = zmax,
             dTdz = dTdz,
             nz = nz,
             nt = nt,
             alpha = diffusivity,
             accumulation = accumulation)

        print('here')
        season_size = U[:,-1].shape[0] - len(d_obs)
    #         plt.figure()
    # #         plt.plot(z, U[:,-1])
    # #         plt.plot(z[season_size:], U[:,-1][season_size:])
    #         plt.plot(z[season_size:], detrend(U[:,-1][season_size:]))
        now = detrend(U[:,-1][season_size:])
    #         print(now.shape)

    #         plt.figure()
    #         plt.imshow(U, aspect='auto', vmin=-0.5, vmax=0.5)
    #         plt.plot(now)
    #         plt.plot(d_obs)


        r = sum(np.abs(now - d_obs))
        print(r)
        sigma = 1 * np.sqrt(len(this_t_surf))
        #print(sigma)
        L = np.exp(-r**2/sigma**2 / 2) / np.sqrt(2*np.pi) / sigma 
        #print(L)
        #r = (1 / (np.sqrt(2*np.pi) * sigma)) * np.exp(-(sum((d_obs - d_pred[:-1])**2))**2 / 2*(sigma**2))
        #print('r = ' + str(r)+ ', and r_0 = ' + str(r_i))
        alpha = np.random.rand()
        #print(alpha)

        ### Do this to always improve ###
        print('L = '+ str(L))

        if L > alpha:
    #             r_values.append(r)
            m_accepted_all.append(this_t_surf)
            predicted_data.append(now)
            m_accepted.append(this_t_surf)
            #print('r = ' + str(r))
            print('L = '+ str(L)+'  , alpha = '+str(alpha))
            print("accepted")
            Ls.append(L)
        if L < alpha:
            m_not_accepted.append(m_next_copy)
        
        with open('hastings_metro_accepted_histories/' + str(np.random.rand()) + '_accpeted_histories.pkl','wb') as f:
            pkl.dump(m_accepted, f)


random_model_estimates = np.zeros((restarts, nt))

for i in range(restarts):
    val_gauss = np.random.normal(loc=0,scale=0.05, size=nt)
    m_i = np.cumsum(val_gauss) - np.mean(np.cumsum(val_gauss)) 
    random_model_estimates[i] = m_i

random_model_estimates = random_model_estimates 


pool = ProcessPool(nodes=1)
inputs = random_model_estimates
outputs = pool.map(hastings_met_brute, inputs)
print("Input: {}".format(inputs))
print("Output: {}".format(outputs))