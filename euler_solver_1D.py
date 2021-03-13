#Importing Numpy, Matplotlib, and Math
import numpy as np
import matplotlib.pyplot as plt
import math

#Setting required variables and initial conditions
n       = 100
x_min   = 0.0
x_max   = 10.0
x_start = 5.0
gamma   = 1.4
t       = 0.035
R       = 287.05

CFL     = 0.5

rho_L = 1.598
u_L   = -383.64
p_L   = 91.88*1000.0

rho_R = 2.787
u_R   = -216.97
p_R   = 200.0*1000.0 

#Setting up delta x and current t
delta_x = (x_max-x_min)/n
delta_t = 0.0
current_t = 0.0

#Creating arrays for v, p, rho, m, a, and T
u   = np.zeros(n)
p   = np.zeros(n)
rho = np.zeros(n)
m   = np.zeros(n)
a   = np.zeros(n)
T   = np.zeros(n)

#Creating arrays for U, Utilda, F, Ftilda, and x
U      = np.zeros((3,n))
Utilda = np.zeros((3,n))
F      = np.zeros((3,n))
Ftilda = np.zeros((3,n))
x      = np.zeros(n)

max_lambda = 0.0

def find_U(rho, u, p):
    U = np.zeros((3,1))
    U[0, 0] = rho
    U[1, 0] = rho*u
    U[2, 0] = p/(gamma-1.0)+(rho*u**2.0/2.0)
    return U

def find_F(rho, u, p):
    F = np.zeros((3,1))
    F[0, 0] = rho*u
    F[1, 0] = rho*u**2.0+p
    F[2, 0] = u*((gamma*p)/(gamma-1.0)+rho*u**2.0/2.0)
    return F

def find_h(rho, u, p):
    return gamma*p/((gamma-1.0)*rho) + u**2.0/2.0

def find_a(rho, p):
    return math.sqrt(gamma*p/rho)

def find_rho_p_u_a(U):
    rho = U[0]
    u   = U[1]/U[0]
    p   = (U[2]-(rho*u**2.0)/2.0)*(gamma-1.0)
    a   = find_a(rho, p)
    return rho, u, p, a

def wavelambda_noFix(u_hat, a_hat):
    wave_lambda = np.zeros((3,3))
    wave_lambda[0, 0] = abs(u_hat - a_hat)
    wave_lambda[1, 1] = abs(u_hat)
    wave_lambda[2, 2] = abs(u_hat + a_hat)
    return wave_lambda

def wavelambda_Fix(u_L, a_L, u_hat, a_hat, u_R, a_R):
    wave_lambda = np.zeros((3,3))

    #Lambda Minus [0,0]
    LM_L = u_L - a_L
    LM_hat = u_hat - a_hat
    LM_R = u_R - a_R

    deltaM = max(0.0, 4.0*(LM_R-LM_L))

    if abs(LM_hat) > deltaM/2.0:
        wave_lambda[0,0] = abs(LM_hat)
    else:
        wave_lambda[0,0] = LM_hat**2.0/deltaM + deltaM/4.0

    #Lambda Hat [1,1]
    wave_lambda[1,1] = abs(u_hat)

    #Lambda Plus [2,2]
    LP_L  = u_L + a_L
    LP_hat  = u_hat + a_hat
    LP_R  = u_R + a_R

    deltaP = max(0.0, 4.0*(LP_R-LP_L))

    if abs(LP_hat) > deltaP/2.0:
        wave_lambda[2,2] = abs(LP_hat)
    else:
        wave_lambda[2,2] = LP_hat**2.0/deltaP + deltaP/4.0

    return wave_lambda

def roe_averageStates(rho_L, u_L, p_L, a_L, rho_R, u_R, p_R, a_R):
    h_L   = find_h(rho_L, u_L, p_L)
    h_R   = find_h(rho_R, u_R, p_R)
    rho_hat = math.sqrt(rho_L*rho_R)
    u_hat = (u_L*rho_L**0.5 + u_R*rho_R**0.5)/(rho_L**0.5+rho_R**0.5)
    h_hat = (h_L*rho_L**0.5 + h_R*rho_R**0.5)/(rho_L**0.5+rho_R**0.5)
    p_hat = ((h_hat-u_hat**2.0/2.0)*(gamma-1.0)*rho_hat)/gamma
    a_hat = find_a(rho_hat, p_hat)

    return rho_hat, u_hat, h_hat, p_hat, a_hat

def plot_fig(fig_number, x_value, y_value, x_name, y_name):
    fig = plt.figure(fig_number)
    plt.plot(x_value, y_value)
    plt.xlabel(x_name)
    plt.ylabel(y_name)
    plt.savefig(y_name)
    fig.show()

#Initializing cell initial conditions
for i in range(n):
    x[i] = x_min + (i+0.5)*delta_x

    if x[i] < x_start:
        U_here  = find_U(rho_L, u_L, p_L) 
        U[0, i] = U_here[0, 0]
        U[1, i] = U_here[1, 0]
        U[2, i] = U_here[2, 0]
    else:
        U_here  = find_U(rho_R, u_R, p_R)
        U[0, i] = U_here[0, 0]
        U[1, i] = U_here[1, 0]
        U[2, i] = U_here[2, 0]

def hlle(rho_L, u_L, p_L, a_L, rho_R, u_R, p_R, a_R):
    rho_hat, u_hat, h_hat, p_hat, a_hat = roe_averageStates(rho_L, u_L, p_L, a_L, rho_R, u_R, p_R, a_R)

    LM = min(u_L - a_L, u_hat - a_hat)
    LP = max(u_R + a_R, u_hat + a_hat)

    if LM > 0.0:
        Fl = find_F(rho_L, u_L, p_L)
        return Fl
    elif LP < 0.0:
        Fr = find_F(rho_R, u_R, p_R)
        return Fr
    else:
        Ul_mat = find_U(rho_L, u_L, p_L)
        Ur_mat = find_U(rho_R, u_R, p_R)

        Fl = find_F(rho_L, u_L, p_L)
        Fr = find_F(rho_R, u_R, p_R)

        Fm = (LP*Fl-LM*Fr)/(LP-LM)+LP*LM*(Ur_mat-Ul_mat)/(LP-LM)
        return Fm

def vanleer(U, i):
    psi = np.zeros(3)
    a   = np.zeros(3)
    b   = np.zeros(3)
    epsilon = 1e-12

    if i == 0: 
        a = (U[:, i] - U[:, i])/delta_x 
        b = (U[:, i+1] - U[:, i])/delta_x 
    elif i == n-1:
        a = (U[:, i] - U[:, i-1])/delta_x 
        b = (U[:, i] - U[:, i])/delta_x  
    else:
        a = (U[:, i] - U[:, i-1])/delta_x 
        b = (U[:, i+1] - U[:, i])/delta_x 

    if a[0] + b[0] > 0.0:
        psi[0] = (abs(a[0]*b[0])+a[0]*b[0])/(a[0]+b[0]+epsilon)
    else:
        psi[0] = (abs(a[0]*b[0])+a[0]*b[0])/(a[0]+b[0]-epsilon)

    if a[1] + b[1] > 0.0:
        psi[1] = (abs(a[1]*b[1])+a[1]*b[1])/(a[1]+b[1]+epsilon)
    else:
        psi[1] = (abs(a[1]*b[1])+a[1]*b[1])/(a[1]+b[1]-epsilon)

    if a[2] + b[2] > 0.0:
        psi[2] = (abs(a[2]*b[2])+a[2]*b[2])/(a[2]+b[2]+epsilon)
    else:
        psi[2] = (abs(a[2]*b[2])+a[2]*b[2])/(a[2]+b[2]-epsilon)

    return psi

#Marching in time (Predictor Corrector)
while current_t < t:
    F.fill(0.0)
    delta_t = 1e12
    max_lambda = 1e-12

    for i in range(0, n):
        rho_i, u_i, p_i, a_i = find_rho_p_u_a(U[:, i])
        max_lambda = max(abs(u_i)+a_i, max_lambda)

    delta_t = min(CFL*delta_x/max_lambda, delta_t)

    if current_t+delta_t > t:
      delta_t = t - current_t

    #Evaluating Fluxes and Utilda
    for i in range(0, n+1):
        iL, iR = 0,0
        if i == 0:
            iL, iR = i, i
            rho_L, u_L, p_L, a_L = find_rho_p_u_a(U[:, iL])
            rho_R, u_R, p_R, a_R = find_rho_p_u_a(U[:, iR]-0.5*delta_x*vanleer(U, i))
        elif i == n:
            iL, iR = i-1, i-1
            rho_L, u_L, p_L, a_L = find_rho_p_u_a(U[:, iL]+0.5*delta_x*vanleer(U, i-1))
            rho_R, u_R, p_R, a_R = find_rho_p_u_a(U[:, iR])
        else:
            iL, iR = i-1, i
            rho_L, u_L, p_L, a_L = find_rho_p_u_a(U[:, iL]+0.5*delta_x*vanleer(U, i-1))
            rho_R, u_R, p_R, a_R = find_rho_p_u_a(U[:, iR]-0.5*delta_x*vanleer(U, i))

        flux = hlle(rho_L, u_L, p_L, a_L, rho_R, u_R, p_R, a_R)

        if i == 0:
            F[0, i] += flux[0, 0]
            F[1, i] += flux[1, 0]
            F[2, i] += flux[2, 0]
        elif i == n:
            F[0, i-1] -= flux[0, 0]
            F[1, i-1] -= flux[1, 0]
            F[2, i-1] -= flux[2, 0]
        else:
            F[0, i] += flux[0, 0]
            F[1, i] += flux[1, 0]
            F[2, i] += flux[2, 0]

            F[0, i-1] -= flux[0, 0]
            F[1, i-1] -= flux[1, 0]
            F[2, i-1] -= flux[2, 0]

    Utilda = U + (delta_t/delta_x)*F

    #Evaluating tilda fluxes and U
    for i in range(0, n+1):
        iL, iR = 0,0
        if i == 0:
            iL, iR = i, i
            rho_L, u_L, p_L, a_L = find_rho_p_u_a(Utilda[:, iL])
            rho_R, u_R, p_R, a_R = find_rho_p_u_a(Utilda[:, iR]-0.5*delta_x*vanleer(Utilda, i))
        elif i == n:
            iL, iR = i-1, i-1
            rho_L, u_L, p_L, a_L = find_rho_p_u_a(Utilda[:, iL]+0.5*delta_x*vanleer(Utilda, i-1))
            rho_R, u_R, p_R, a_R = find_rho_p_u_a(Utilda[:, iR])
        else:
            iL, iR = i-1, i
            rho_L, u_L, p_L, a_L = find_rho_p_u_a(Utilda[:, iL]+0.5*delta_x*vanleer(Utilda, i-1))
            rho_R, u_R, p_R, a_R = find_rho_p_u_a(Utilda[:, iR]-0.5*delta_x*vanleer(Utilda, i))

        flux = hlle(rho_L, u_L, p_L, a_L, rho_R, u_R, p_R, a_R)

        if i == 0:
            F[0, i] += flux[0, 0]
            F[1, i] += flux[1, 0]
            F[2, i] += flux[2, 0]
        elif i == n:
            F[0, i-1] -= flux[0, 0]
            F[1, i-1] -= flux[1, 0]
            F[2, i-1] -= flux[2, 0]
        else:
            F[0, i] += flux[0, 0]
            F[1, i] += flux[1, 0]
            F[2, i] += flux[2, 0]

            F[0, i-1] -= flux[0, 0]
            F[1, i-1] -= flux[1, 0]
            F[2, i-1] -= flux[2, 0]

    #Predictor Corrector
    U += delta_t/(2.0*delta_x)*F
    current_t += delta_t #evaluating time

for i in range(n):
    rho[i], u[i], p[i], a[i] = find_rho_p_u_a(U[:, i])
    m[i]   = abs(u[i]/a[i])
    T[i]   = p[i]/(R*rho[i])

plot_fig(1, x, u,   "X", "Speed")
plot_fig(2, x, p,   "X", "Pressure")
plot_fig(3, x, rho, "X", "Density")
plot_fig(4, x, m,   "X", "Mach Number")
plot_fig(5, x, T,   "X", "Temperature")

input()
