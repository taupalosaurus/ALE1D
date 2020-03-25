import matplotlib
import matplotlib.pyplot as plt
import numpy as np


##### Define analytic functions

def f1(x):
    return 2*np.ones(x.shape);

def f2(x):
    return 3*x+2;

##### Displacement
def dx(x, alpha, t):
    return alpha * np.sin(np.pi/2*t)*np.sin(np.pi*x)


def move_mesh_and_rho(nver, f, alpha, plot=False): 

    ##### Initial mesh
    
    x = np.linspace(0,1,nver)
    #one = 1.5*np.ones(x.shape)
    rho = f(x)
    
    ##### Finite Volume cells
    
    xip1s2 = np.zeros(x.shape)
    xim1s2 = np.zeros(x.shape)
    
    xim1s2[1:] = 0.5*(x[1:] + x[0:-1])
    xim1s2[0] = x[0] - 0.5*(x[1]-x[0])
    
    xip1s2[0:-1] = 0.5*(x[0:-1] + x[1:])
    xip1s2[-1] = x[-1] + 0.5*(x[-1]-x[-2])

    #print(x)
    #print(xim1s2)
    #print(xip1s2)
    
    cell_area = xip1s2 - xim1s2
    #print(cell_area)
    if cell_area.min() < 0:
        print("BIGGG PROBLEM: nver: %d  alpha: %1.2f" %(nver, alpha))
    
    rhoip1s2 = f(xip1s2) 
    rhoim1s2 = f(xim1s2)
    
    ##### New mesh
    
    dt = 1.
    x_new = x +  dx(x, alpha, dt)
    
    xip1s2_new = np.zeros(x_new.shape)
    xim1s2_new = np.zeros(x_new.shape)
    
    xim1s2_new[1:] = 0.5*(x_new[1:] + x_new[0:-1])
    xim1s2_new[0] = x_new[0] - 0.5*(x_new[1]-x_new[0])
    
    xip1s2_new[0:-1] = 0.5*(x_new[0:-1] + x_new[1:])
    xip1s2_new[-1] = x_new[-1] + 0.5*(x_new[-1]-x_new[-2])
    
    cell_area_new = xip1s2_new - xim1s2_new
    if cell_area_new.min() < 0:
        print("BIG PROBLEM: nver: %d  alpha: %1.2f" %(nver, alpha))
    
    vip1s2 = xip1s2_new - xip1s2
    vim1s2 = xim1s2_new - xim1s2
    
    #### Move the function
    
    rho_new = dt/cell_area_new * (cell_area/dt * rho + rhoip1s2 * vip1s2 - rhoim1s2 * vim1s2 )
    
    ##### Compute the error
    
    rho_new_exact = f(x_new)
    
    errL1 = np.linalg.norm(cell_area_new*(rho_new-rho_new_exact), 1)
    errL2 = np.linalg.norm(np.sqrt(cell_area_new)*(rho_new-rho_new_exact), 2)
    errLinf = np.linalg.norm(rho_new-rho_new_exact, np.inf)
    
    #print("## Erreur L1: %1.2e   L2: %1.2e   Linf: %1.2e" % (errL1, errL2, errLinf))
    
    #### Plot the meshes
    
    if plot:
        plt.plot(x, x, '+b', x_new, x, '+r')
        #plt.plot(x, one, '*b', x_new, one, '*r')
        plt.plot(x, rho, '+b', label=r'$t^n$')
        plt.plot(x_new, rho_new, '+r', label=r"$t^{n+1}$")
        plt.title("$f(x) = 3x+2$  and $n = %d$" % (nver))
        
        
        plt.annotate('courbe x', xy=(x[-1], x[-1]),  xycoords='data',
                    xytext=(0, 55), textcoords='offset points',
                    arrowprops=dict(facecolor='black', width=1, headwidth=5, headlength=5, shrink=0.08),
                    horizontalalignment='right', verticalalignment='top',
                    )
        
        plt.annotate('courbe f(x)', xy=(x[-1], rho[-1]),  xycoords='data',
                    xytext=(0, -55), textcoords='offset points',
                    arrowprops=dict(facecolor='black', width=1,headwidth=5, headlength=5, shrink=0.08),
                    horizontalalignment='right', verticalalignment='top',
                    )
        
        plt.legend()
        #plt.savefig("ale1d_n=%d_alpha=%.2f.png"%(nver, alpha))
        plt.show()

    return errL1, errL2, errLinf


def move_mesh_and_rho_small_steps(nver, f, alpha, nsteps, plot=False): 

    ##### Initial mesh
    
    x = np.linspace(0,1,nver)
    #one = 1.5*np.ones(x.shape)
    rho = f(x)
    
    ##### Finite Volume cells
    
    xip1s2 = np.zeros(x.shape)
    xim1s2 = np.zeros(x.shape)
    
    xim1s2[1:] = 0.5*(x[1:] + x[0:-1])
    xim1s2[0] = x[0] - 0.5*(x[1]-x[0])
    
    xip1s2[0:-1] = 0.5*(x[0:-1] + x[1:])
    xip1s2[-1] = x[-1] + 0.5*(x[-1]-x[-2])
    
    cell_area = xip1s2 - xim1s2
    
    rhoip1s2 = f(xip1s2) 
    rhoim1s2 = f(xim1s2)

    dt = 1.
    dsp =  dx(x, alpha, dt)

    i = 0
    while (i < nsteps):

        ##### New mesh

        x_new = x + dsp/nsteps

        xip1s2_new = np.zeros(x_new.shape)
        xim1s2_new = np.zeros(x_new.shape)
        
        xim1s2_new[1:] = 0.5*(x_new[1:] + x_new[0:-1])
        xim1s2_new[0] = x_new[0] - 0.5*(x_new[1]-x_new[0])
    
        xip1s2_new[0:-1] = 0.5*(x_new[0:-1] + x_new[1:])
        xip1s2_new[-1] = x_new[-1] + 0.5*(x_new[-1]-x_new[-2])
        
        cell_area_new = xip1s2_new - xim1s2_new
        
        vip1s2 = xip1s2_new - xip1s2
        vim1s2 = xim1s2_new - xim1s2
    
        #### Move the function
    
        rho_new = dt/cell_area_new * (cell_area/dt * rho + rhoip1s2 * vip1s2 - rhoim1s2 * vim1s2 )

        i += 1
    
    ##### Compute the error
    
    rho_new_exact = f(x_new)
    
    errL1 = np.linalg.norm(cell_area_new*(rho_new-rho_new_exact), 1)
    errL2 = np.linalg.norm(np.sqrt(cell_area_new)*(rho_new-rho_new_exact), 2)
    errLinf = np.linalg.norm(rho_new-rho_new_exact, np.inf)
    
    #print("## Erreur L1: %1.2e   L2: %1.2e   Linf: %1.2e" % (errL1, errL2, errLinf))
    
    #### Plot the meshes
    
    if plot:
        plt.plot(x, x, '+b', x_new, x, '+r')
        #plt.plot(x, one, '*b', x_new, one, '*r')
        plt.plot(x, rho, '+b', label=r'$t^n$')
        plt.plot(x_new, rho_new, '+r', label=r"$t^{n+1}$")
        plt.title("$f(x) = 3x+2$  and $n = %d$" % (nver))
        
        
        plt.annotate('courbe x', xy=(x[-1], x[-1]),  xycoords='data',
                    xytext=(0, 55), textcoords='offset points',
                    arrowprops=dict(facecolor='black', width=1, headwidth=5, headlength=5, shrink=0.08),
                    horizontalalignment='right', verticalalignment='top',
                    )
        
        plt.annotate('courbe f(x)', xy=(x[-1], rho[-1]),  xycoords='data',
                    xytext=(0, -55), textcoords='offset points',
                    arrowprops=dict(facecolor='black', width=1,headwidth=5, headlength=5, shrink=0.08),
                    horizontalalignment='right', verticalalignment='top',
                    )
        
        plt.legend()
        #plt.savefig("ale1d_n=%d_alpha=%.2f.png"%(nver, alpha))
        plt.show()

    return errL1, errL2, errLinf


##########################################################################

nver_list = [11, 21, 51, 101, 201, 501, 1001]
alpha_list = [0.05, 0.1, 0.2, 0.25]

for alpha in alpha_list:
    ErrL1 = []
    ErrL2 = []
    ErrLinf = []
    for nver in nver_list:
        errL1, errL2, errLinf = move_mesh_and_rho(nver, f2, alpha)
        ErrL1.append(errL1)
        ErrL2.append(errL2)
        ErrLinf.append(errLinf)

    slope, intercept = np.polyfit(np.log(nver_list[-3:]), np.log(ErrL2[-3:]), 1)
    plt.loglog(nver_list, ErrL2, '+-', label=r"$\alpha = %1.2f$"%(alpha))

    plt.annotate("$(slope: %1.2f)$"%(slope), xy=(nver_list[-1], ErrL2[-1]),  xycoords='data',
                    xytext=(0, 15), textcoords='offset points',
                    horizontalalignment='right', verticalalignment='top', rotation=0
                    )

#plt.loglog(nver_list, np.array(nver_list)*0.12, 'k-',label="$O(1)$")
#plt.annotate("$(slope: %1.2f)$"%(1), xy=(nver_list[-1], 0.12*nver_list[-1]),  xycoords='data',
#                    xytext=(0, 15), textcoords='offset points',
#                    horizontalalignment='right', verticalalignment='top', rotation=15
#                    )

plt.title("$L_2$ error for $dx  = \alpha * \sin(\pi x)$ and $f(x) = 3x+2$")
plt.xlabel("Number of vertices")
plt.ylabel("$L_2$ error")
plt.legend()
plt.savefig("ale1d_errorL2.png")
plt.show()

###########################################################################

nsteps = 50

for alpha in alpha_list:
    ErrL1 = []
    ErrL2 = []
    ErrLinf = []
    for nver in nver_list:
        errL1, errL2, errLinf = move_mesh_and_rho_small_steps(nver, f2, alpha, nsteps)
        ErrL1.append(errL1)
        ErrL2.append(errL2)
        ErrLinf.append(errLinf)

    slope, intercept = np.polyfit(np.log(nver_list[-3:]), np.log(ErrL2[-3:]), 1)
    plt.loglog(nver_list, ErrL2, '+-', label=r"$\alpha = %1.2f$"%(alpha))

    plt.annotate("$(slope: %1.2f)$"%(slope), xy=(nver_list[-1], ErrL2[-1]),  xycoords='data',
                    xytext=(0, 15), textcoords='offset points',
                    horizontalalignment='right', verticalalignment='top', rotation=0
                    )

#plt.loglog(nver_list, np.array(nver_list)*0.12, 'k-',label="$O(1)$")
#plt.annotate("$(slope: %1.2f)$"%(1), xy=(nver_list[-1], 0.12*nver_list[-1]),  xycoords='data',
#                    xytext=(0, 15), textcoords='offset points',
#                    horizontalalignment='right', verticalalignment='top', rotation=15
#                    )

plt.title("$L_2$ error for $dx  = \alpha * \sin(\pi x)$ subdivided into %d steps and $f(x) = 3x+2$" %(nsteps))
plt.xlabel("Number of vertices")
plt.ylabel("$L_2$ error")
plt.legend()
plt.savefig("ale1d_%dsteps_errorL2.png" % nsteps)
plt.show()
