import numpy as np
import matplotlib.pyplot as plt
import input as I



def fresnelCoefficients_TE(theta1, theta2, n1, n2):
    tmp1 = n1 * np.cos(theta1)
    tmp2 = n2 * np.cos(theta2)
    Efield_mod = 0
    r_te = (tmp1 - tmp2) / (tmp1 + tmp2)
    t_te = 2 * tmp1 / (tmp1 + tmp2)
    if Efield_mod == 1:
        Efield_conv = np.sqrt(np.cos(theta2)*np.abs(n2)/(np.cos(theta1)*np.abs(n1)))
        return [np.abs(r_te), np.abs(t_te)*Efield_conv] 
        
    else:
        return [r_te, t_te] 
    

def fresnelCoefficients_TM(theta1, theta2, n1, n2):
    tmp1 = n1 * np.cos(theta2)
    tmp2 = n2 * np.cos(theta1)
    r_te = (tmp1 - tmp2) / (tmp1 + tmp2)
    t_te = 2 * n1 * np.cos(theta1) / (tmp1 + tmp2)
    return [r_te, t_te]



def multiLayerTransferMatrix(theta_in, t, er, frequency, pol):
# Transfer matrix for multilayer slab consisting of N finite thickness
# layers, for N+2 permittivities (for the initial region, the N layers, and
# the final region).
#
# The transfer matrix is refered to the interface between layer 0 (the
# incident region) and layer 1, and the interface between layer N and layer
# N+1 (the final region). The thickness values of layers 0 and N+1 are
# hence assumed to be zero (and not used).
#
    # Wavelength
    lambd = 299792458/frequency
    k0 = 2*np.pi/lambd

    # Transfer matrix
    A = np.identity(2)
    for n in range (1,np.size(er)):
        # Angle in outgoing layer (5.4)
        theta_out = np.arcsin(np.sqrt(er[n-1]/er[n])*np.sin(theta_in))
        if np.isnan(theta_out):
            theta_out = np.pi/2

    # Propagation constant normal to boundary ok for plane wave for ray?!
        k_n = k0 * np.sqrt(er[n]) * np.cos(theta_out) #SHOULD BE A *COS

    # Fresnel reflection and transmission coefficients
        if pol == 'te':
            [R, T] = fresnelCoefficients_TE(theta_in, theta_out, np.sqrt(er[n-1]), np.sqrt(er[n]))
        else:
            [R, T] = fresnelCoefficients_TM(theta_in, theta_out, np.sqrt(er[n-1]), np.sqrt(er[n]))
        
    # Transfer matrix
        e = 1j * k_n * t[n]
        A = 1/T  * np.matmul(A, [ [np.exp(e), R*np.exp(-e)],  [R*np.exp(e),  np.exp(-e)]])

    # Update angle
        theta_in = theta_out
    return A



theta_in = np.deg2rad(0)
t =  6e-3
epsr = 4
tand = 0
er = epsr*(1-1j*tand)

def getT_coef(incidentAngle, layerThickness_in, complexRelativePermittivity, frequency):
    layerThickness = [0, layerThickness_in, 0]
    incidentAngle = np.pad(incidentAngle, (1,1), 'edge')

    er = [1, complexRelativePermittivity, 1]
    # Transfer matrix model
    rTE1 = np.ones([np.size(incidentAngle),1],dtype=np.complex128)
    tTE1 = np.ones([np.size(incidentAngle),1], dtype=np.complex128)
    rTM1 = np.ones([np.size(incidentAngle),1],dtype=np.complex128)
    tTM1 = np.ones([np.size(incidentAngle),1], dtype=np.complex128)
    A = multiLayerTransferMatrix(incidentAngle, layerThickness, er, frequency, 'te')
    rTE1 = A[1][0]/A[0][0]
    tTE1 = 1/A[0][0]    #*np.exp(-1j*k_0*distance*np.sin(abs(incidenceAngleRadians1))) #we add phase compensation because the model is for plane waves
    A = multiLayerTransferMatrix(incidentAngle, layerThickness, er, frequency, 'tm')
    rTM1 = A[1,0]/A[0,0]
    tTM1 = 1/A[0,0]
    print('transmitted ' + str(np.abs(tTE1)**2*100))
    print('reflected ' + str(np.abs(rTE1)**2*100))
    print('abs ' + str(100 - np.abs(rTE1)**2*100 - np.abs(tTE1)**2*100))
    return tTE1

trans = getT_coef(theta_in, t, er, 25e9)

# multiLayerTransferMatrix(theta_in, t, er, 25e6, 'te'  )