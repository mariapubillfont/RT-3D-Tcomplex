import numpy as np

def angles_snell(theta0, n0, n1):
    # cos(theta1) robusto con n complejos: cos^2 = 1 - (n0/n1)^2 sin^2
    s0 = np.sin(theta0)
    cos1 = np.sqrt(1 - (n0/n1)**2 * s0**2)  # rama principal
    cos0 = np.cos(theta0)
    return cos0, cos1

def fresnel_coeffs(nj, nk, cosj, cosk, pol):
    if pol.lower() in ['te','s']:
        r = (nj*cosj - nk*cosk)/(nj*cosj + nk*cosk)
        t = (2*nj*cosj)/(nj*cosj + nk*cosk)
    else:  # TM / p
        r = (nk*cosj - nj*cosk)/(nk*cosj + nj*cosk)
        t = (2*nj*cosj)/(nk*cosj + nj*cosk)
    return r, t

def slab_closed_form(theta0, d, n0, n1, n2, k0, pol='te'):
    # Ángulos y cosenos (0->1 y 1->2 comparten theta1 por planaridad)
    cos0, cos1 = angles_snell(theta0, n0, n1)
    # Para el medio de salida:
    _,  cos2 = angles_snell(theta0, n0, n2)

    # Fase en la losa
    delta = k0 * n1 * d * cos1

    # Fresnel en ambas interfaces
    r01, t01 = fresnel_coeffs(n0, n1, cos0, cos1, pol)
    r12, t12 = fresnel_coeffs(n1, n2, cos1, cos2, pol)

    # Closed form (campo)
    denom = 1 + r01*r12*np.exp(-2j*delta)
    T = (t01*t12*np.exp(-1j*delta)) / denom
    R = (r01 + r12*np.exp(-2j*delta)) / denom
    return R, T

def power_from_field(R, T, n0, n2, cos0, cos2):
    # Potencias relativas (no magnéticos)
    Rpow = np.abs(R)**2
    Tpow = np.real(n2*cos2)/np.real(n0*cos0) * np.abs(T)**2
    return Rpow, Tpow

# Ejemplo de uso:
# frecuencia, lambda0, k0
f = 25e9
c = 299792458.0
lambda0 = c/f
k0 = 2*np.pi/lambda0

theta0 = np.deg2rad(45)
d = 6e-3
n0 = 1.0
epsr = 4*(1-1j*0.1)  # ejemplo con pérdidas
n1 = np.sqrt(epsr)
n2 = 1.0

R_te, T_te = slab_closed_form(theta0, d, n0, n1, n2, k0, pol='te')
R_tm, T_tm = slab_closed_form(theta0, d, n0, n1, n2, k0, pol='tm')
print(np.abs(T_te)**2*100)
# Potencia (opcional)
cos0, cos1 = angles_snell(theta0, n0, n1)
_,   cos2  = angles_snell(theta0, n0, n2)
Rpow_te, Tpow_te = power_from_field(R_te, T_te, n0, n2, cos0, cos2)
Rpow_tm, Tpow_tm = power_from_field(R_tm, T_tm, n0, n2, cos0, cos2)
