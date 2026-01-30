
from scipy.spatial import Delaunay
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import plots as plot
import numpy as np
import pyvista as pv
import readFile as rdFl
import input as I


#===========================================================================================================
def get_rayTubes(Pk, sk, theta_t, nk, surfaces):
    Pk_src = np.array([Pk_i[0] for Pk_i in Pk])                                 #points at the source 
    Pk_ap =  np.array([Pk_i[1] for Pk_i in Pk])                                 #pints at the aperture
    
    sk_src = np.array([sk_i[0] for sk_i in sk])                                 #direction (Poyting) of the source ray
    sk_ap = np.array([sk_i[1] for sk_i in sk])                                  #direction (Poyting) of the transmitted ray

    nk_src = np.array([nk_i[0] for nk_i in nk])                                 #direction (Poyting) of the source ray
    nk_ap = np.array([nk_i[1] for nk_i in nk]) 

    pts_2d = Pk_ap[:, :2]
    tri = Delaunay(pts_2d)                              
    triangles = tri.simplices   
                                                     #we create the triangles in the first interface of the dielectric

    Nrays = len(Pk_src)
    Ex_src = np.zeros(Nrays, dtype=complex)
    Ey_src = np.zeros(Nrays, dtype=complex)
    Ez_src = np.zeros(Nrays, dtype=complex)

    for i in range(Nrays):                                                      #calculation of the Ex, Ey, Ex of the source. Read file from CST.
        sx, sy, sz = sk_src[i]
        # ángulos del rayo i
        theta_i = np.arccos(sz)
        phi_i   = np.arctan2(sy, sx)
        if phi_i < 0:
            phi_i += np.pi*2
        Ex_src[i], Ey_src[i], Ez_src[i] = rdFl.get_cartesian_E(theta_i, phi_i)
        if np.isnan(Ex_src[i]):
            print('isnan E field in ray tubes line 42')
        Ex_src[i], Ey_src[i], Ez_src[i] = [1, 1, 1]

    A_src = np.sqrt(np.abs(Ex_src)**2 + np.abs(Ey_src)**2 + np.abs(Ez_src)**2)      #Amplitude of the electric field in the source
    # A_src = np.ones(Nrays)

    Ntri = triangles.shape[0]
    C_ap   = np.zeros((Ntri, 3))                #baricenters of the triangles
    dS_src = np.zeros(Ntri)                     #ray-tube area at the source
    dS_ap  = np.zeros(Ntri)                     #ray-tube area at the dielectric interface
    cos_th = np.zeros(Ntri)                     #angle between nk and sk (of the ray in the baricenter)
    A_ap   = np.zeros(Ntri, dtype=complex)      #ray-tube amplitude in the aperture 

    n_hat = nk_ap / np.linalg.norm(nk_ap)             #unit vector of the normal of the dielectric

    for l, (i, j, k) in enumerate(triangles):
        
        Ps_src = Pk_src[[i, j, k], :]    # (3,3)                            #points in the source and in the aperture
        Ps_ap  = Pk_ap[[i, j, k], :]     # (3,3)

        
        v1_src = Ps_src[1] - Ps_src[0]                                      
        v2_src = Ps_src[2] - Ps_src[0]
        dS_src[l] = 0.5 * np.linalg.norm(np.cross(v1_src, v2_src))          #areas in the source

        v1_ap = Ps_ap[1] - Ps_ap[0]
        v2_ap = Ps_ap[2] - Ps_ap[0]
        dS_ap[l] = 0.5 * np.linalg.norm(np.cross(v1_ap, v2_ap))             #areas in the aperture

        C_ap[l] = Ps_ap.mean(axis=0)                                        #baricenter in the aperture

        n_mean = nk_ap[[i, j, k], :].sum(axis=0)                            #mean normal
        n_mean = n_mean / np.linalg.norm(n_mean)

        s_mean = sk_ap[[i, j, k], :].sum(axis=0)                            #mean direction of the ray in the interface
        s_mean = s_mean / np.linalg.norm(s_mean)
        cos_th[l] = np.dot(n_mean, s_mean)
        
        A_l = (A_src[i] + A_src[j] + A_src[k]) / 3.0                        #mean amplitude of the ray in the source
       
        A_ap[l] = A_l * np.sqrt(dS_src[l] / (dS_ap[l] * cos_th[l]))         #amplitude at the aperture

        if np.isnan(A_ap[l]):            
            print('nan')

    if I.plotTubes: plot.plot_ray_tubes(Pk_src, sk_src, Pk_ap, triangles, surfaces)
    # plot_ray_tubes_with_all_rays(Pk, triangles, surfaces)

    return triangles, C_ap, A_ap, A_l, dS_src, dS_ap, cos_th, Ex_src, Ey_src, Ez_src
  #===========================================================================================================  






#===========================================================================================================
def ray_amplitudes_from_tube_amplitudes(triangles, A_tri, Nrays=None, weights=None):
    """
    Each ray (vertex) is shared by multiple ray tubes (triangles). The ray amplitude is
    defined as the mean of the amplitudes of all triangles incident to that ray. """

    triangles = np.asarray(triangles, dtype=int)
    A_tri = np.asarray(A_tri)

    if triangles.ndim != 2 or triangles.shape[1] != 3:
        raise ValueError("triangles must have shape (Ntri, 3)")
    if A_tri.ndim != 1 or A_tri.shape[0] != triangles.shape[0]:
        raise ValueError("A_tri must have shape (Ntri,) and match triangles.shape[0]")

    if Nrays is None:
        Nrays = int(triangles.max()) + 1

    # Accumulators for (weighted) sum and (weighted) count per ray
    A_sum = np.zeros(Nrays, dtype=np.complex128)
    W_sum = np.zeros(Nrays, dtype=np.float64)

    if weights is None:
        # Unweighted mean: each incident triangle contributes equally
        for t, (i, j, k) in enumerate(triangles):
            A = A_tri[t]
            A_sum[i] += A; W_sum[i] += 1.0
            A_sum[j] += A; W_sum[j] += 1.0
            A_sum[k] += A; W_sum[k] += 1.0
    else:
        # Weighted mean: each incident triangle contributes proportionally to weights[t]
        w = np.asarray(weights, dtype=np.float64)
        if w.ndim != 1 or w.shape[0] != triangles.shape[0]:
            raise ValueError("weights must have shape (Ntri,) and match triangles.shape[0]")

        for t, (i, j, k) in enumerate(triangles):
            wt = float(w[t])
            if wt <= 0.0 or not np.isfinite(wt):
                continue
            A = A_tri[t] * wt
            A_sum[i] += A; W_sum[i] += wt
            A_sum[j] += A; W_sum[j] += wt
            A_sum[k] += A; W_sum[k] += wt

    # Final mean; mark rays with no incident triangles as NaN
    A_ray = np.full(Nrays, np.nan + 1j*np.nan, dtype=np.complex128)
    valid = W_sum > 0.0
    A_ray[valid] = A_sum[valid] / W_sum[valid]

    # For unweighted case, counts is integer; for weighted, counts is still useful as "total weight"
    counts = W_sum.astype(int) if weights is None else W_sum
    return A_ray, counts
#===========================================================================================================

#===========================================================================================================
def sk_to_angles(s_vec):
    sx, sy, sz = s_vec
    # asumimos |s| = 1
    theta = np.arccos(sz)           # [0, pi]
    phi   = np.arctan2(sy, sx)      # [-pi, pi]
    return theta, phi
#===========================================================================================================

#==================================================
def distance(pointA, pointB):
    return (
        ((pointA[0] - pointB[0]) ** 2) +
        ((pointA[1] - pointB[1]) ** 2) +
        ((pointA[2] - pointB[2]) ** 2)
    ) ** 0.5 # fast sqrt
#==================================================

#=============================================================================
def snell(theta_inc, n1, n2):
    arg = abs(n1)/abs(n2) * np.sin(theta_inc)
    if abs(arg) <= 1:
        theta_ref = np.arcsin(abs(n1) / abs(n2) * np.sin(theta_inc))
    else:
        theta_ref = 0.
    return theta_ref
#=============================================================================


#=============================================================================
def getAngleBtwVectors(a, b):
    a = np.asarray(a)
    b = np.asarray(b)
    cos_theta = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    return np.arccos(cos_theta)
#=============================================================================


#=============================================================================
def calculateRayTubeAmpl(Pk, Pk1, Pk_ap, Pk_ap1, theta, Gk, Asrc):    #get the amplitude of the E field at the aperture plane
    #Pk - intersection of first ray and array
    #Pk1 - intersection of second ray and array
    #Pk_ap - intersection of first ray and aperture
    #Pk_ap1 - intersection of second ray and aperture
    dLk = distance(Pk, Pk1)#/2                               #ray tube width
    dck_ap = distance(Pk_ap, Pk_ap1)#/2                      #infinitesimal arc length of aperture
    dLk_ap = (dck_ap*np.cos(theta))
    return Asrc*np.sqrt(dLk/dLk_ap)*Gk, dLk, dLk_ap
# =============================================================================

#=============================================================================
def getAmplitude2D( sk_all, nk_all, Pk):
    row = []
    N_rays = len(sk_all)
    Ak_ap = np.zeros(N_rays-2)                                   #amplitude on aperture
    theta_k = np.zeros(N_rays)
    dck = np.zeros(N_rays-2) 
    dLk_src = np.zeros(N_rays-2) 
    dLk_ap = np.zeros(N_rays-2)                                     #infinitesimal arc length of aperture   
    Gk = np.zeros(N_rays) 
    nElement = np.zeros(N_rays)

    Ex_src = np.zeros(N_rays, dtype=complex)
    Ey_src = np.zeros(N_rays, dtype=complex)
    Ez_src = np.zeros(N_rays, dtype=complex)
    sk_src = np.array([sk_i[0] for sk_i in sk_all])
    Ak_src   = np.zeros(N_rays, dtype=complex)

    for i in range(0, N_rays):                                                   #for each ray
        #nk = [rays[i].normals[nSurfaces*2-2], rays[i].normals[nSurfaces*2-1]]       #normal to surface
        
        nk = nk_all[i][0]
        sk = sk_all[i][0]     
        sx, sy, sz = sk                                                       #poynting vector
        theta_k[i] = getAngleBtwVectors(nk, sk)  
        theta_i = np.arccos(sz)
        phi_i   = np.arctan2(sy, sx)
        if phi_i < 0:
            phi_i += np.pi*2
        Ex_src[i], Ey_src[i], Ez_src[i] = rdFl.get_cartesian_E(theta_i, phi_i)
        if np.isnan(Ex_src[i]):
            print('isnan E field in ray tubes line 42')
        
        Ak_src[i] = np.sqrt(np.abs(Ex_src[i])**2 + np.abs(Ey_src[i])**2 + np.abs(Ez_src[i])**2)
        # Ak_src[i] = 1
        
        if i > 1:                                                                   #exclude first ray, code will handle ray i-1 for each loop
            Pstart1 = Pk[i-2][0]                                     #intersection to the left of ray on array
            Pstart2 = Pk[i][0]                                           #intersection to the right of ray on array   
            dl_src = distance(Pstart1, Pstart2)
            Pap1 = Pk[i-2][1]                             #intersection to the left of ray on aperture
            Pap2 = Pk[i][1]                               #intersection to the left of ray of aperture
            dl_ap = distance(Pap1, Pap2)
            Gk = 1
            Ak_ap[i-2], dLk_src[i-2], dLk_ap[i-2]  = calculateRayTubeAmpl(Pstart1, Pstart2, Pap1, Pap2, theta_k[i-1], Gk, Ak_src[i])
    return Ak_ap, Ak_src, dLk_src, dLk_ap, Ex_src, Ey_src, Ez_src
#=============================================================================


def getA_source(sk0):
    Nrays = len(sk0)
    Ex_src = np.zeros(Nrays, dtype=complex)
    Ey_src = np.zeros(Nrays, dtype=complex)
    Ez_src = np.zeros(Nrays, dtype=complex)
    Et = np.zeros(Nrays)
    for i in range(len(sk0)):                                                      #calculation of the Ex, Ey, Ex of the source. Read file from CST.
            sx, sy, sz = sk0[i]
            # ángulos del rayo i
            theta_i = np.arccos(sz)
            phi_i   = np.arctan2(sy, sx)
            if phi_i < 0:
                phi_i += np.pi*2
            Ex_src[i], Ey_src[i], Ez_src[i] = rdFl.get_cartesian_E(theta_i, phi_i)
            Et[i] = rdFl.get_cartesian_E2(theta_i, phi_i)
            if np.isnan(Ex_src[i]):
                print('isnan')
            # Ex_src[i], Ey_src[i], Ez_src[i] = [1, 1, 1]

    A_src = np.sqrt(np.abs(Ex_src)**2 + np.abs(Ey_src)**2 + np.abs(Ez_src)**2)  
    Pt_src = sum(A_src**2)   
    Pt_src2 = sum(Et**2) 
    return A_src, Pt_src 






