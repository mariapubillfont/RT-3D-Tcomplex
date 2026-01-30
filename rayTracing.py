import input as I
import plots
import numpy as np
import reflections as refl
import pyvista as pv
import gmsh

from scipy.interpolate import LinearNDInterpolator


################################## CLASS DEFINITIONS #######################################################

class Surface:
    def __init__(self, surface, er_out, er_in, tand_out, tand_in, isArray, isAperturePlane, isLastSurface, isFirstIx):
        self.surface = surface                          # pyvista PolyData
        self.faces = surface.faces                      # faces of the surface
        self.nodes = surface.points                     # nodes of the surface
        self.er_in = er_in                                  # permittivity "below" (closer to the array) the surface
        self.er_out = er_out                                  # permittivity "above" the surface
        self.tand_in = tand_in                              # loss tangent  inside the surface
        self.tand_out = tand_out                              # loss tangent outside the surface
        self.isArray = isArray                          # true if the surface is the array
        self.isAperturePlane = isAperturePlane          # true if the surface is the aperture plane
        self.isLastSurface = isLastSurface              # true if the surface is the last surface of the radome
        self.isFirstIx = isFirstIx


class Ray:
    def __init__(self, Pki, ski, nki, r_te, t_te, r_tm, t_tm, tandel, n_diel, ray_length, theta_t):
        self.Pk = Pki
        self.sk = ski
        self.nk = nki
        self.r_te = r_te
        self.t_te = t_te
        self.r_tm = r_tm
        self.t_tm = t_tm
        self.tandelta = tandel
        self.n_diel = n_diel
        self.ray_length = ray_length
        self.theta_t = theta_t


############################################################################################################

#===========================================================================================================
def find_normals(point, face_intersected, surface):
    surface_nodes = surface.points
    normals_nodes = surface.compute_normals(cell_normals=False)['Normals']

    same_val1 = np.all(abs(surface_nodes) < abs(surface_nodes[0,:])+1e-3, axis = 0)                         # checking if all the points have the same value in the  
    same_val2 = np.all(abs(surface_nodes) > abs(surface_nodes[0,:])-1e-3, axis = 0)                         # z-axis (flat surface) bc the interpolator needs to treat 
    same_val = same_val1[2] and same_val2[2]                                                                # this case separately
    if same_val:
        surface.flip_normals()
        normals_nodes = surface.compute_normals(cell_normals=False)['Normals']
        f = LinearNDInterpolator(surface_nodes[:,0:2], normals_nodes.reshape([len(normals_nodes),3]))
        interp_val = f(point[0:2])
    else:
        faces_areas = surface.compute_cell_sizes(length=False, volume=False).cell_data['Area']
        faces_nodes = surface.faces
        interp_val = barycentric_mean(point, face_intersected, faces_areas, faces_nodes, surface_nodes, normals_nodes)
    # CORRECCIÓN: Forzar normales en las tapas del cilindro
    z_min = surface_nodes[:, 2].min()
    z_max = surface_nodes[:, 2].max()
    tol = 1e-5  # tolerancia para identificar si está en la tapa

    if abs(point[2] - z_min) < tol:
        interp_val = np.array([0, 0, -1])
    elif abs(point[2] - z_max) < tol:
        interp_val = np.array([0, 0, 1])
             

    return interp_val
#===========================================================================================================

#===========================================================================================================
def snell(i, n, ninc, nt):
## I call it snell but it's actually Heckbert's method
## i --> unit incident vector
## n --> unit normal vector to surface
## ni --> refractive index of the first media
## nt --> refractive index of the second media
    i = np.array(i, dtype=complex)
    n = np.array(n, dtype=complex)
    alpha = ninc / nt
    d = np.dot(i, n)
    in_b = 1 - (alpha**2) * (1 - d**2)
    b = np.sqrt(in_b)
    t_complex = alpha * i + (b - alpha * d) * n
    cos_theta_t_complex = np.dot(t_complex, n)
    t_real_part = np.real(t_complex)
    norm_real = np.linalg.norm(t_real_part)
    if norm_real < 1e-12:
        dir_real = None   # evanescent / no propagating real direction
    else:
        dir_real = t_real_part / norm_real

    return t_complex
    # if in_b < 0:
    #     return np.full_like(i, np.nan)
    # else:
    #     b = np.sqrt(in_b)
    #     t = alpha * i + (b - alpha * d) * n
        # return t / np.linalg.norm(t)
#===========================================================================================================

def reflect(i, n):
    i = np.array(i, dtype=complex)
    n = np.array(n, dtype=complex)

    r = i - 2 * np.dot(i, n) * n
    return r / np.linalg.norm(r)


#===========================================================================================================
def distance(A, B):
    A = np.array(A)
    B = np.array(B)
    return np.linalg.norm(A - B)
#==========================================================================================================


    # surfaces --> meshed dome surfaces
    # ray_origin --> start point of the ray
    # sk --> vector defining the ray direction
    # Pk --> points where the rays intersect with the surfaces
    # nk --> normals to surfaces in intersection points
    # ray_lengths --> lengths of each section of the ray
    # i --> vector of the incident ray to each surface
    # intersected_faces --> list of indexes of each intersected face by a ray
    # next_surf --> next surface to be intersected
#===========================================================================================================
def ray(surfaces, ski, Pki, nki, ray_lengthi, theta_t, id_cell, next_surf, idx, r_te, t_te, r_tm, t_tm, tandel, n_diel, At_tei, Ar_tei, At_tmi, Ar_tmi, num_trans, num_refl):
    lastSurf = surfaces[next_surf].isLastSurface
    isArray = surfaces[next_surf].isArray
    ray_len_t = 0
    if idx == 0:
        isFirstIx = True
        num_refl = 0
        num_trans = 0
    else:
        isFirstIx = False
    current_surf = surfaces[next_surf].surface

    origin = np.array(Pki[-1])
    direction = np.array(ski[-1])
    end_point = origin + 1e3 * direction

    points, id_cells = current_surf.ray_trace(origin, end_point)
    if len(points) > 0:
        distances = np.linalg.norm(points - origin, axis=1)
        valid_mask = distances > 1e-6
        
        if np.any(valid_mask):
            valid_points = points[valid_mask]
            valid_cells = np.array(id_cells)[valid_mask]
            valid_distances = distances[valid_mask]
            min_idx = np.argmin(valid_distances)
            chosen_point = valid_points[min_idx].tolist()
            chosen_cell = valid_cells[min_idx].tolist()
        
        else:
            next_surf += 1
            return ray(surfaces, ski, Pki, nki, ray_lengthi, theta_t, id_cell, next_surf, idx, r_te, t_te, r_tm, t_tm, tandel, n_diel, At_tei, Ar_tei, At_tmi, Ar_tmi, num_trans, num_refl)
            chosen_point = None
            chosen_cell = None
    else:
        next_surf += 1
        return ray(surfaces, ski, Pki, nki, ray_lengthi, theta_t, id_cell, next_surf, idx, r_te, t_te, r_tm, t_tm, tandel, n_diel, At_tei, Ar_tei, At_tmi, Ar_tmi, num_trans, num_refl)
        chosen_point = None
        chosen_cell = None

    
    # idx += 1
     # Snell

    er_in = surfaces[next_surf].er_in
    er_out = surfaces[next_surf].er_out
    tand_in = surfaces[next_surf].tand_in
    tand_out =  surfaces[next_surf].tand_out
    n_in = np.sqrt(er_in*(1-1j*tand_in))  # n_ext
    n_out = np.sqrt(er_out*(1-1j*tand_out))  # n_int
    normal = find_normals(chosen_point, chosen_cell, current_surf)
    if isFirstIx: normal = -normal


    Pki.append(chosen_point)
    id_cell.append(chosen_cell)
    ray_lengthi.append(distance(origin, chosen_point)) 
    nki.append(normal)
    
    i = ski[-1]
    n = nki[-1]
    r = reflect(i,n)

    if isFirstIx:
        t = snell(i, n, n_out, n_in)
        ski.append(t)
        r_tei, t_tei, r_tmi, t_tmi, theta_ti = refl.fresnel_coefficients2(i, n, n_out, n_in, np.abs(np.dot(i, n)) )
        kc = I.k0*n_in
            
    else:
        t = snell(i, n, n_in, n_out)
        ski.append(r)
        r_tei, t_tei, r_tmi, t_tmi, theta_ti = refl.fresnel_coefficients2(i, n, n_in, n_out, theta_t[-1])
        tandel.append(surfaces[next_surf].tand_in)
        n_diel.append(np.sqrt(surfaces[next_surf].er_in))    
        # kc = I.k0*n_in
        kc = I.k0*n_diel[0]*np.sqrt(1-complex(0,tandel[0]))
    r_te.append(r_tei) #s-polarized
    t_te.append(t_tei)
    r_tm.append(r_tmi) #p-polarized
    t_tm.append(t_tmi)
    theta_t = np.append(theta_t, theta_ti) 

    if r_tei == 1:   #first reflection is total reflextion
        # print('total reflection in ray ' + ', and reflection num. ' + str(idx))
        return ray(surfaces, ski, Pki, nki, ray_lengthi, theta_t, id_cell, next_surf, idx, r_te, t_te, r_tm, t_tm, tandel, n_diel, At_tei, Ar_tei, At_tmi, Ar_tmi, num_trans, num_refl)

    
    if idx == 0:
        Ar_tei = np.append(Ar_tei, r_tei)
        Ar_tmi = np.append(Ar_tmi, r_tmi)

    elif not lastSurf:                 
        theta_1 = theta_t[0]
        # ray_len_t = sum(ray_lengthi[0:-1])     #change it for each iteration add its length, not saving all the ray lengths
        ray_len_t = sum(ray_lengthi[1:])
        if n[2] >= 0 :           #for transmission handling, when z-component of normal is positive
            # if r_tei != 1:
            num_trans += 1
            At_tei = np.append(At_tei, (-r_te[0]*r_te[1])**(num_trans-1)*(t_te[0]*t_te[1])*np.exp(-1j * kc * (ray_len_t) *(theta_1)*np.abs(theta_1)))
            At_tmi = np.append(At_tmi, (-r_tm[0]*r_tm[1])**(num_trans-1)*(t_tm[0]*t_tm[1])*np.exp(-1j * kc * (ray_len_t) *(theta_1)*np.abs(theta_1)))
            
        else:
            num_refl += 1
            Ar_tei =  np.append(Ar_tei, t_te[0]*t_te[1] * r_te[1] * (-r_te[0]*r_te[1])**(num_refl-1) * np.exp(-1j * kc * (ray_len_t)* (theta_1)*np.abs(theta_1)))
            Ar_tmi =  np.append(Ar_tmi, t_tm[0]*t_tm[1] * r_tm[1] * (-r_tm[0]*r_tm[1])**(num_refl-1) * np.exp(-1j * kc * (ray_len_t)* (theta_1)*np.abs(theta_1)))
        if np.abs(Ar_tei[-1]) < I.Ampl_treshold and np.abs(At_tei[-1]) < I.Ampl_treshold and next_surf < len(surfaces)-1: 
            next_surf += 1
            # idx = 1
            ski[-1] = t

    idx += 1
    surfaces[next_surf].isFirstIx = False
    if np.abs(Ar_tei[-1]) < I.Ampl_treshold and idx == 1: 
        next_surf += 1
        ski[-1] = t
    if lastSurf:
        return Pki, ski, nki, r_te, t_te, r_tm, t_tm, tandel, n_diel, ray_lengthi, theta_t, At_tei, Ar_tei, At_tmi, Ar_tmi
    else:
        
        return ray(surfaces, ski, Pki, nki, ray_lengthi, theta_t, id_cell, next_surf, idx, r_te, t_te, r_tm, t_tm, tandel, n_diel, At_tei, Ar_tei, At_tmi, Ar_tmi, num_trans, num_refl)
#===========================================================================================================


#===========================================================================================================
def DRT (ray_origins, Nx, Ny, sk_0, surfaces):
    N_rays = I.Nrays

    Pk = [[ray_origins[i].tolist()] for i in range(N_rays)]                            #List of N_rays elements, 1 element per each ray
    sk = [[] for _ in range(N_rays)] 
    for i in range(N_rays):
        sk[i].append(sk_0[i])
    nk = [[[0.0, 0.0, 1.0]] for _ in range(N_rays)] 
    ray_lengths = [[] for _ in range(N_rays)] 
    id_cell = [[] for _ in range(N_rays)] 
    r_tes = [[] for _ in range(N_rays)] 
    t_tes = [[] for _ in range(N_rays)] 
    r_tms = [[] for _ in range(N_rays)] 
    t_tms = [[] for _ in range(N_rays)] 
    At_te = [[] for _ in range(N_rays)]
    Ar_te = [[] for _ in range(N_rays)]
    At_tm = [[] for _ in range(N_rays)]
    Ar_tm = [[] for _ in range(N_rays)]
    theta_ts = [[] for _ in range(N_rays)] 
    tandels = [[] for _ in range(N_rays)] 
    n_diel = [[] for _ in range(N_rays)]
    
   

    for i in range(0, N_rays):
        next_surf = 1
        idx = 0


        if i == 8:
            print(i)
        Pk[i], sk[i], nk[i], r_tes[i], t_tes[i], r_tms[i], t_tms[i], tandels[i], n_diel[i], ray_lengths[i], theta_ts[i], At_te[i], Ar_te[i], At_tm[i], Ar_tm[i]  = \
        ray(surfaces, sk[i], Pk[i], nk[i], ray_lengths[i], theta_ts[i],  id_cell[i], next_surf, idx, r_tes[i], t_tes[i], r_tms[i], t_tms[i], tandels[i], n_diel[i], At_te[i], Ar_te[i], At_tm[i], Ar_tm[i], num_trans = 0, num_refl = 0)



 

    valid_idx = [i for i, P in enumerate(Pk) if len(P) > 2]
    ray_id = valid_idx[:]  # ray_id[i] = índice original del rayo

    Pk          = [Pk[i]          for i in valid_idx]
    sk          = [sk[i]          for i in valid_idx]
    nk          = [nk[i]          for i in valid_idx]
    r_tes       = [r_tes[i]       for i in valid_idx]
    t_tes       = [t_tes[i]       for i in valid_idx]
    r_tms       = [r_tms[i]       for i in valid_idx]
    t_tms       = [t_tms[i]       for i in valid_idx]
    tandels     = [tandels[i]     for i in valid_idx]
    n_diel      = [n_diel[i]      for i in valid_idx]
    ray_lengths = [ray_lengths[i] for i in valid_idx]
    theta_ts    = [theta_ts[i]    for i in valid_idx]
    At_te       = [At_te[i]       for i in valid_idx]
    Ar_te       = [Ar_te[i]       for i in valid_idx]
    At_tm       = [At_tm[i]       for i in valid_idx]
    Ar_tm       = [Ar_tm[i]       for i in valid_idx]
   
    return ray_id, Pk, nk, sk, r_tes, t_tes, r_tms, t_tms, tandels, n_diel, ray_lengths, theta_ts, At_te, Ar_te, At_tm, Ar_tm
#===========================================================================================================


 #===========================================================================================================
def barycentric_mean(point, face_idx, faces_areas, faces_nodes, surface_nodes, normals_nodes):
    # Reshape faces_nodes from PyVista format: (F*4,) → (F, 3)
    n_faces = len(faces_areas)
    faces = faces_nodes.reshape((n_faces, 4))[:, 1:]  # Drop the leading '3'

    i0, i1, i2 = faces[face_idx]                  # Nodo de la cara intersectada
    A = surface_nodes[i0]
    B = surface_nodes[i1]
    C = surface_nodes[i2]

    nA = normals_nodes[i0]
    nB = normals_nodes[i1]
    nC = normals_nodes[i2]

    # Vectores para coordenadas baricéntricas
    v0 = B - A
    v1 = C - A
    v2 = point - A

    d00 = np.dot(v0, v0)
    d01 = np.dot(v0, v1)
    d11 = np.dot(v1, v1)
    d20 = np.dot(v2, v0)
    d21 = np.dot(v2, v1)

    denom = d00 * d11 - d01 * d01
    if np.abs(denom) < 1e-12:
        return nA  # Fallback: usa la normal del primer vértice

    v = (d11 * d20 - d01 * d21) / denom
    w = (d00 * d21 - d01 * d20) / denom
    u = 1.0 - v - w

    normal = u * nA + v * nB + w * nC

    norm = np.linalg.norm(normal)
    if norm > 1e-12:
        normal /= norm

    return normal
 #===========================================================================================================


 #===========================================================================================================
def shootRays(Nx, Ny, Lx, Ly, typeSrc):
    n_theta = int(I.Ntheta)
    n_phi = int(I.Nphi)
    Nrays = int(n_theta*n_phi*Nx*Ny)
    origins = np.zeros([n_theta*n_phi, 3])

    if typeSrc[0] == 'iso':
        sk0 = []
        phi = np.linspace(I.rangePhi[0], I.rangePhi[1], n_phi )
        theta = np.linspace(I.rangeTheta[0], I.rangeTheta[1], n_theta)
        sk0 = np.zeros([Nrays, 3])
        ij = 0
        for i in range(0, n_theta):
            for j in range(0, n_phi):    
                x = np.sin(theta[i]) * np.cos(phi[j])
                y = np.sin(theta[i]) * np.sin(phi[j])
                z = np.cos(theta[i])
                sk0[ij] = np.vstack((x, y, z)).T  # shape (n_rays, 3)
                ij += 1
    return origins, sk0
 #===========================================================================================================
    





