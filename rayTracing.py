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
def ray(surfaces, ski, Pki, nki, ray_lengthi, theta_t, id_cell, next_surf, idx, r_te, t_te, r_tm, t_tm, tandel, n_diel):
    lastSurf = surfaces[next_surf].isLastSurface
    isArray = surfaces[next_surf].isArray
    isFirstIx = idx == 0
    # isFirstIx = surfaces[next_surf].isFirstIx
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
            return ray(surfaces, ski, Pki, nki, ray_lengthi,theta_t, id_cell, next_surf, idx, r_te, t_te, r_tm, t_tm, tandel, n_diel)
            chosen_point = None
            chosen_cell = None
    else:
        next_surf += 1
        return ray(surfaces, ski, Pki, nki, ray_lengthi, theta_t, id_cell, next_surf, idx, r_te, t_te, r_tm, t_tm, tandel, n_diel)
        chosen_point = None
        chosen_cell = None

    
    idx += 1
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
            
    else:
        t = snell(i, n, n_in, n_out)
        ski.append(r)
        r_tei, t_tei, r_tmi, t_tmi, theta_ti = refl.fresnel_coefficients2(i, n, n_in, n_out, theta_t[-1])
        tandel.append(surfaces[next_surf].tand_in)
        n_diel.append(np.sqrt(surfaces[next_surf].er_in))    

    r_te.append(r_tei)
    t_te.append(t_tei)
    r_tm.append(r_tmi)
    t_tm.append(t_tmi)
    theta_t = np.append(theta_t, theta_ti)  
    
    surfaces[next_surf].isFirstIx = False
    if idx >+ I.maxRefl: 
        next_surf += 1
        idx = 1
    if lastSurf:
        return Pki, ski, nki, r_te, t_te, r_tm, t_tm, tandel, n_diel, ray_lengthi, theta_t
    else:
        return ray(surfaces, ski, Pki, nki, ray_lengthi, theta_t, id_cell, next_surf, idx, r_te, t_te, r_tm, t_tm, tandel, n_diel)
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
    theta_ts = [[] for _ in range(N_rays)] 
    tandels = [[] for _ in range(N_rays)] 
    n_diel = [[] for _ in range(N_rays)]
    
    

    for i in range(0, N_rays):
        next_surf = 1
        idx = 0

        if i == 2:
            print('drt')
        Pk[i], sk[i], nk[i], r_tes[i], t_tes[i], r_tms[i], t_tms[i], tandels[i], n_diel[i], ray_lengths[i], theta_ts[i]  = \
        ray(surfaces, sk[i], Pk[i], nk[i], ray_lengths[i], theta_ts[i],  id_cell[i], next_surf, idx, r_tes[i], t_tes[i], r_tms[i], t_tms[i], tandels[i], n_diel[i])


    return Pk, nk, sk, r_tes, t_tes, r_tms, t_tms, tandels, n_diel, ray_lengths, theta_ts
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
    

 #===========================================================================================================
def fibonacci_sphere(n_rays, randomize=False):
    rnd = 1.
    if randomize:
        rnd = np.random.random() * n_rays

    points = []
    offset = 2.0 / n_rays
    increment = np.pi * (3.0 - np.sqrt(5))  # ángulo dorado

    for i in range(n_rays):
        y = ((i * offset) - 1) + (offset / 2)
        r = np.sqrt(1 - y*y)
        phi = ((i + rnd) % n_rays) * increment
        x = np.cos(phi) * r
        z = np.sin(phi) * r
        points.append([x, y, z])  # ya está normalizado
    return np.array(points)
 #===========================================================================================================






# #===========================================================================================================
# def polarization(Pk, sk, nk, e_arr):
#     # e --> unit vector in the e-field direction
#     # v_perp --> unit vector perpendicular to the incidence plane
#     # e_perp --> projection of the e-field on v_perp
#     # v_paral --> unit vector parallel to the incidence plane
#     # e_paral --> projection of the e-field on v_paral
#     e = np.zeros([I.nSurfaces+1,np.shape(e_arr)[0],np.shape(e_arr)[1]])
#     e[0] = e_arr - np.tile(np.diagonal(e_arr@np.transpose(sk[0,:,:])).reshape(-1,1),(1,3))*sk[0,:,:]
#     e[0] = e[0]/np.tile(np.sqrt(e[0,:,0]**2+e[0,:,1]**2+e[0,:,2]**2).reshape(-1,1), (1,3))
#     # dot = np.diagonal(e_arr@np.transpose(sk[0,:,:]))
#     # insk = np.tile(dot.reshape(100,1),(1,3))*sk[0,:,:]

#     T_tot = np.ones(np.shape(e_arr)[0])
#     for i in range(I.nSurfaces):
#         # Compute parallel and perpendicular components of the e-field to the incident plane
#         v_perp = np.cross(nk[i+1,:,:],sk[i+1,:,:])
#         e_perp = np.tile(np.diagonal(e[i]@np.transpose(v_perp)).reshape(-1,1),(1,3))*v_perp
#         e_paral = e[i]-e_perp
#         # Debugging lines
#         # sum = e_perp+e_paral
#         # test = np.sqrt(sum[:,0]**2+sum[:,1]**2+sum[:,2]**2)
#         # Compute Fresnel transmisison coeffs
#         cos_theta_i = np.diagonal(nk[i+1,:,:]@np.transpose(sk[i,:,:]))
#         theta_i = np.arccos(cos_theta_i)
#         sin_theta_t = np.cross(nk[i+1,:,:],sk[i+1,:,:])
#         sin_theta_t = np.sqrt(sin_theta_t[:,0]**2+sin_theta_t[:,1]**2+sin_theta_t[:,2]**2)
#         theta_t = np.arcsin(sin_theta_t)
#         T_perp = (2*cos_theta_i*sin_theta_t)/(np.sin(theta_i+theta_t))
#         T_paral = (2*cos_theta_i*sin_theta_t)/(np.sin(theta_i+theta_t)*np.cos(theta_i-theta_t))
#         # Case where incident angle is 0
#         idxs1 = np.where(theta_i>-1e-3,1,0)
#         idxs2 = np.where(theta_i<1e-3,1,0)
#         idxs = np.where(idxs1*idxs2)
#         T_perp[idxs] = (2*np.sqrt(np.real(I.er[i])))/(np.sqrt(np.real(I.er[i]))+np.sqrt(np.real(I.er[i+1])))
#         T_paral[idxs] = T_perp[idxs]
#         # Apply coeffs
#         e_perp *= np.tile(T_perp.reshape(-1,1), (1,3))
#         e_paral *= np.tile(T_paral.reshape(-1,1), (1,3))    

#         sum = e_perp+e_paral
#         T_tot *= np.sqrt(sum[:,0]**2+sum[:,1]**2+sum[:,2]**2)

#         e[i+1] = sum/np.tile(np.sqrt(sum[:,0]**2+sum[:,1]**2+sum[:,2]**2).reshape(-1,1),(1,3))

#     return T_tot, e
# #===========================================================================================================


# def RRT(surfaces, ap_ray_origins, sk0):
#     # Inputs
#     direction = 'RRT'
#     N_sections = I.nSurfaces + 1
#     N_rays = I.N_rrt
#     sk = np.zeros([N_sections+1, N_rays, 3])
#     sk[0,:,:] = np.tile(sk0, (N_rays,1))
#     Pk = np.zeros([N_sections+1, N_rays, 3])
#     Pk[0,:,:] = ap_ray_origins.T
#     nk = np.zeros([N_sections+1, N_rays, 3])
#     nk[0,:,:] = np.tile(np.array([0,0,1]), (N_rays,1))
#     ray_lengths = np.zeros([N_rays, N_sections])
#     phi_ap = np.zeros([N_rays,1])
#     e_ap = np.zeros([N_rays,3])
#     intersected_faces = np.zeros([N_rays, N_sections])
#     next_surf = I.nSurfaces
#     idx = 0

#     # GO
#     Pk, sk, nk, ray_lengths, phi_ap, e_ap, idx_intersected_faces = ray(surfaces, direction, sk, Pk, nk, ray_lengths, phi_ap, e_ap, intersected_faces, next_surf, idx)
#     N_used_rays = np.shape(Pk)[1]

#     # Interpolate poynting vectors
#     f_inc_vectors = LinearNDInterpolator(Pk[-1][:,0:2], -sk[-2])

#     # Calculate phases
#     phases_rrt = np.zeros([N_used_rays])
#     path_length = np.zeros([N_used_rays])
#     for ii in range(N_sections):
#         path_length += ray_lengths[:,ii] * np.sqrt(abs(surfaces[N_sections-ii].er1))
#     phases_rrt =  I.k0 * path_length    

#     # Interpolate phases
#     f_phases_rrt = LinearNDInterpolator(Pk[-1][:,0:2], phases_rrt)

#     # Plot
#     # if I.plotRRT:
#         # plots.plot_rays(surfaces, Pk, N_sections, sk, N_used_rays, direction)

#     return f_inc_vectors, f_phases_rrt, path_length

# #===========================================================================================================


