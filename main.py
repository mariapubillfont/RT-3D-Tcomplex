
import input as I
import numpy as np
import pyvista as pv
import mesh
import plots
import reflections as refl
import rayTracing as rt
import gain as gn
import rayTubes as tubes
import readFile as rdFl




############### Input import ##################
k0 = I.k0                                  
wv = I.wv
D = I.D   
p = I.p
Nx = I.Nx
Ny = I.Ny
bodies = I.bodies
typeSurface = I.typeSurface
################ end input import###########

surfaces = []
surfaces = mesh.create_surfaces()
N_sections = I.nSurfaces 
Nrays = I.Nrays

Pk = []
ray_ids = []
idx_intersected_faces = []
nk = []
sk = []
ray_lengths = []
T_tot = []
e = []
r_tes = []
t_tes = []
r_tms = []
t_tms = []
tandels = []
ndiel = []
theta_ts = []
gains = np.zeros(Nrays)
At_te = []
Ar_te = []
At_tm = []
Ar_tm = []
extra = 1e-6


if I.typeSrc == 'pw':
    x = np.linspace(-I.Lx/2 + extra, I.Lx/2 - extra, I.Nx)
    y = np.linspace(-I.Ly/2 + extra, I.Ly/2 - extra, I.Ny) 
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)
    origins = np.stack((X.ravel(), Y.ravel(), Z.ravel()), axis=1)
    theta = np.deg2rad(I.theta_pw)
    phi   = np.deg2rad(I.phi_pw) 
    k_dir = np.array([np.sin(theta)*np.cos(phi), 
                  np.sin(theta)*np.sin(phi), 
                  np.cos(theta)])
    sk0 = np.tile(k_dir, (Nrays, 1))

elif I.typeSrc == '2D':
    x = np.linspace(-I.Lx/2 + extra, I.Lx/2 - extra, Nrays)
    y = np.zeros(Nrays)
    z = np.zeros(Nrays)    
    origins = np.stack((x.ravel(), y.ravel(), z.ravel()), axis=1)
    theta = np.linspace(-np.pi, np.pi - 2*np.pi/(Nrays-1), Nrays)
    phi   = np.ones(Nrays)*np.deg2rad(0) 
    x_sk = np.sin(theta)*np.cos(phi)
    y_sk = np.sin(theta)*np.sin(phi)
    z_sk = np.cos(theta)
    sk0 = np.column_stack([x_sk, y_sk, z_sk])

else:
    ##### if same directions (debbug)
    # x = np.linspace(-I.Lx/2 + extra, I.Lx/2 - extra, I.Nx)
    # y = np.linspace(-I.Ly/2 + extra, I.Ly/2 - extra, I.Ny) 
    # X, Y = np.meshgrid(x, y)
    # Z = np.zeros_like(X)
    # origins = np.stack((X.ravel(), Y.ravel(), Z.ravel()), axis=1)
    # theta = np.deg2rad(0)
    # phi   = np.deg2rad(0) 
    # k_dir = np.array([np.sin(theta)*np.cos(phi), 
    #               np.sin(theta)*np.sin(phi), 
    #               np.cos(theta)])

    # sk0 = np.tile(k_dir, (N_rays, 1))

    origins = mesh.fibonacci_sphere_points(Nrays, I.Lx)
    sk0 = mesh.fibonacci_sphere_points(Nrays, 1)

if I.plotSurf: plots.plotSurfaces(surfaces, origins, sk0)


ray_ids, Pk, nk, sk, r_tes, t_tes, r_tms, t_tms, tandels, ndiel, ray_lengths, theta_ts, At_te, Ar_te, At_tm, Ar_tm  = rt.DRT(origins, Nx, Ny, sk0, surfaces )

if I.plotDRT: plots.plotDRT(surfaces, Pk, sk)
if I.plotNormals: plots.plot_normals(surfaces, nk, Pk)


# [Ak, Ak_src, dLk_src, dLk_ap, Ex_src, Ey_src, Ez_src] = tubes.getAmplitude2D(sk, nk, Pk)
[triangles, C_ap, A_ap, A_src, dS_src, dS_ap, cos_th, Ex_src, Ey_src, Ez_src] = tubes.get_rayTubes(Pk, sk, theta_ts, nk, surfaces)

# [A0_ray, counts] = tubes.ray_amplitudes_from_tube_amplitudes(triangles, A_ap)

[Ei_te, Ei_tm] = refl.field_decomposition(sk, nk, Ex_src, Ey_src, Ez_src)

# [A_src, Pt_src] = tubes.getA_source(sk0)
#[p_trans_te, p_refl_te, p_abs_te, p_trans_tm, p_refl_tm, p_abs_tm] = 
# refl.get_Pabs2D(At_te, Ar_te, At_tm, Ar_tm, Ak, Ak_src, Ei_te, Ei_tm, nk, sk, dLk_src, dLk_ap)
refl.get_Pabs(At_te, Ar_te, At_tm, Ar_tm, A_ap, A_src, Ei_te, Ei_tm, nk, sk, dS_src, dS_ap)

# p_abs = refl.getAbsorption(r_tes, t_tes, tandels, ndiel, ray_lengths, sk, nk, theta_ts)
# p_abs = refl.getAbsorption2(r_tes, t_tes, r_tms, t_tms, tandels, ndiel, ray_lengths, sk, nk, theta_ts)
# ray_length = 0.5  # longitud de los rayos

# origin = np.array([0, 0, 0])





