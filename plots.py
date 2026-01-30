import pyvista as pv
import numpy as np
from matplotlib.cm import get_cmap
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


#===========================================================================================================
def plotSurfaces(surfaces, origins, sk0, ray_length=1, rays_as_arrows=False,
                 ray_color="black", ray_width=2, origin_color="red", origin_size=10):
    p = pv.Plotter()

    for i, si in enumerate(surfaces):
        if i == len(surfaces) - 1:
            p.add_mesh(si.surface, show_edges=False, opacity=0.0, color='pink', style='wireframe')
        else:
            p.add_mesh(si.surface, opacity=0.8, color=True, style='wireframe')

    # Plot rays if provided
    if origins is not None and sk0 is not None:
        origins = np.asarray(origins, dtype=float)
        sk0 = np.asarray(sk0, dtype=float)

        # Normalize directions for display (avoid division by zero)
        norms = np.linalg.norm(sk0, axis=1)
        valid = norms > 0
        dirs = np.zeros_like(sk0)
        dirs[valid] = sk0[valid] / norms[valid, None]

        # Origins as points
        p.add_points(origins, color=origin_color, point_size=origin_size, render_points_as_spheres=True)

        lines = []
        for o, d in zip(origins, dirs):
            if np.allclose(d, 0.0):
                continue
            a = o
            b = o + ray_length * d
            lines.append(pv.Line(a, b))
        if lines:
            rays_mesh = lines[0]
            for ln in lines[1:]:
                rays_mesh = rays_mesh.merge(ln)
            p.add_mesh(rays_mesh, color=ray_color, line_width=ray_width)

    p.show()
#===========================================================================================================

#===========================================================================================================
def plot_axes(plotter, origen):
    # plotter = pv.Plotter()
    # origen = np.array([0, 0, 0])
    eje_x = np.array([0.05, 0, 0])
    eje_y = np.array([0, 0.05, 0])
    eje_z = np.array([0, 0, 0.05])

    plotter.add_arrows(origen, eje_x, mag=1, color="red")
    plotter.add_arrows(origen, eje_y, mag=1, color="green")
    plotter.add_arrows(origen, eje_z, mag=1, color="blue")

    # plotter.add_point_labels(origen, ['X\nY\nZ'], point_size=0, font_size=24, text_color='black')

    plotter.add_point_labels([origen + eje_x], ['X'], point_size=0, font_size=12, text_color='red')
    plotter.add_point_labels([origen + eje_y], ['Y'], point_size=0, font_size=12, text_color='green')
    plotter.add_point_labels([origen + eje_z], ['Z'], point_size=0, font_size=12, text_color='blue')

 #===========================================================================================================

#===========================================================================================================
def plotDRT(surfaces, Pk, sk, show_ray_ids=True, show_dirs=False, dir_scale=0.04, show_all = True):
    plotter = pv.Plotter()
    cmap = get_cmap("viridis")
    
    # Plot surfaces
    for i in range(len(surfaces)-1):
        if i == 0:
            color = 'lightgray'
        else:
            # color = '#51A2FF'
            color = 'lightblue'
        surf = surfaces[i]
        mesh = surf.surface
        plotter.add_mesh(mesh, color=color, opacity=0.5, show_edges=False, edge_color='#155DFC')

    
    ray_origins = []
    ray_labels = []
    for ray_id, rp in enumerate(Pk):
        # ray_points = rp[:-1]
        ray_points = rp
        n_points = len(ray_points)
        if n_points > 1 :
            ray_path = np.array(ray_points)
            if len(ray_path) == 2:
                ray_color = 'silver'
                point_color = 'red'
                if not show_all: continue 
            else:
                # ray_color = "#{:06x}".format(random.randint(0, 0xFFFFFF))
                ray_color = 'black'
                point_color = 'lightblue'

            for i in range(n_points - 1):
                segment = np.array([ray_path[i], ray_path[i+1]])
                plotter.add_lines(segment, color=ray_color, width=2)
            # for pt in ray_path:
            #     plotter.add_mesh(pv.Sphere(radius=0.0001, center=pt), color=point_color)
            
            if show_dirs:
                for pt, dir_vec in zip(ray_path, sk[Pk.index(ray_points)]):  # synchronize points and directions
                    dir_vec = np.array(dir_vec)
                    if np.linalg.norm(dir_vec) > 0:
                        arrow = pv.Arrow(start=pt, direction=dir_vec, scale=dir_scale)
                        plotter.add_mesh(arrow, color='black')

            if show_ray_ids:
                ray_origins.append(ray_path[1])
                ray_labels.append(str(ray_id))

    if show_ray_ids and ray_origins:
        ray_origins = np.array(ray_origins)
        plotter.add_point_labels(ray_origins, ray_labels, point_size=0, font_size=10, text_color="black")   

    plotter.add_title("Direct RT", font_size=14)
    plot_axes(plotter, np.array([0, 0.2, 0]))
    plotter.show()
#===========================================================================================================



#===========================================================================================================
def plot_normals(surfaces, nk, Pk):
    
    plotter = pv.Plotter()
    # 1. Plot all surfaces
    for i in range(len(surfaces) - 1):
        mesh = surfaces[i].surface
        plotter.add_mesh(mesh, opacity=0.4, color='lightgray', show_edges=False)

        if not mesh.point_data.get("Normals"):
                    mesh.compute_normals(point_normals=True, inplace=True)

        normals = mesh.point_normals
        points = mesh.points

    # 2. Plot normals for each ray, excluding first and last points
    for i in range(len(Pk)):
        points = np.array(Pk[i])
        normals = np.array(nk[i])

        if len(points) <= 2:
            continue  # No internos para plotear

        # Exclude first and last point and normal
        internal_points = points[1:-1]
        internal_normals = normals[1:-1]

        # Plot points
        plotter.add_points(internal_points, color='black', point_size=1, render_points_as_spheres=True)

        # Plot normals as arrows
        point_cloud = pv.PolyData(internal_points)
        point_cloud['normals'] = internal_normals
        arrows = point_cloud.glyph(orient='normals', scale=True, factor=0.005)
        plotter.add_mesh(arrows, color='blue')

    plotter.show()
#===========================================================================================================

#===========================================================================================================
def plot_ray_tubes(
        ray_origins,        # Pk_src : (Nrays, 3)
        ray_dirs,           # sk_src : (Nrays, 3)
        Pk_ap,              # (Nrays, 3)
        triangles,          # (Ntri, 3)
        surfaces,
        max_ray_length=None,
        ray_sample_step=0.1,
        show_source=True,
        show_aperture=True,
):
    import numpy as np
    import pyvista as pv

    Nrays = ray_origins.shape[0]
    plotter = pv.Plotter()

    # ------------------------------------------------------------
    # Surfaces
    # ------------------------------------------------------------
    for i in range(len(surfaces) - 1):
        surf = surfaces[i].surface
        plotter.add_mesh(
            surf,
            color="lightblue",
            opacity=0.5,
            show_edges=False
        )

    # ------------------------------------------------------------
    # Rays
    # ------------------------------------------------------------
    for i in range(Nrays):
        r0 = ray_origins[i]
        s  = ray_dirs[i]
        P  = Pk_ap[i]

        # distancia hasta la intersección
        t_int = np.linalg.norm(P - r0)

        # puntos del rayo
        if max_ray_length is None:
            ts = np.array([0.0, t_int])
        else:
            t_end = min(max_ray_length, t_int)
            ts = np.linspace(0.0, t_end, int(t_end / ray_sample_step) + 2)

        ray_points = r0 + ts[:, None] * s
        ray_line = pv.Spline(ray_points)

        plotter.add_mesh(ray_line, color="black", line_width=2)

    # ============================================================
    # SOURCE: puntos + triangulación
    # ============================================================
    if show_source:
        # puntos source
        pts_src = pv.PolyData(ray_origins)
        plotter.add_mesh(
            pts_src,
            color="green",
            point_size=8,
            render_points_as_spheres=True
        )

        # triangulación source (ray tubes en la source)
        faces_src = []
        for (i, j, k) in triangles:
            faces_src.extend([3, i, j, k])

        tri_mesh_src = pv.PolyData(ray_origins, faces_src)
        plotter.add_mesh(
            tri_mesh_src,
            color="lightgreen",
            opacity=0.35,
            show_edges=True,
            edge_color="darkgreen",
            line_width=1
        )

    # ============================================================
    # APERTURE: puntos + triangulación
    # ============================================================
    if show_aperture:
        pts_ap = pv.PolyData(Pk_ap)
        plotter.add_mesh(
            pts_ap,
            color="red",
            point_size=8,
            render_points_as_spheres=True
        )

        faces_ap = []
        for (i, j, k) in triangles:
            faces_ap.extend([3, i, j, k])

        tri_mesh_ap = pv.PolyData(Pk_ap, faces_ap)
        plotter.add_mesh(
            tri_mesh_ap,
            color="cyan",
            opacity=0.35,
            show_edges=True,
            edge_color="blue",
            line_width=1
        )

    plotter.show()
#===========================================================================================================

# def plot_ray_tubes(ray_origins, ray_dirs, Pk_ap, triangles, surfaces, max_ray_length=None,
#                                ray_sample_step=0.1):
#     Nrays = ray_origins.shape[0]
#     plotter = pv.Plotter()


#     for i in range(len(surfaces) - 1):
#         surf = surfaces[i].surface
#         plotter.add_mesh(surf,
#                          color="lightblue",
#                          opacity=0.5,
#                          show_edges=False)


#     for i in range(Nrays):
#         r0 = ray_origins[i]
#         s  = ray_dirs[i]
#         P  = Pk_ap[i]

#         # distancia hasta la intersección
#         t_int = np.linalg.norm(P - r0)

#         # definimos puntos del rayo
#         if max_ray_length is None:
#             ts = np.array([0.0, t_int])
#         else:
#             t_end = min(max_ray_length, t_int)
#             ts = np.linspace(0.0, t_end, int(t_end/ray_sample_step)+2)

#         ray_points = r0 + ts[:, None] * s

#         # Lo convertimos a una línea PolyData
#         ray_line = pv.Spline(ray_points)
#         plotter.add_mesh(ray_line, color="black", line_width=2)


#     pts = pv.PolyData(Pk_ap)
#     plotter.add_mesh(pts, color="red", point_size=8, render_points_as_spheres=True)


#     faces = []
#     for (i, j, k) in triangles:
#         faces.extend([3, i, j, k])   # formato de PyVista: [3, i, j, k]

#     tri_mesh = pv.PolyData(Pk_ap, faces)
#     plotter.add_mesh(tri_mesh,
#                      color="cyan",
#                      opacity=0.35,
#                      show_edges=True,
#                      edge_color="blue",
#                      line_width=1)


#     plotter.show()




