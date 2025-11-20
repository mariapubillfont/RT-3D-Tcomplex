import pyvista as pv
import numpy as np
from matplotlib.cm import get_cmap
import random


def plotSurfaces(surfaces):
    p = pv.Plotter()    
    for i in range(0, len(surfaces)):
        
        si = surfaces[i]
        if i == len(surfaces)-1:    
            p.add_mesh(si.surface, show_edges=False, opacity=0.0, color='pink', style='wireframe')
        else:
            p.add_mesh(si.surface, opacity=0.8, color = True, style='wireframe')
    p.show()


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

    # plotter.show()


def plotDRT(surfaces, Pk, sk, show_ray_ids=True, show_dirs=False, dir_scale=0.04, show_all = False):
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
        plotter.add_mesh(mesh, color=color, opacity=0.5, show_edges=True, edge_color='#155DFC')

    
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
                plotter.add_lines(segment, color=ray_color, width=3)
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

        # Plot surface node normals in red
        # pc = pv.PolyData(points)
        # pc['normals'] = normals
        # arrows = pc.glyph(orient='normals', scale=True, factor=0.01)
        # plotter.add_mesh(arrows, color='red')

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