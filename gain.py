import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import Rbf
from matplotlib import cm
from matplotlib.colors import Normalize

filename = '3d_horn_patt.txt'



def readFile(filename):
    # Leer el archivo, saltando cabecera y línea de guiones
    data = np.genfromtxt(filename, skip_header=2, invalid_raise=False)

    # Filtrar filas incompletas (opcional)
    data = data[~np.isnan(data).any(axis=1)]

    # Extraer columnas que nos interesan
    theta = data[:, 0]      # Theta [deg.]
    phi = data[:, 1]        # Phi [deg.]
    abs_grlz = data[:, 2]   # Abs(Grlz) [dBi]

    radiation_spline = Rbf(theta, phi, abs_grlz, function='cubic')
    return radiation_spline



def getGain(sk0):
    x, y, z = sk0[:, 0], sk0[:, 1], sk0[:, 2]
    
    r = np.sqrt(x**2 + y**2 + z**2)  # módulo de cada vector
    
    # θ desde el eje Z
    theta_vec = np.degrees(np.arccos(z / r))
    theta_vec[z < 0] *= -1  # signo negativo si z < 0
    
    # φ en plano XY
    phi_vec = np.degrees(np.arctan2(y, x))  

    phi_vec = ((phi_vec + 90) % 180) - 90

    func = readFile(filename)
    gain_dbi = func(theta_vec,phi_vec)
    gain_lin = 10**(gain_dbi / 10)
    gain_norm = gain_lin / np.max(gain_lin)
    
    return gain_dbi, gain_lin, gain_norm




if 0:
    data = np.genfromtxt(filename, skip_header=2, invalid_raise=False)

    # Filtrar filas incompletas (opcional)
    data = data[~np.isnan(data).any(axis=1)]

    # Extraer columnas que nos interesan
    theta_org = data[:, 0]      # Theta [deg.]
    phi_org = data[:, 1]        # Phi [deg.]
    abs_grlz_org = data[:, 2]   # Abs(Grlz) [dBi]
    
    theta_plot = np.linspace(-180, 180, 20)  # theta de -90 a 90 grados

    # Plano E (phi = 0)
    phi_E = np.full_like(theta_plot, 20)
    sk0_E = np.column_stack([
        np.sin(np.radians(theta_plot)) * np.cos(np.radians(phi_E)),
        np.sin(np.radians(theta_plot)) * np.sin(np.radians(phi_E)),
        np.cos(np.radians(theta_plot))
    ])
    theta_rad = np.radians(theta_plot)
    phi_rad  = np.sin(np.radians(phi_E))
    z = np.sin(theta_rad)          # componente Z
    xy_proj = np.cos(theta_rad)    # proyección en XY
    x = xy_proj * np.cos(phi_rad)
    y = xy_proj * np.sin(phi_rad)
    gain_E = getGain(sk0_E)
    # gain_E = rbf(theta_plot, phi_E)

    # Plano H (phi = 90)
    phi_H = np.full_like(theta_plot, 90)
    sk0_H = np.column_stack([
        np.sin(np.radians(theta_plot)) * np.cos(np.radians(phi_H)),
        np.sin(np.radians(theta_plot)) * np.sin(np.radians(phi_H)),
        np.cos(np.radians(theta_plot))
    ])
    gain_H = getGain(sk0_H)

    # --- Plot 2D ---
    plt.figure(figsize=(8,5))
    plt.plot(theta_plot, gain_E, label='H-plane (phi=0°)')
    plt.plot(theta_plot, gain_H, label='E-plane (phi=90°)')
    plt.xlabel('Theta [°]')
    plt.ylabel('Gain [dBi]')
    plt.title('Radiation Pattern')
    plt.grid(True)
    plt.legend()
    plt.show()

def get_gain(theta_query, phi_query):
    radiation_spline = readFile(filename)
    return radiation_spline(theta_query, phi_query)

if 0:
    theta_test = -44.6     # grados
    phi_test = 90       # grados
    gain = get_gain(theta_test, phi_test)
    print(f"Ganancia interpolada en Theta={theta_test}°, Phi={phi_test}°: {gain:.2f} dBi")


if 0: # === 1. Leer datos originales ===
    data = np.genfromtxt(filename, skip_header=2, invalid_raise=False)
    data = data[~np.isnan(data).any(axis=1)]  # eliminar filas incompletas

    theta_org = data[:, 0]      # Theta [deg]
    phi_org = data[:, 1]        # Phi [deg]
    abs_grlz_org = data[:, 2]   # Abs(Grlz) [dBi]

    # === 2. Crear función interpolada ===
    rbf = Rbf(theta_org, phi_org, abs_grlz_org, function='cubic')

    # === 3. Definir planos de corte (φ = 0° y 90°) ===
    theta_plot = np.linspace(-180, 180, 721)  # más denso para suavidad

    # φ = 0° (plano E)
    phi_E = np.full_like(theta_plot, 0)
    gain_interp_E = rbf(theta_plot, phi_E)

    # φ = 90° (plano H)
    phi_H = np.full_like(theta_plot, 90)
    gain_interp_H = rbf(theta_plot, phi_H)

    # === 4. Extraer puntos originales de esos mismos planos ===
    tol = 1e-2
    mask_phi0 = np.isclose(phi_org % 360, 0, atol=tol)
    mask_phi90 = np.isclose(phi_org % 360, 90, atol=tol)

    theta_phi0 = theta_org[mask_phi0]
    gain_phi0 = abs_grlz_org[mask_phi0]
    theta_phi90 = theta_org[mask_phi90]
    gain_phi90 = abs_grlz_org[mask_phi90]

    # Ordenar para trazado
    order0 = np.argsort(theta_phi0)
    order90 = np.argsort(theta_phi90)
    theta_phi0, gain_phi0 = theta_phi0[order0], gain_phi0[order0]
    theta_phi90, gain_phi90 = theta_phi90[order90], gain_phi90[order90]

    # === 5. Plot comparativo ===
    plt.figure(figsize=(7, 5))

    # --- Plano H (φ=0°) ---
    plt.plot(theta_phi0, gain_phi0, 'o', color='tab:blue', label='CST φ=0° ')
    plt.plot(theta_plot, gain_interp_E, '-', color='navy', label='Interp. φ=0° ')

    # --- Plano E (φ=90°) ---
    plt.plot(theta_phi90, gain_phi90, 'o', color='tab:orange', label='CST φ=90° ')
    plt.plot(theta_plot, gain_interp_H, '-', color='darkred', label='Interp. φ=90°')

    # --- Formato del gráfico ---
    plt.xlabel('Theta [°]')
    plt.ylabel('R. Gain [dBi]')
    plt.title('E-plane and H-plane Radiation Patterns')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

