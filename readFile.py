import numpy as np
from scipy.interpolate import griddata



def read_cst_farfield(filename):
    with open(filename, "r") as f:
        lines = f.readlines()

    data = []
    for line in lines[2:]:  # skip header lines
        parts = line.split()
        if len(parts) == 8:
            data.append([float(x) for x in parts])

    data = np.array(data)

    return {
        "theta_deg": data[:, 0],
        "phi_deg": data[:, 1],
        "abs_theta": data[:, 3],
        "phase_theta": data[:, 4],
        "abs_phi": data[:, 5],
        "phase_phi": data[:, 6],
    }


class FarFieldInterpolator:
    def __init__(self, filename):
        raw = read_cst_farfield(filename)

        # angles
        self.theta = raw["theta_deg"]
        self.phi   = raw["phi_deg"]

        # rebuild complex fields
        self.Etheta = raw["abs_theta"] * np.exp(1j * np.deg2rad(raw["phase_theta"]))
        self.Ephi   = raw["abs_phi"]   * np.exp(1j * np.deg2rad(raw["phase_phi"]))

        # points for interpolation
        self.points = np.column_stack([self.theta, self.phi])

    def evaluate(self, theta_q, phi_q, method="linear"):
        theta_q = np.atleast_1d(theta_q).astype(float)
        phi_q   = np.atleast_1d(phi_q).astype(float)

        # wrap phi into [0, 360)
        phi_q = np.mod(phi_q, 360)

        query = np.column_stack([theta_q, phi_q])

        # interpolate real + imag separately
        Eth_r = griddata(self.points, self.Etheta.real, query, method=method)
        Eth_i = griddata(self.points, self.Etheta.imag, query, method=method)
        Ephi_r = griddata(self.points, self.Ephi.real, query, method=method)
        Ephi_i = griddata(self.points, self.Ephi.imag, query, method=method)

        Etheta_q = Eth_r + 1j * Eth_i
        Ephi_q   = Ephi_r + 1j * Ephi_i

        return {
            "Etheta":       Etheta_q,
            "Ephi":         Ephi_q,
            "Etheta_mag":   np.abs(Etheta_q),
            "Etheta_phase": np.rad2deg(np.angle(Etheta_q)),
            "Ephi_mag":     np.abs(Ephi_q),
            "Ephi_phase":   np.rad2deg(np.angle(Ephi_q)),
        }


ff = FarFieldInterpolator("Sources/dipole_efield_lin.txt")

def get_cartesian_E(theta, phi):
    
    #read the E_field from the file
    # ff = FarFieldInterpolator("Sources/dipole_rgain_lin.txt")
    # ff = FarFieldInterpolator("Sources/dipole_efield_lin.txt")
    result = ff.evaluate(np.rad2deg(theta), np.rad2deg(phi))

    Etheta_mag = result["Etheta_mag"]
    Etheta_ph = np.deg2rad(result["Etheta_phase"])
    Ephi_mag = result["Ephi_mag"]
    Ephi_ph = np.deg2rad(result["Ephi_phase"])

    Etheta = Etheta_mag*(np.cos(Etheta_ph) + 1j*np.sin(Etheta_ph)) 
    Ephi   = Ephi_mag*(np.cos(Ephi_ph) + 1j*np.sin(Ephi_ph)) 
    # Precompute trigonometric functions
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    cos_phi   = np.cos(phi)
    sin_phi   = np.sin(phi)

    # Apply spherical-to-Cartesian transformation
    Ex = Etheta * cos_theta * cos_phi - Ephi * sin_phi
    Ey = Etheta * cos_theta * sin_phi + Ephi * cos_phi
    Ez = -Etheta * sin_theta
    # if np.isnan(Ex):
    #     print('lol')
    return Ex, Ey, Ez


def get_cartesian_E2(theta, phi):
    
    #read the E_field from the file
    # ff = FarFieldInterpolator("Sources/dipole_rgain_lin.txt")
    # ff = FarFieldInterpolator("Sources/dipole_efield_lin.txt")
    result = ff.evaluate(np.rad2deg(theta), np.rad2deg(phi))

    Etheta_mag = result["Etheta_mag"]
    Etheta_ph = np.deg2rad(result["Etheta_phase"])
    Ephi_mag = result["Ephi_mag"]
    Ephi_ph = np.deg2rad(result["Ephi_phase"])

    Etheta = Etheta_mag*(np.cos(Etheta_ph) + 1j*np.sin(Etheta_ph)) 
    Ephi   = Ephi_mag*(np.cos(Ephi_ph) + 1j*np.sin(Ephi_ph)) 
    # Precompute trigonometric functions
    Et =  np.abs(Etheta)**2 + np.abs(Ephi)**2
    #Pt = 1/(2*377)*Et**2
    return Et