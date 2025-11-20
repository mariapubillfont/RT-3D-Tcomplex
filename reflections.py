import numpy as np
import input as I

#===========================================================================================================
def fresnel_coefficients2(i, n, n_inc, n_t, cosi):
    i = np.array(i, dtype=complex)
    n = np.array(n, dtype=complex)

    cos_theta_i = np.dot(i, n)     
    # cos_theta_i = np.abs(np.dot(i, n))
    # cos_theta_i = cosi
    sin_theta_i2 = 1 - cos_theta_i**2
    sin_theta_t2 = (n_inc / n_t)**2 * sin_theta_i2

    if sin_theta_t2 > 1.0:
        # Total in_ternal reflection
        return 1,0, 1j
    cos_theta_t = np.sqrt(1 - sin_theta_t2)

    rs = (n_inc * cos_theta_i - n_t * cos_theta_t) / (n_inc * cos_theta_i + n_t * cos_theta_t)                      # s-polarized (TE)
    ts = (2 * n_inc * cos_theta_i) / (n_inc * cos_theta_i + n_t * cos_theta_t)
   
    rp = (n_t * cos_theta_i - n_inc * cos_theta_t) / (n_t * cos_theta_i + n_inc * cos_theta_t)                       # p-polarized (TM)
    tp = (2 * n_inc * cos_theta_i) / (n_t * cos_theta_i + n_inc * cos_theta_t)

    R = np.abs(rs)**2
    T = 1 -  R

    return rs, ts, rp, tp, cos_theta_t
#===========================================================================================================

#===========================================================================================================
def getAbsorption(rs, ts, tandels, n_diel, ray_lengths, sk, nk, theta_ts, gains_norm):
    abs_total = 0
    nrays_refl = 0
    absr = []
    pt_total = 0
    pr_total = 0
    for i in range(0, len(rs)):
        n_refl = len(rs[i]) - 1
        abs_total = np.append(abs_total, gains_norm[i] )

        if n_refl > 2:
            
            ray_len = ray_lengths[i]
            r_te = rs[i]
            t_te = ts[i]
            absr_i = np.zeros(n_refl + 1)
            An = []
            tandel = tandels[i][0]
            nr = n_diel[i][0]
            kc = I.k0*nr*np.sqrt(1-complex(0,tandel))
            alfa = np.abs(np.imag(kc))
            An_tr = []
            An_rf = []
            ptr_aux = 0
            pref_aux = 0
            ray_len_t = 0
            num_trans = 0
            num_refl = 0
            r1 = r_te[0]
            r2 = r_te[1]
            # r2 = 0.718580058533991 - 0.01302887870235758j
            t1 = t_te[0]
            t2 = t_te[1]
            # t2 = 1.7185800585339910 - 0.01302887870235758j
            cost = theta_ts[i]
            A0 = np.sqrt(gains_norm[i])

            if r_te[1] == 1:
                continue
            if i == 19:
                print('stop')
            nrays_refl += 1
            for j in range(0, n_refl):
                r_tej = r_te[j]
                t_tej = t_te[j]
                # t2 = t_tej
                # r2 = r_tej
                cos_trj = np.acos(np.abs(np.dot(nk[i][j+1], sk[i][j+1]))) #nos saltamos nk[0] porque es s0
                cos_trj = cost[0]
               
                if j == 0:
                    An_rf = np.append(An_rf, r1)

                else:                 
                    ray_len_t += ray_len[j]
                    if nk[i][j+1][2] >= 0 :           #for transmission handling, when z-component of normal is positive
                        if r_tej != 1:
                            num_trans += 1
                            if num_trans <= 1: 
                                # An_tr = np.append(An_tr, t_te[0]*t_te[j]*np.exp(-1j * kc * ray_len[j] * j * np.cos(cos_trj)**2)) 
                                An_tr = np.append(An_tr, t1*t2*np.exp(-1j * kc * ray_len[j] * j * (cos_trj)*np.abs(cos_trj))) 
                                # An_tr = np.append(An_tr,  t_te[0]*t_te[j]*np.exp(-1j * kc * ray_len[j] * j * (cos_trj)*np.abs(cos_trj)))
                            else: 
                                An_tr = np.append(An_tr, (-r1*r2)**(num_trans-1)*(t1*t2)*np.exp(-1j * kc * (ray_len_t) *(cos_trj)*np.abs(cos_trj)))
                    
                    else:
                        num_refl += 1
                        if j <= 2: 
                            An_rf =  np.append(An_rf, t1*t2 * r2 * np.exp(-1j * kc * (ray_len_t)* (cos_trj)*np.abs(cos_trj)))
                        else: 
                            An_rf =  np.append(An_rf, t1*t2 * r2 *(-r2*r1)**(num_refl-1) * np.exp(-1j * kc * (ray_len_t)* (cos_trj)*np.abs(cos_trj)))

               
            ptr_aux = sum(An_tr)
            pref_aux = sum(An_rf)
            pt = np.abs(ptr_aux)**2*gains_norm[i]
            pr = np.abs(pref_aux)**2*gains_norm[i]
            pt_total += pt
            pr_total += pr
            print(str(i) + '-ray and power T: '+ str(pt) + ' and power R: '+ str(pr)) 
    
    # p_trans_t = pt_total/nrays_refl
    # p_refl_t = pr_total/nrays_refl
    p_trans_t = pt_total/nrays_refl
    p_refl_t = pr_total/nrays_refl
    p_abs_t = 1 - p_trans_t - p_refl_t
    print('Absorbed - ' + str(p_abs_t*100))
    print('Ptrans - ' + str(p_trans_t*100))
    print('Prefl - ' + str(p_refl_t*100))
    print('Absorbed 2 - ' + str(p_abs_t*100/(I.Nrays)*nrays_refl))
    print('Absorbed 3 - ' + str(p_abs_t*100/(I.Nrays)*sum(abs_total)))
    # print('Ptrans - ' + str(pt_total*100/(I.Nrays/2)))
    # print('Prefl - ' + str(pr_total*100/(I.Nrays/2)))
    
    return p_abs_t
#===========================================================================================================




# ===========================================================================================================
def getAbsorption2(r_tes, t_tes, r_tms, t_tms, tands, ns, r_lens, sk, nk, theta_ts, gains_norm):
    abs_total = 0
    nrays_refl = 0
    absr = []
    pt_total = 0
    pr_total = 0
    for i in range(0, len(r_tes)):
        n_refl = len(r_tes[i]) - 1
        
        if n_refl > 2:
            r_te = r_tes[i]
            t_te = t_tes[i]
            r_tm = r_tms[i]
            t_tm = t_tms[i]
            ski = sk[i]
            nki = nk[i]

            if r_te[1] == 1:   #first reflection is total reflextion
                continue

            for j in range(0, n_refl):
                #decompose the field
                e_perp, e_pll = field_decomposition(ski[j], nki[j])
            
    return 0
# ===========================================================================================================





#===========================================================================================================
def unit(v):
    v = np.array(v, dtype=float)
    n = np.linalg.norm(v)
    if n == 0:
        raise ValueError("Nul vector")
    return v / n 
#===========================================================================================================


#===========================================================================================================
def field_decomposition(k, n):
    k_hat = unit(k)
    n_hat = unit(n)

    # Formula: ê_parallel = k̂ × ( n̂ × k̂ ) / | k̂ × ( n̂ × k̂ ) |
    cross_inner = np.cross(n_hat, k_hat)
    cross_outer = np.cross(k_hat, cross_inner)
    norm_cross_outer = np.linalg.norm(cross_outer)

    e_par = cross_outer / norm_cross_outer

    # Formula: ê_perp = - ( ê_parallel × k̂ )
    e_perp = -np.cross(e_par, k_hat)

    return e_par, e_perp
#===========================================================================================================