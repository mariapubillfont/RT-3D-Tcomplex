import numpy as np
import input as I

#===========================================================================================================
def fresnel_coefficients(i, n, n_inc, n_t):
    i = np.array(i, dtype=float)
    n = np.array(n, dtype=float)

    # cos_theta_i = np.dot(i, n)     
    cos_theta_i = np.abs(np.dot(i, n))
    sin_theta_i2 = 1 - cos_theta_i**2
    sin_theta_t2 = (n_inc / n_t)**2 * sin_theta_i2

    if sin_theta_t2 > 1.0:
        # Total in_ternal reflection
        return 1,0
    cos_theta_t = np.sqrt(1 - sin_theta_t2)

    rs = (n_inc * cos_theta_i - n_t * cos_theta_t) / (n_inc * cos_theta_i + n_t * cos_theta_t)                      # s-polarized (TE)
    ts = (2 * n_inc * cos_theta_i) / (n_inc * cos_theta_i + n_t * cos_theta_t)
   
    rp = (n_t * cos_theta_i - n_inc * cos_theta_t) / (n_t * cos_theta_i + n_inc * cos_theta_t)                       # p-polarized (TM)
    tp = (2 * n_inc * cos_theta_i) / (n_t * cos_theta_i + n_inc * cos_theta_t)

    R = np.abs(rs)**2
    T = 1 -  R

    return R,T
#===========================================================================================================



#===========================================================================================================
def fresnel_coefficients2(i, n, n_inc, n_t):
    i = np.array(i, dtype=float)
    n = np.array(n, dtype=float)

    # cos_theta_i = np.dot(i, n)     
    cos_theta_i = np.abs(np.dot(i, n))
    sin_theta_i2 = 1 - cos_theta_i**2
    sin_theta_t2 = (n_inc / n_t)**2 * sin_theta_i2

    if sin_theta_t2 > 1.0:
        # Total in_ternal reflection
        return 1,0
    cos_theta_t = np.sqrt(1 - sin_theta_t2)

    rs = (n_inc * cos_theta_i - n_t * cos_theta_t) / (n_inc * cos_theta_i + n_t * cos_theta_t)                      # s-polarized (TE)
    ts = (2 * n_inc * cos_theta_i) / (n_inc * cos_theta_i + n_t * cos_theta_t)
   
    rp = (n_t * cos_theta_i - n_inc * cos_theta_t) / (n_t * cos_theta_i + n_inc * cos_theta_t)                       # p-polarized (TM)
    tp = (2 * n_inc * cos_theta_i) / (n_t * cos_theta_i + n_inc * cos_theta_t)

    R = np.abs(rs)**2
    T = 1 -  R

    return rs, ts
#===========================================================================================================

#===========================================================================================================
def getAbsorption(rs, ts, tandels, n_diel, ray_lengths, sk, nk):
    abs_total = 0
    nrays_refl = 0
    absr = []
    pt_total = 0
    pr_total = 0
    for i in range(0, len(rs)):
        n_refl = len(rs[i]) - 1

        if n_refl > 2:
            nrays_refl += 1
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
            r2 = 0.2709 - 0.00307j
            t1 = t_te[0]
            t2 = t_te[1]
            t2 = 1.2703 - 0.00307j

            if i == 18:
                print('stop')

            for j in range(0, n_refl):
                r_tej = r_te[j]
                t_tej = t_te[j]
                theta_j = np.acos(np.abs(np.dot(nk[i][j+1], sk[i][j+1]))) #nos saltamos nk[0] porque es s0
                #print(np.rad2deg(theta_j))
                #attenuation = np.exp(-2*alfa*ray_len[j])
                if j == 0:
                    # An_rf = np.append(An_rf, r_tej)
                    An_rf = np.append(An_rf, r1)
                 #   An_rf[j] = r_tej

                else:    #if j % 2 != 0:              #for transmission handling
                    ray_len_t += ray_len[j]
                    if nk[i][j+1][2] >= 0 :           #for transmission handling, when z-component of normal is positive
                        if r_tej != 1:
                            num_trans += 1
                        #An_tr[j] = An_tr[j-1]*np.exp(-1j * kc * ray_len[j] * j) 
                            if num_trans <= 1: 
                                An_tr = np.append(An_tr, t_te[0]*t_te[j]*np.exp(-1j * kc * ray_len[j] * j * np.cos(theta_j)**2)) 
                                An_tr = np.append(An_tr, t1*t2*np.exp(-1j * kc * ray_len[j] * j * np.cos(theta_j)**2)) 

                            else: 
                                # An_tr = np.append(An_tr, r_tej**2*An_tr[-1]*np.exp(-1j * kc * (ray_len[j] + ray_len[j-1]) *np.cos(theta_j)**2))   #* np.cos(theta_j)**2)

                                An_tr = np.append(An_tr, (-r1*r2)**(num_trans-1)*(t1*t2)*np.exp(-1j * kc * (ray_len_t) *np.cos(theta_j)**2))
                                #An_tr[j] = r_tej*An_tr[j-2]*np.exp(-1j * kc * (ray_len[j] + ray_len[j-1]))
                                # An_tr = np.append(An_tr, r_tej**2*An_tr[-1]*np.exp(-1j * kc * (ray_len[j] + ray_len[j-1])* np.cos(theta_j)**2))
                            #ptr_aux += np.abs(An_tr[-1])**2
                    else:
                        num_refl += 1
                        if j <= 2: 
                            # An_rf =  np.append(An_rf, t_te[0]*t_tej * r_tej * np.exp(-1j * kc * (ray_len[j] + ray_len[j-1])* np.cos(theta_j)**2))   
                            An_rf =  np.append(An_rf, t1*t2 * r2 * np.exp(-1j * kc * (ray_len_t)* np.cos(theta_j)**2))
                        else: 
                            An_rf =  np.append(An_rf, t1*t2 * r2 *(-r2*r1)**(num_refl-1) * np.exp(-1j * kc * (ray_len_t)* np.cos(theta_j)**2))
                            # An_rf =  np.append(An_rf, An_rf[-1] * r_tej**2 * np.exp(-1j * kc * (ray_len[j] + ray_len[j-1])* np.cos(theta_j)**2))

               
            ptr_aux = sum(An_tr)
            pref_aux = sum(An_rf)
            pt = np.abs(ptr_aux)**2
            # pt = ptr_aux
            pr = np.abs(pref_aux)**2
            pt_total += pt
            pr_total += pr
            print(str(i) + '-ray and power T: '+ str(pt) + ' and power R: '+ str(pr)) 
    
    p_trans_t = pt_total/nrays_refl
    p_refl_t = pr_total/nrays_refl
    #p_abs_t = 1 - p_trans_t - p_refl_t
    p_abs_t = 1 - p_trans_t - p_refl_t
    print('Absorbed - ' + str(p_abs_t*100))
    print('Ptrans - ' + str(p_trans_t*100))
    print('Prefl - ' + str(p_refl_t*100))
    
    return p_abs_t
#===========================================================================================================
