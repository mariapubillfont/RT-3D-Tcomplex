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
        return 1,0, 1, 0, 1j 
    cos_theta_t = np.sqrt(1 - sin_theta_t2)

    rs = (n_inc * cos_theta_i - n_t * cos_theta_t) / (n_inc * cos_theta_i + n_t * cos_theta_t)                      # s-polarized (TE)
    ts = (2 * n_inc * cos_theta_i) / (n_inc * cos_theta_i + n_t * cos_theta_t)
   
    rp = (n_inc * cos_theta_t - n_t * cos_theta_i) / (n_t * cos_theta_i + n_inc * cos_theta_t)                       # p-polarized (TM)
    tp = (2 * n_inc * cos_theta_i) / (n_t * cos_theta_i + n_inc * cos_theta_t)

    R = np.abs(rs)**2
    T = 1 -  R

    return rs, ts, rp, tp, cos_theta_t
#===========================================================================================================



def cos_angle(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))



def get_Pabs(At_te, Ar_te, At_tm, Ar_tm, Ak, Ak_src, Ei_te, Ei_tm, nk, sk, dLk, dLk_ap):
    pi_t, pr_t, pt_t= [0, 0, 0]
    nrays_refl = len(At_te)
    n_a = 377
    n_b = 377
    # theta_a = 0
    pi_te, pi_tm, pr_te, pr_tm, pt_te, pt_tm =[ 0, 0, 0, 0, 0, 0]
    R, T, A = [0, 0, 0]
    norm = 0
    R_k = []
    T_k = []
    A_k = []

    for i in range(0, len(At_te)-2):
        k = i + 1
        Ak[i] = 1
        pt_te_aux = 0
        pt_tm_aux = 0
        pr_te_aux = 0
        pr_tm_aux = 0
        pt_te_aux = sum(At_te[k])#*Ak[i]
        pt_tm_aux = sum(At_tm[k])#*Ak[i]
        pr_te_aux = sum(Ar_te[k])#*Ak[i]
        pr_tm_aux = sum(Ar_tm[k])#*Ak[i]

        pt_total_te += np.abs(pt_te_aux)**2
        pr_total_te += np.abs(pr_te_aux)**2
        pt_total_tm += np.abs(pt_tm_aux)**2
        pr_total_tm += np.abs(pr_tm_aux)**2

        theta_a = np.abs(cos_angle(nk[k][0], sk[k][0]))
        theta_b = np.abs(cos_angle(nk[k][-2], sk[k][-2]))
        Ei_tei = Ei_te[k]
        Ei_tmi = Ei_tm[k]

        pi_te = np.abs(Ei_tei)**2/(2*n_a*theta_a)#*np.abs(Ak[i])**2
        pi_tm = np.abs(Ei_tmi)**2*theta_a/(2*n_a)#*np.abs(Ak[i])**2
        pr_te = np.abs(pr_te_aux)**2*np.abs(Ei_tei)**2/(2*n_a*theta_a)
        pr_tm = np.abs(pr_tm_aux)**2*np.abs(Ei_tmi)**2*theta_a/(2*n_a)
        pt_te = np.abs(pt_te_aux)**2*np.abs(Ei_tei)**2/(2*n_b*theta_b)
        pt_tm = np.abs(pt_tm_aux)**2*np.abs(Ei_tmi)**2*theta_b/(2*n_b)

        pi_t = pi_te + pi_tm
        pr_t = pr_te + pr_tm
        pt_t = pt_te + pt_tm

        R += pr_t/pi_t
        T += pt_t/pi_t
        A += 1 - pr_t/pi_t - pt_t/pi_t
        # norm += A0_ray[i]
        print(pr_te/pi_te*100, pt_te/pi_te*100)
        print(pr_t/pi_t*100, pt_t/pi_t*100)

        norm += np.abs(Ak[i])**2
        # print(norm, np.abs(A0_ray[i])**2)
    # R = pr_t/pi_t
    # T = pt_t/pi_t
    # A = 1 - R - T
    p_trans_te = pt_total_te/norm
    p_refl_te = pr_total_te/norm
    p_abs_te = 1 - p_trans_te - p_refl_te
    p_trans_tm = pt_total_tm/norm
    p_refl_tm = pr_total_tm/norm
    p_abs_tm = 1 - p_trans_tm - norm
    print('Absorbed MAIN - ' + str(p_abs_te*100))
    print('Absorbed omni - ' + str(p_abs_te*100/(I.Nrays)*nrays_refl))
    print('Ptrans MAIN - ' + str(p_trans_te*100))
    print('Prefl MAIN - ' + str(p_refl_te*100))
    print((pi_t - pt_t - pr_t)*100/pi_t)
    return 1 #p_trans_te, p_refl_te, p_abs_te, p_trans_tm, p_refl_tm, p_abs_tm





def get_Pabs2D(At_te, Ar_te, At_tm, Ar_tm, Ak, Ak_src, Ei_te, Ei_tm, nk, sk, dLk, dLk_ap):

    pi_t, pr_t, pt_t= [0, 0, 0]
    nrays_refl = len(At_te)
    n_a = 377
    n_b = 377
    # theta_a = 0
    pi_te, pi_tm, pr_te, pr_tm, pt_te, pt_tm =[ 0, 0, 0, 0, 0, 0]
    R, T, A = [0, 0, 0]
    norm = 0
    R_k = []
    T_k = []
    A_k = []

    for i in range(0, len(At_te)-2):
        k = i + 1
        pt_te_aux, pt_tm_aux, pr_te_aux, pr_tm_aux = [0, 0, 0, 0]
        pt_te_aux = sum(At_te[k])#*Ak[i]
        pt_tm_aux = sum(At_tm[k])#*Ak[i]
        pr_te_aux = sum(Ar_te[k])#*Ak[i]
        pr_tm_aux = sum(Ar_tm[k])#*Ak[i]

        theta_a = np.abs(cos_angle(nk[k][0], sk[k][0]))
        theta_b = np.abs(cos_angle(nk[k][-2], sk[k][-2]))
        Ei_tei = Ei_te[k]*np.abs(Ak[i])
        Ei_tmi = Ei_tm[k]*np.abs(Ak[i])

        pi_te = np.abs(Ei_tei)**2/(2*n_a*theta_a)#*np.abs(Ak[k])**2
        pi_tm = np.abs(Ei_tmi)**2*theta_a/(2*n_a)#* np.abs(Ak[k])**2
        pr_te = np.abs(pr_te_aux)**2*np.abs(Ei_tei)**2/(2*n_a*theta_a)
        pr_tm = np.abs(pr_tm_aux)**2*np.abs(Ei_tmi)**2*theta_a/(2*n_a)
        pt_te = np.abs(pt_te_aux)**2*np.abs(Ei_tei)**2/(2*n_b*theta_b)
        pt_tm = np.abs(pt_tm_aux)**2*np.abs(Ei_tmi)**2*theta_b/(2*n_b)

        Si_k = pi_te + pi_tm
        Sr_k = pr_te + pr_tm
        St_k = pt_te + pt_tm

        dl_ap = dLk_ap[i]   # el que ya calculas en calculateRayTubeAmpl
        Pi_k = Si_k * dl_ap
        Pr_k = Sr_k * dl_ap
        Pt_k = St_k * dl_ap

        R_k = np.append(R_k, Pr_k)
        T_k = np.append(T_k , Pt_k)
        A_k = np.append(A_k, Pi_k)

        # R_k = np.append(R_k, Pr_k/Pi_k)
        # T_k = np.append(T_k , Pt_k/Pi_k)
        # A_k = np.append(A_k, 1 - R_k[-1] - T_k[-1])
        # A += 1 - R - T
        # norm += A0_ray[i]
    # R_tot = sum(R_k)
    # T_tot = sum(T_k)
    R_total = sum(R_k)/sum(A_k) 
    T_total = sum(T_k)/sum(A_k)    # norm += np.abs(Ak[i])**2
    A_total = 1 - R_total - T_total    # print(norm, np.abs(A0_ray[i])**2)
    A_tnorm = A_total*nrays_refl/I.Nrays
    print(A_tnorm)
    return 1


    # R = pr_t/pi_t
    # T = pt_t/pi_t
    # A = 1 - R - T
    # p_trans_te = pt_total_te/norm
    # p_refl_te = pr_total_te/norm
    # p_abs_te = 1 - p_trans_te - p_refl_te
    # p_trans_tm = pt_total_tm/norm
    # p_refl_tm = pr_total_tm/norm
    # p_abs_tm = 1 - p_trans_tm - norm
    # print('Absorbed MAIN - ' + str(p_abs_te*100))
    # print('Absorbed omni - ' + str(p_abs_te*100/(I.Nrays)*nrays_refl))
    # print('Ptrans MAIN - ' + str(p_trans_te*100))
    # print('Prefl MAIN - ' + str(p_refl_te*100))
    # print((pi_t - pt_t - pr_t)*100/pi_t)

    #p_trans_te, p_refl_te, p_abs_te, p_trans_tm, p_refl_tm, p_abs_tm


#===========================================================================================================
def getAbsorption(rs, ts, tandels, n_diel, ray_lengths, sk, nk, theta_ts):
    abs_total = 0
    nrays_refl = 0
    absr = []
    pt_total = 0
    pr_total = 0
    for i in range(0, len(rs)):
        n_refl = len(rs[i]) - 1
       

        # if n_refl > 1:
            
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
        


        if r_te[1] == 1:
            print('hola')
        
        nrays_refl += 1
        for j in range(0, n_refl):
            r_tej = r_te[j]
            t_tej = t_te[j]
            # t2 = t_tej
            # r2 = r_tej
            cos_trj = np.acos(np.abs(np.dot(nk[i][j+1], sk[i][j+1]))) #esto es el angulo theta_in
            cos_trj = cost[0]
            if r_tej == 1:
                print('rte 1')
            
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
            print(An_tr)
               
        ptr_aux = sum(An_tr)
        pref_aux = sum(An_rf)
        pt = np.abs(ptr_aux)**2
        pr = np.abs(pref_aux)**2
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

    
    return p_abs_t
#===========================================================================================================




# ===========================================================================================================
def getAbsorption2(r_tes, t_tes, r_tms, t_tms, tands, ns, r_len, sk, nk, theta_ts):
    nrays_refl = 0
    pt_total = 0
    pr_total = 0
    for i in range(0, len(r_tes)):
        n_refl = len(r_tes[i]) - 1
        keep_refl = True
        
        if n_refl > 2:
            nrays_refl += 1
            
            ray_len_t = 0
            num_trans = 0
            num_refl = 0
            r_te = r_tes[i]
            t_te = t_tes[i]
            r_tm = r_tms[i]
            t_tm = t_tms[i]
            ski = sk[i]
            nki = nk[i]
            tandel = tands[i][0]
            nr = ns[i][0]
            kc = I.k0*nr*np.sqrt(1-complex(0,tandel))
            alpha = np.imag(kc)

            Ar_te = []
            Ar_te2 = []
            At_te = []
            At_te2 = []
            Ar_tm = []
            At_tm = []

            pt_te_aux = 0
            pt_tm_aux = 0
            pr_te_aux = 0
            pr_tm_aux = 0

            alpha = np.imag(kc)

            # if r_te[1] == 1:   #first reflection is total reflextion
            #     continue
            
            for j in range(0,6):
            # while keep_refl:
                # e_perp, e_pll = field_decomposition(ski[j], nki[j])                                 #Decompose the field in parallel and perpendicular component
                cos_trj = np.acos(np.abs(np.dot(nk[i][j+1], sk[i][j+1]))) #nos saltamos nk[0] porque es s0
                cos_trj = theta_ts[i][0]

                # cos_trj = 1

                if r_te[j] == 1:   #first reflection is total reflextion
                    print('total reflection in ray ' + str(i) + ', and reflection num. ' + str(j))
                
                if j == 0:
                    Ar_te = np.append(Ar_te, r_te[0])
                    Ar_te2 = np.append(Ar_te2, r_te[0])
                    Ar_tm = np.append(Ar_tm, r_tm[0])

                else:                 
                    ray_len_t += r_len[i][j]
                    if nk[i][j+1][2] >= 0 :           #for transmission handling, when z-component of normal is positive
                        if r_te[j] != 1:
                            num_trans += 1
                            At_te = np.append(At_te, (-r_te[0]*r_te[1])**(num_trans-1)*(t_te[0]*t_te[1])*np.exp(-1j * kc * (ray_len_t) *(cos_trj)*np.abs(cos_trj)))
                            At_tm = np.append(At_tm, (-r_tm[0]*r_tm[1])**(num_trans-1)*(t_tm[0]*t_tm[1])*np.exp(-1j * kc * (ray_len_t) *(cos_trj)*np.abs(cos_trj)))
                            # At_te2 = np.append(At_te2, (-r_te[0]*r_te[1])**(num_trans-1)*(t_te[0]*t_te[1])*np.exp(-j * kc * ray_len_t * alpha))
                            # At_tm = np.append(At_tm, (-r_tm[0]*r_tm[1])**(num_trans-1)*(t_tm[0]*t_tm[1])*np.exp(-1j * kc * (cos_trj) * 6e-3))
                    else:
                        num_refl += 1
                        Ar_te =  np.append(Ar_te, t_te[0]*t_te[1] * r_te[1] * (-r_te[0]*r_te[1])**(num_refl-1) * np.exp(-1j * kc * (ray_len_t)* (cos_trj)*np.abs(cos_trj)))
                        Ar_tm =  np.append(Ar_tm, t_tm[0]*t_tm[1] * r_tm[1] * (-r_tm[0]*r_tm[1])**(num_refl-1) * np.exp(-1j * kc * (ray_len_t)* (cos_trj)*np.abs(cos_trj)))
                        # Ar_te2 =  np.append(Ar_te2, t_te[0]*t_te[1] * r_te[1] * (-r_te[0]*r_te[1])**(num_refl-1) * np.exp(-2j * kc * ray_len_t * alpha))
                        # Ar_tm =  np.append(Ar_tm, t_tm[0]*t_tm[1] * r_tm[1] * (-r_tm[0]*r_tm[1])**(num_refl-1) * np.exp(-1j * kc *  (cos_trj)*6e-3))


            pt_te_aux = sum(At_te)
            pt_tm_aux = sum(At_tm)
            pr_te_aux = sum(Ar_te)
            pr_tm_aux = sum(Ar_tm)

            pt = np.abs(pt_te_aux)**2
            pr = np.abs(pr_te_aux)**2
            pt_total += pt
            pr_total += pr
    
    p_trans_t = pt_total/nrays_refl
    p_refl_t = pr_total/nrays_refl
    p_abs_t = 1 - p_trans_t - p_refl_t
    print('Absorbed v2 - ' + str(p_abs_t*100))
    print('Absorbed omni - ' + str(p_abs_t*100/(I.Nrays)*nrays_refl))
    print('Ptrans v2 - ' + str(p_trans_t*100))
    print('Prefl v2 - ' + str(p_refl_t*100))
            
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
def field_decomposition(sk, nk, Ex, Ey, Ez):
    Ei_te = np.zeros(len(sk), dtype=complex)
    Ei_tm = np.zeros(len(sk), dtype=complex)
    
    for i in range(0, len(sk)):

        k_hat = unit(sk[i][0])
        n_hat = unit(nk[i][0])
        Ei = [Ex[i], Ey[i], Ez[i]]

        # Formula: ê_parallel = k̂ × ( n̂ × k̂ ) / | k̂ × ( n̂ × k̂ ) |
        cross_inner = np.cross(n_hat, k_hat)
        cross_outer = np.cross(k_hat, cross_inner)
        norm_cross_outer = np.linalg.norm(cross_outer)

        e_par = cross_outer / norm_cross_outer
        Ei_tm[i] = np.dot(Ei, e_par)

        # Formula: ê_perp = - ( ê_parallel × k̂ )
        e_perp = -np.cross(e_par, k_hat)
        Ei_te[i] = np.dot(Ei, e_perp)

    return Ei_te, Ei_tm
#===========================================================================================================