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

