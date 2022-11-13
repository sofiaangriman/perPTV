import numpy as np
import pandas as pd
import glob
import os
from tqdm import tqdm
from skimage import measure

#---------------------------------------------------------
def crear_lbl_parts(lista_df):
    lbls_part = [s[1:] for s in lista_df[1::2]]
    return lbls_part

#---------------------------------------------------------
def convertir_lista_parts(lista):
    lbls_part = []
    for n in range(len(lista)):
        lbls_part.append('x'+lista[n])
        lbls_part.append('y'+lista[n])
    return lbls_part

#---------------------------------------------------------
def dist_z12(z1, z2, maxdif_len):
    '''
    Calcula la desviación estándar de la distancia L2 de dos vectores z1 y z2.
    z1 y z2 tienen el mismo largo, pero diferente cantidad de valores no NaN.
    El cálculo se hace sólamente cuando la diferencia de valores no NaN en ambos
    arrays es menor o igual a 'maxdif_len' (en ese caso devuelve NaN).

    También devuelve un vector de booleanos, donde valores True corresponden a
    los índices donde z1 y z2 son *no NaN* en simultáneo.

    '''
    nans_1 = np.isnan(z1)
    nans_2 = np.isnan(z2)

    L1 = np.sum(~nans_1)
    L2 = np.sum(~nans_2)

    D = np.abs(L1 - L2)

    if D<=maxdif_len:
        idx_12 = (~nans_1)*(~nans_2)
        idx_12 = idx_12.astype('bool')
        N = np.sum(idx_12)

        if N >= 5:
            M = N
            S = np.sqrt(np.std((z1[idx_12][:M] - z2[idx_12][:M])**2))
            S = S/np.sqrt(N)
        else:
            S = np.nan
            N = 0
    else:
        S = np.nan
        idx_12 = np.zeros_like(z1).astype('bool')
        N = 0
    return S, idx_12, N

#---------------------------------------------------------
def distancia_df(df_X, df_Y, maxdif_len=5, dist_umbral = 50):
    '''
    Calcula la distancia entre trayectorias de dos DataFrames 2D
    df_X --> contiene coordenadas x, z1
    df_Y --> contiene coordenadas y, z2

    Se calcula la distancia entre z1 y z2 (coordenada redundante)
    para *todos* los pares de trayectorias que tengan una diferencia de largo
    <= maxdif_len (ver función df_dist)

    Estas distancias se almacenan en un DataFrame 'df_dist', que tiene
    como etiquetas:
            - columnas --> etiquetas del df_X
            - filas    --> etiquetas del df_Y
    También se almacenan en el DataFrame 'df_idx' los índices donde z1 y z2
    son no NaN en simultáneo. 
    '''
    lbls_1 = crear_lbl_parts(df_X.columns)
    lbls_2 = crear_lbl_parts(df_Y.columns)


    df_dist = pd.DataFrame(columns=lbls_1, index=lbls_2, dtype=np.float32)
    df_idx  = pd.DataFrame(columns=lbls_1, index=lbls_2, dtype='object')
    df_N    = pd.DataFrame(columns=lbls_1, index=lbls_2, dtype='int')
    for n in tqdm(range(len(lbls_1)), desc='distancia_df', leave=False):
        lbl_n = lbls_1[n]
        z1 = df_X.loc[:,'y'+lbl_n]

        nac = z1.first_valid_index()
        mue = z1.last_valid_index()

        coexistentes = (~df_Y.loc[nac:mue, :].isnull()).sum(axis=0).astype(bool)
        cols_coex    = df_Y.columns[coexistentes.values]

        z_otras = df_Y.loc[nac:mue, cols_coex[1::2]].copy()
        temp    = z1.values
        temp    = temp[nac:mue + 1]
        temp    = np.reshape(temp, (temp.size, 1))

        z_otras.loc[:, :] -= temp
        z_otras = np.abs(z_otras)

        cercanas = (np.nanmax(z_otras, axis=0) < dist_umbral)

        XY_cercanas = cols_coex[np.repeat(cercanas, 2)]

        z1 = z1.values.astype(float)
        for m in range(len(XY_cercanas)//2):
            lbl_m = XY_cercanas[2*m][1:]

            z2 = df_Y.loc[:,'y'+lbl_m].values.astype(float) 

            S, idx_12, N = dist_z12(z1, z2, maxdif_len)
            df_dist.loc[lbl_m, lbl_n] = S
            df_idx.loc[ lbl_m, lbl_n] = idx_12
            df_N.loc[   lbl_m, lbl_n] = int(N)

    return df_dist, df_idx, df_N

#---------------------------------------------------------
def get_df_3D(df_X = [], df_Y = [], maxdif_len=100, dist_umbral = 50, df_3D_prev = pd.DataFrame()):
    '''
    Compute 3D DataFrame from 2D DataFrames.
    In:
            - df_X          --> coordinates (x,z) of particles
            - df_Y          --> coordinates (y,z) of particles
            - maxdif_len    --> max. value of length difference between two curves
            - dist_umbral   --> distance threshold for finding candidates
    '''
    # First compute the distance between all co-existing trajectories
    # that are close enough.
    df_dist, df_idx, df_N = distancia_df(df_X, df_Y, maxdif_len=maxdif_len, dist_umbral = dist_umbral)
    
    # If a pair of curves coexist for one frame or less, discard the pair.
    df_dist.iloc[(df_N <= 1).values] = np.nan

    df_dist = df_dist.dropna(how='all', axis='columns')
    df_dist = df_dist.dropna(how='all', axis='index')
    
    PX = df_dist.columns
    PY = df_dist.index

    MATCHES = []

    for n in tqdm(range(len(PX)), desc='get DF 3D', leave=False, disable=False):
        P_n = PX[n]
        # All trajectories (candidates)
        # in df_Y that could match with P_n
        C_n = list(df_dist[P_n].dropna().index)

        # For all the trajs. C_n, find those alive simultaneously
        alive_sim = find_if_alive_sim(df_idx, P_n, C_n)

        parts_to_discard = []
        parts_to_match   = []
        for m in range(len(C_n)):
            C = C_n[m]
            if C not in parts_to_discard: 
                # for each C_n find the 'best' among those
                # that are alive simultaneously
                C_nm = list(alive_sim[C].index[alive_sim[C].values])
                C_nm = list(set(C_nm).difference(parts_to_discard))
                L = [C]
                L.extend(C_nm)

                C_to_match = str(df_dist.loc[L, P_n].idxmin())
                if C_to_match not in parts_to_match:
                    parts_to_match.append(C_to_match)
                L.remove(C_to_match)
                parts_to_discard.extend(L)

        df_dist.loc[parts_to_discard, P_n] = np.nan

        # now check if for all trajs. in 'parts_to_match'
        # P_n is also the 'best' for a given interval.
        for m in range(len(parts_to_match)):
            P_m = parts_to_match[m]

            C_m                     = df_dist.loc[P_m,:].dropna().index
            alive_sim               = find_if_alive_sim(df_idx.T, P_m, C_m)
            parts_to_discard        = []
            parts_already_matched   = []
            for C in C_m:
                if C not in parts_to_discard and\
                C not in parts_already_matched:
                    C_nm = list(alive_sim[C].index[alive_sim[C].values])
                    L = [C]
                    L.extend(C_nm)
                    best_cand = df_dist.loc[P_m, L].idxmin()
                    L.remove(best_cand)
                    parts_to_discard.extend(L)
                    if best_cand == P_n:
                        MATCHES.append((P_n, P_m))
                        parts_already_matched.append(best_cand)

            df_dist.loc[P_m, parts_to_discard] = np.nan

    #------------
    # Create particles' labels for df_3D.

    label_prev = df_3D_prev.shape[1]//3
    lbls = ['P' + str(n) for n in range(1 + label_prev, len(MATCHES)+1 + label_prev) ]
    cols = []
    [cols.extend(['x'+s, 'y'+s, 'z'+s]) for s in lbls];
    
    time_idx = df_X.index
    df_tot = pd.DataFrame(index=time_idx, columns=cols)
    
    for n in range(len(MATCHES)):
        # Each element in MATCHES is a tuple containing
        # pairs of (Px, Py)
        Plbl   = lbls[n]
        Px, Py = MATCHES[n]
        idx_n = df_idx.loc[Py, Px]
        time_n = time_idx[idx_n]
        df_tot.loc[time_n, 'x'+Plbl] = df_X.loc[time_idx[idx_n], 'x'+Px]
        df_tot.loc[time_n, 'y'+Plbl] = df_Y.loc[time_idx[idx_n], 'x'+Py]
        df_tot.loc[time_n, 'z'+Plbl] = (df_X.loc[time_idx[idx_n], 'y'+Px] + df_Y.loc[time_idx[idx_n], 'y'+Py])/2

        # Also clean these pairs in the original 2D DataFrames
        df_X.loc[idx_n, 'x'+Px] = np.nan
        df_X.loc[idx_n, 'y'+Px] = np.nan

        df_Y.loc[idx_n, 'x'+Py] = np.nan
        df_Y.loc[idx_n, 'y'+Py] = np.nan

    df_tot = pd.concat([df_3D_prev, df_tot], axis='columns')

    return df_tot

#---------------------------------------------------------
def split_DFs_mindist(df_X = [], df_Y = [], dist_umbral = 50):
    '''
    For each frame find if potential matches exist based on
    distance criterion. If no possible match exists,
    split the 2D trajectory into segments.
    '''
    Nframes = df_X.shape[0]
    for n in tqdm(range(Nframes), leave=False, 
                desc='Finding parts. to remove in 2D. . .'):
        z1 = df_X.iloc[n, 1::2].values.flatten()
        z2 = df_Y.iloc[n, 1::2].values.flatten()

        dist = np.abs(z1[None,:] - z2[:, None])	
        # fixed column --> fixed part in df_X
        # fixed row    --> fixed part in df_Y
        dist[dist >= dist_umbral] = np.nan
        dist_nonan = ~np.isnan(dist)

        parts_X_to_remove = (np.sum(dist_nonan, axis=0) == 0)
        parts_Y_to_remove = (np.sum(dist_nonan, axis=1) == 0)

        df_X.iloc[n, np.repeat(parts_X_to_remove,2)] = np.nan
        df_Y.iloc[n, np.repeat(parts_Y_to_remove,2)] = np.nan
    # After checking if any compatible pairs coexist,
    # split each 2D trajectory 
    df_X = df_X.dropna(axis='columns', how='all')
    df_Y = df_Y.dropna(axis='columns', how='all')

    df_X = find_islands_and_split_DF(df_X)
    df_Y = find_islands_and_split_DF(df_Y)

    df_X = df_X.dropna(axis='columns', how='all')
    df_Y = df_Y.dropna(axis='columns', how='all')

    return df_X, df_Y
#---------------------------------------------------------
def find_islands_and_split_DF(DF_orig):
    '''
    For a 2D DataFrame with NaNs in between non-NaN values,
    find non-Nan `islands' and split the trajectories.
    (leading and trailing NaNs are not considered).
    '''
    DF       = DF_orig.copy()
    PARTS_ID = sorted([int(elem[2:]) for elem in DF.columns[::2]])
    N        = len(PARTS_ID)
    for n in tqdm(range(N), leave=False, desc='Splitting 2D DataFrame'):
        X = DF.loc[:, 'xP'+str(PARTS_ID[n])]
        Y = DF.loc[:, 'yP'+str(PARTS_ID[n])]
        idx   = ~np.isnan(X)
        label, Nreg = measure.label(idx, return_num=True)
        for m in range(1, Nreg):
            new_PART_ID = PARTS_ID[-1] + 1
            PARTS_ID.append(new_PART_ID)
            DF.loc[:,'xP'+str(new_PART_ID)] = np.nan
            DF.loc[:,'yP'+str(new_PART_ID)] = np.nan

            idx_label = label == (m+1)
            DF['xP'+str(new_PART_ID)][idx_label] = X[idx_label].values
            DF['yP'+str(new_PART_ID)][idx_label] = Y[idx_label].values

        DF['xP' + str(PARTS_ID[n])][label!=1] = np.nan
        DF['yP' + str(PARTS_ID[n])][label!=1] = np.nan
    return DF

#---------------------------------------------------------
def find_if_alive_sim(DF_idx, P_X, P_Y):
    '''
    Find if particles 'P_Y' are alive simultaneously while
    'P_X' is still alive.
    '''
    DF_a = pd.DataFrame(False, index=P_Y, columns=P_Y, dtype='bool')
    for n in range(len(P_Y)):
        for m in range(n+1, len(P_Y)):
            val = np.sum(DF_idx.loc[P_Y[n], P_X] * DF_idx.loc[P_Y[m], P_X]).astype(bool)
            DF_a.iloc[m,n] = val
            DF_a.iloc[n,m] = val
    return DF_a

#---------------------------------------------------------
def write_params_3D(output_path, DIR = [], maxdif_len = [], dist_umbral = [], px_to_m_C1 = [], px_to_m_C2 = []):
    '''
    Writes to output directory parameters used for 3D matching.
    '''
    f = open(os.path.join(output_path, 'params_3D.txt'), 'w+')
    f.write(f'DIR\t{DIR}');
    f.write(f'\npx_to_m_C1\t{px_to_m_C1}');
    f.write(f'\npx_to_m_C2\t{px_to_m_C2}');
    f.write(f'\nmaxdif_len\t{maxdif_len}');
    f.write(f'\ndist_umbral\t{dist_umbral}\n');
    f.close()
    return None
