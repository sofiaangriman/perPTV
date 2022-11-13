"""
Object detection and tracking in time
from a set of 2D images.

"""
import os
import cv2
import glob
import yaml
import numpy as np
import pandas as pd
from tqdm import tqdm
from skimage import measure, filters, io, feature
from skimage.transform import hough_circle, hough_circle_peaks
from skimage.morphology import disk, remove_small_objects
from scipy.ndimage import convolve
from scipy.interpolate import InterpolatedUnivariateSpline, UnivariateSpline
from scipy.spatial.distance import cdist
from scipy.signal import correlate2d

# -----------------------------------
def analyze_by_PTV(folder, tol = 20, max_nan = 5, min_frames = 2,\
                   writing_threshold = 20,\
                   use_predictive=True, use_curvature_history=False,\
                   print_progress=True):
    """
    Detects objects and reconstructs trajectories in 2D.
    Args:
        - folder                : str; path to PTV_parameter.yaml file.
        - tol                   : float; proximity tolerance for finding possible
                                  matches between objects in successive frames.
        - max_nan               : int; trajectories are kept alive synthethically 
                                  for a maximum of 'max_nan' frames.
        - min_frames            : int; minimum number of frames required for
                                  extrapolating trajectories.
        - writing_threshold     : int; maximum number of "dead" trajectories kept
                                  in memory before writing to disk.
        - use_predictive        : bool; extrapolate trajectories before finding
                                  best candidate when tracking. Defaults to True.
        - use_curvature_history : bool; compare the hystorical curvature of
                                  the trajectories when picking best match.
                                  Defaults to False.
        - print_progress        : bool; print a progress bar.
        
    Returns:
        - df_objects            : pandas DataFrame with detected objects.
        - df_trajectories       : pandas DataFrame with 2D trajectories.
        - datasetID             : dataset ID.

    """
    # First create dataset for PTV analysis
    d1, d2, particulas_por_frame, df_objects, datasetID,\
    output_path = analizar_set_datos(folder, print_progress=print_progress)

    # Write in output directory the parameters used for trajectory matching
    write_matching_params(output_path, tol = tol, max_nan = max_nan,\
                          min_frames = min_frames,\
                          writing_threshold = writing_threshold,\
                          use_predictive = use_predictive,\
                          use_curv_hist = use_curvature_history)
    #---------------------------------------------------------
    # Write the objects DataFrame df_objects to output directory
    df_objects.to_pickle(os.path.join(output_path, 'dforig.out'))

    # Run PTV analysis and generate a trajectories dataset
    df_trajectories = build_trajectories(df_objects, proximity_tolerance = tol,\
                      use_predictive = use_predictive,\
                      use_curvature_history = use_curvature_history,\
                      max_number_nans = max_nan,\
                      min_number_frames = min_frames,\
                      umbral_trayectorias_muertas = writing_threshold,\
                      output_path = output_path,\
                      datasetID = datasetID,\
                      print_progress=print_progress)

    return df_objects, df_trajectories, datasetID

# -----------------------------------
def write_matching_params(output_path,\
                          tol = [],\
                          max_nan = [],\
                          min_frames = [],\
                          writing_threshold = [],\
                          use_predictive = [],\
                          use_curv_hist = []):
    """
    Writes to output directory parameters used for
    2D trajectory matching.
    """
    f = open(os.path.join(output_path, 'trajectories_params.txt'), 'w+')
    f.write('tol\t{}'.format(tol));
    f.write('\nmax_nan\t{}'.format(max_nan));
    f.write('\nmin_frames\t{}'.format(min_frames));
    f.write('\nwriting_threshold\t{}'.format(writing_threshold));
    f.write('\nuse_predictive\t'+str(use_predictive));
    f.write('\nuse_curv_hist\t'+str(use_curv_hist)+'\n');
    f.close()
    return None

# -----------------------------------
def analizar_set_datos(folder, print_progress=True):
    """
    Builds a dataframe with position of objects found in image sequence, 
    without attempting to establish time-correspondence (i.e., no trajectory
    calculation yet!). 
    """

    path_archivo_parametros = os.path.join(folder, "PTV_parameter.yaml")
    param = yaml.load(
                    open(os.path.abspath(path_archivo_parametros)), Loader=yaml.FullLoader
                    )

    path_mediciones = param["RAW_IMAGE_PATH"]
    path_fondo  = param["PTV"]["BACKGROUND"]["BACKGROUND_PATH"] 
    output_path = param["OUTPUT"]["PATH"]
    path_camara = param['CALIBRATION']['CALIBRATION_PATH']
    usar_calibracion = param['CALIBRATION']["USE_CALIBRATION"]
    xmin = int(param["PTV"]["ROI"]["CMIN"])
    xmax = int(param["PTV"]["ROI"]["CMAX"])
    ymin = int(param["PTV"]["ROI"]["RMIN"])
    ymax = int(param["PTV"]["ROI"]["RMAX"])
    dmin = int(param["PTV"]["MINIMUM_DIAMETER"])
    dmax = int(param["PTV"]["MAXIMUM_DIAMETER"])
    first_img = int(param["PTV"]["PROCESSING"]["FIRST_IMAGE"])
    last_img  = int(param["PTV"]["PROCESSING"]["LAST_IMAGE"])
    f_first   = int(param["PTV"]["BACKGROUND"]["FIRST_IMAGE"])
    f_last    = int(param["PTV"]["BACKGROUND"]["LAST_IMAGE"])
    use_inerp = int(param['PTV']['USE_INERP'])
    # extension = param['EXTENSION_IMAGENES']
    extension = "tif"
    threshold_value = float(param["PTV"]["BINARIZATION_THRESHOLD"])
    gamma_value = param["PTV"]["GAMMA_VALUE"]
    # idx_ryt = param['INDICE_RyT']
    idx_ryt = 4

    # -----
    if usar_calibracion is True:
        cam_mtx, roi, mapx, mapy,\
        rvecs, tvecs, mean_error = cargar_calibracion(path_camara, idx_ryt)
        f_de_desdistorsion = undistort_image
        f_de_mapear = mapear
    else:
        f_de_desdistorsion = no_undistort
        mapx, mapy, cam_mtx, rvecs, tvecs = None, None, None, None, None
        f_de_mapear = no_mapear

    # -----
    if gamma_value:
        apply_gamma_correction = change_gamma
        gamma_value = float(gamma)
    if not gamma_value:
        apply_gamma_correction = not_change_gamma

    # /sigma/ and /use_DIC/ --> hard coded values!
    # Relevant if detecting large particles
    sigma   = 0.1
    use_DIC = False
    if use_inerp == 1:
        circle_detection = detectar_circulos_inerp
    if use_inerp == 0:
        circle_detection = detectar_circulos

    # Checks if a background file exists. If it does not,
    # it computes it and saves it.
    fondo = Fondo(path_fondo, f_first, f_last, extension)
    fondo = apply_gamma_correction(fondo, gamma_value)
    files = sorted(glob.glob(os.path.join(path_mediciones, "*." + extension)))
    Num   = last_img - first_img
    Num  += 1

    # ----------------
    # Object detection
    data = []

    for n in tqdm(range(Num), disable=~np.array(print_progress), desc='Object detection...'):
        img = io.imread(files[n + first_img - 1])
        img = apply_gamma_correction(img, gamma_value)
        img = np.abs(img - fondo)
        img = aplicar_mascara(img, xmin, xmax, ymin, ymax)
        thr = np.std(img) * threshold_value
        img = f_de_desdistorsion(img, mapx, mapy)
        img = binarizar(img, thr)
        a   = circle_detection(img, dmin, dmax, sigma=sigma, use_DIC=use_DIC)
        data.append(a)

    data_map = f_de_mapear(data, cam_mtx, rvecs, tvecs)

    # Create pandas DataFrame to store 
    # position of detected objects
    Nframes = Num
    particulas_por_frame = [len(elemento[:][0]) for elemento in data_map]
    max_nro_particulas = np.max(particulas_por_frame)
    # generamos los indices de filas y columnas
    # indice_frames = np.arange(Nframes)
    indice_frames = np.arange(first_img -1, last_img)
    indice_columnas = ["Npart"]
    for i in range(1, int(max_nro_particulas) + 1):
        indice_columnas.append("xP" + str(i))
        indice_columnas.append("yP" + str(i))
    # generamos el dataframe vacio, con NaNs en todos sus elementos
    df = pd.DataFrame(index=indice_frames, columns=indice_columnas, dtype=float)
    # llenamos con lo hallado (que esta necesariamente 'desordenado'!)

    for i, tiempo in enumerate(indice_frames):
        for j in range(particulas_por_frame[i]):
            # la primera columna tiene el numero de particulas en el frame
            df.at[tiempo, indice_columnas[0]] = particulas_por_frame[i]
            # asignamos las componentes x y y de cada particula en cada frame:
            # la coordenada x (notar el '0')
            df.at[tiempo, indice_columnas[2 * j + 1]] = np.asscalar(data_map[i][0][j]) + xmin
            # la coordenada y (notar el '1')
            df.at[tiempo, indice_columnas[2 * j + 2]] = np.asscalar(data_map[i][1][j]) + ymin

    # Generar el datasetID
    # la fecha esta en YYYY/MM/DD, la llevamos a: YYYYMMDD
    fecha = param["LOGGING_INFO"]["DATE"].split("/")
    fecha = [str(item).zfill(2) for item in fecha]
    fecha = "".join(fecha)
    # solo retenemos el nombre del cih asociado (sin extension)
    dataset_name = param["LOGGING_INFO"]["DATASET_NAME"].split(".")[0]
    datasetID = fecha + "_" + dataset_name

    # usar data_map; data no es necesario!
    return data, data_map, particulas_por_frame, df, datasetID, output_path

# -----------------------------------
def aplicar_mascara(imagen, xmin, xmax, ymin, ymax):
    """
    Applies mask to image.
    """
    imagen = imagen[ymin:ymax, xmin:xmax]
    return imagen

# -----------------------------------
def crear_fondo(path_fondo, f_first, f_last, extension):
    """
    Creates a background image and stores it as 'fondo.npz' file.
    """
    files = sorted(glob.glob(os.path.join(path_fondo, "*." + extension)))
    Lfondo = f_last - f_first + 1

    f = io.imread(files[f_first]).astype('float64')
    for n in tqdm(range(f_first+1, f_last+1), leave=False):
        f_n = io.imread(files[n]).astype('float64')
        f += f_n
    
    f = f/Lfondo
    np.savez(os.path.join(path_fondo, "fondo.npz"), f)
    return f

# -----------------------------------
def Fondo(path_fondo, f_first, f_last, extension):
    """
    Either loads or generates a background image, and returns it.
    """
    files = set(glob.glob(os.path.join(path_fondo, "*.npz")))
    file_fondo = set([os.path.join(path_fondo, "fondo.npz")])
    L = files.intersection(file_fondo)

    if len(L) == 1:
        print(" ")
        print("Using precalculated background.")
        fondo = np.load(os.path.join(path_fondo, "fondo.npz"))["arr_0"]
    else:
        print(" ")
        print("Generating background from selected images.")
        fondo = crear_fondo(path_fondo, f_first, f_last, extension)

    return fondo

# -----------------------------------
def binarizar(imagen, threshold_value):
    """
    Straightforward binarization of image based on intensity value.
    """
    vec = imagen < threshold_value
    imagen[ vec] = 0
    imagen[~vec] = 255
    return imagen

# -----------------------------------
def detectar_circulos(imagen, Dmin, Dmax, sigma=0.1, use_DIC=False):
    """
    From binary image find position of particles by labeling
    the image. Only areas whose equivalent diameter d verifies
    Dmin <= d <= Dmax
    are considered.
    """
    label = measure.label(imagen)
    props = measure.regionprops(label)

    L = len(props)
    x = []
    y = []

    for m in range(L):
        X = props[m].centroid[1]
        Y = props[m].centroid[0]
        d = props[m].equivalent_diameter

        if props[m].area > 1:
            if d >= Dmin and d <= Dmax:
                x.append(X)
                y.append(Y)

    x = np.array(x)
    y = np.array(y)

    return x, y

# -----------------------------------
def detectar_circulos_inerp(imagen, Dmin, Dmax, sigma=0.1, use_DIC=True):
    """
    From binary image find position of particles
    using Hough transform and correlation.
    """
    # ----------------
    # SA: 
    #    This function has *several*
    #    hard-coded values!
    # ----------------

    # Estimated area of inertial particle
    R_eff = (Dmin + Dmax)/4
    A = np.pi*R_eff**2

    label	= measure.label(imagen)
    img_c	= remove_small_objects(label, A/8)
    regions	= measure.regionprops(img_c)

    L = len(regions)
    x = []
    y = []
    R = []

    Amin = np.pi*(Dmin/2)**2

    idx_parts_edges = []
    max_val = np.min(img_c.shape)
    for m in range(L):
        if 0 in regions[m].bbox or max_val in regions[m].bbox:
            idx_parts_edges.append(m)
    
    areas_edges  = np.array([regions[m].area for m in idx_parts_edges])
    
    idx_small_areas_in_edges = np.array(idx_parts_edges)[areas_edges < 0.35*Amin]

    idx_others   = list( set(range(L)).difference(set(idx_parts_edges)))
    idx_others   = np.array(idx_others)
    
    # From the particles that are *not* in edges of img,
    # discard those whose area is < min. expected area.

    areas_others = np.array([regions[m].area for m in idx_others])
    idx_small_areas_not_in_edges = idx_others[areas_others < 0.5*Amin]
    
    # Keep all areas in edges and *not small* areas not in edges
    idx_to_analyze = (set(range(L)).difference(set(idx_small_areas_not_in_edges)))
    idx_to_analyze = list(idx_to_analyze.difference(set(idx_small_areas_in_edges)))
    idx_to_analyze = np.array(idx_to_analyze)

    for m in idx_to_analyze:
        # isolate current region and labeled pixels
        minr, minc, maxr, maxc = regions[m].bbox
        reg_m = (img_c[minr:maxr, minc:maxc] == regions[m].label).astype(float)
        reg_m[reg_m > 0] = 1

        edges	= feature.canny(reg_m, sigma=sigma)

        h_radii	= np.arange(Dmin//2, Dmax//2 +1)
        h_res	= hough_circle(edges, h_radii, full_output=False)
        # estimate number of particles in current region
        active_area	= np.sum(reg_m)
        rel_active_area	= active_area/A
        Npart_est	= np.ceil(rel_active_area).astype(int)
        Npart_est	*= 5

        # Find possible centers and radii using hough transform
        accums_m, x_m, y_m, R_m = hough_circle_peaks(
                        h_res, h_radii, total_num_peaks=Npart_est)	
        
        
        # Correlate the region with a ball with the effective radius
        ball = disk(R_eff).astype('float')
        corr = correlate2d(reg_m, ball, mode='same')
        corr_m = corr[y_m, x_m] # --> correlation at the position of the centers
        max_corr = np.max(corr)

        # Find distance between centers and threshold
        # with the goal of finding hough centers that
        # correspond to the *same* particle.
        Z = ((x_m[:,None] - x_m[None,:])**2 + (y_m[:,None] - y_m[None,:])**2)**0.5
        Z = Z < 0.3*R_eff
        
        clus = find_clusters(Z)
        
        X_m = np.array([])
        Y_m = np.array([])

        # For each cluster of points determine an unique
        # center --> that will be the position of the particle.
        # Note only points whose mean correlation is above a
        # certain threshold are considered.
        for n in range(clus.shape[0]):
            idx_n = clus[n][clus[n] != -1]
            x_mn = x_m[idx_n]
            y_mn = y_m[idx_n]

            mean_corr_mn = np.mean(corr_m[idx_n])
            if mean_corr_mn >= 0.75*max_corr:
                x_c = np.sum(x_mn*corr_m[idx_n])/np.sum(corr_m[idx_n])
                y_c = np.sum(y_mn*corr_m[idx_n])/np.sum(corr_m[idx_n])
                X_m = np.append(X_m, np.sum(x_mn*corr_m[idx_n])/np.sum(corr_m[idx_n]))
                Y_m = np.append(Y_m, np.sum(y_mn*corr_m[idx_n])/np.sum(corr_m[idx_n]))

        
        if use_DIC == True:
            xe, ye = use_dic_to_detect_missing(X_m, Y_m, R_eff, reg_m, minr, minc) 
            if xe != -1:
                X_m = np.append(X_m, xe)
                Y_m = np.append(Y_m, ye)
                R_m = np.append(R_m, R_eff+1)
                Npart_est += 1
                accums_m = np.append(accums_m, -1)

        # x --> col / y --> row / in cartesian coordinates
        # should be (col, row), not (row, col)
        x = np.append(x, X_m + minc) 
        y = np.append(y, Y_m + minr)
        #R = np.append(R, R_m)

    return np.array(x), np.array(y)

# -----------------------------------
def find_clusters(Z_bool):
    """
    For a boolean 2D array of size LxL
    find connected points (clusters).
    """
    L = Z_bool.shape[0]

    clus = -1*np.ones((1, L), dtype=int)
    idxs = np.arange(L)
    for p in idxs:
        V = clus[clus != -1].flatten()
        if len(set(idxs).difference(set(V))) == 0:
            break
        else:
            neighbors = idxs[Z_bool[p]]

            temp = np.in1d(clus, neighbors).reshape(clus.shape)
            temp = (temp == True).any(axis=1)

            if not any(temp):
                k = clus.shape[0] - 1
                N_elems = len(neighbors)
                clus[k,:N_elems] = neighbors
                clus = np.vstack((clus, -1*np.ones(L, dtype=int)))

            else:
                irow = np.arange(clus.shape[0])[temp][0]

                elems_prev = clus[irow][clus[irow] != -1]
                N_elems    = len(elems_prev)
                new_elems  = set(neighbors).difference(set(elems_prev))
                l          = len(new_elems)
                if l != 0:
                    new_elems  = np.fromiter(new_elems, int, l)
                    clus[irow,N_elems:N_elems + l] = new_elems
    clus = clus[~(clus == -1).all(axis=1)]
    return clus

# -----------------------------------
def use_dic_to_detect_missing(x, y, R, target_region, minr, minc):
    """
    Use Digital Image Correlation to detect
    missing particles in a target region.
    """
    ball  = disk(R+1).astype('float')
    recov = np.zeros_like(target_region)
    for n in range(len(x)):
            recov[int(y[n]), int(x[n])] = 1

    recov              = convolve(recov, ball)
    recov[recov>0]     = 1
    recov_active_area  = np.sum(recov)
    target_active_area = np.sum(target_region)

    # first check if undetected circles
    if (target_active_area - recov_active_area) > 0.2*np.pi*(R**2):
        dif	= np.logical_xor(target_region, recov).astype('double')
        ballDIC	= disk(R+1).astype('float')
        corr	= correlate2d(dif, ballDIC, mode='same')
        maxs	= np.sum(corr == np.max(corr))
        if maxs == 1:
            xe, ye = np.unravel_index(np.argmax(corr), corr.shape)
        else:
            gx, gy = np.gradient(corr)
            corr   = gx**2 + gy**2
            maxs   = np.sum(corr == np.max(corr))
            xe, ye = np.unravel_index(np.argmax(corr), corr.shape)
    # all particles detected in current region.
    else:
        xe, ye = -1, -1
    return xe, ye

# -----------------------------------
def cargar_calibracion(path_camara, idx_ryt):
    """
    Loads a camera calibration file.
    """
    calibracion = np.load(os.path.abspath(path_camara))
    cam_mtx = calibracion["arr_0"]
    roi = calibracion["arr_1"]
    mapx = calibracion["arr_2"]
    mapy = calibracion["arr_3"]
    rvecs = calibracion["arr_4"][idx_ryt]
    tvecs = calibracion["arr_5"][idx_ryt]
    mean_error = calibracion["arr_6"]  # error de reproyección
    distortion = calibracion["arr_7"]  # contiene coeficientes de distorsion

    rvecs = cv2.Rodrigues(rvecs)[0]  # Pasa de representación vectorial a matricial
    # invierto la matriz de rotación
    rvecs = np.linalg.inv(rvecs)
    # invierto la matriz intrinseca
    cam_mtx = np.linalg.inv(cam_mtx)
    return cam_mtx, roi, mapx, mapy, rvecs, tvecs, mean_error, distortion

# -----------------------------------
def undistort_image(img, mapx, mapy):
    """
    Corrects image for distortion effects usign mapx and mapy previously
    generated.
    """
    img = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)
    return img

# -----------------------------------
def no_undistort(img, mapx, mapy):
    """
    No transformation is made to the image. This function exists just for ease
    of coding.
    """
    return img

# -----------------------------------
def not_change_gamma(img, gamma):
    """
    No gamma correction is made to the image.
    """
    return img

# -----------------------------------
def mapear(data, cam_mtx, rvecs, tvecs):
    """
    Needs docs.
    """
    tvecs.shape = 3
    N = len(data)
    data_mapeado = []
    for n in range(N):
        M = len(data[n][0])
        X = np.zeros(M)
        Y = np.zeros(M)

        for m in range(M):
            x = data[n][0][m]
            y = data[n][1][m]

            vec = np.array([x, y, 1])
            Vec = np.dot(rvecs, np.dot(cam_mtx, vec) - tvecs)

            X[m] = Vec[0]
            Y[m] = Vec[1]

        data_mapeado.append((X, Y))
    return data_mapeado

# -----------------------------------
def no_mapear(data, cam_mtx, rvecs, tvecs):
    """
    Needs docs.
    """
    return data

# -----------------------------------
def build_trajectories(dfo, dt=1, use_predictive=True,\
                       use_curvature_history = True,\
                       proximity_tolerance=30,\
                       max_number_nans = 3,\
                       min_number_frames = 2,\
                       umbral_trayectorias_muertas = 20,\
                       output_path = './',\
                       datasetID = '',\
                       print_progress=True):
    """
    Builds a trajectories in a (pandas) DataFrame from an input dataframe
    of particles detected in each frame [previously generated by 'analizar_set_datos'].
    """

    df, Nframes, N_inic_particulas, indice_frames,\
    indice_columnas, nframe = crear_dataframe_destino(dfo)
    
    labels_particulas_eliminadas = np.array([])
     
    for current_frame in tqdm(df.index.values[1:], disable=~np.array(print_progress),
                    desc='Analyzing snapshots for particle trajectories...'):
        
        previous_frame = current_frame - 1

        frames_perdidos, trayectorias_vivas, N_trayectorias_vivas,\
        ultimo_frame_valido, primer_frame_valido,\
        numero_de_frames_validos, previous_pool,\
        Nro_trayectorias, trayectorias_a_continuar,\
        indice_particulas, indice_particulas_r,\
        df_r, Nro_trayectorias_a_continuar = analizar_trayectorias_a_continuar(df, current_frame, Nframes, maximo_de_nans_tolerables= max_number_nans, minimo_de_frames_validos=min_number_frames)

        
        current_pool,\
        Nro_candidatos,\
        indice_candidatos = generar_current_pool(dfo, current_frame)
        
        if Nro_candidatos > 0:
            # SA: las condiciones a chequear son que
            #   - (1) haya trayectorias vivas con al menos *2* elementos
            #   - (2) haya trayectorias continuadas sinteticamente en el df_r
            #       (estas siempre tienen al menos 2 elementos)

            # Condición (1)
            frames_validos_tray_vivas = numero_de_frames_validos[trayectorias_vivas]
            condicion_1 = np.sum(frames_validos_tray_vivas > 1)

            # Condición (2) 
#            condicion_2 = (trayectorias_a_continuar == True).any()
            condicion_2 = Nro_trayectorias_a_continuar>0
            
            condicion = condicion_1.astype('bool') + condicion_2

            #----------------------------------------------------------
            # Caso 1: Hay trayectorias vivas con al menos dos elementos en la historia.
            #         Hay trayectorias continuadas sintéticamente.
            if (condicion_1 > 0 and condicion_2 == True):
                pre_prev_pool = pool_antepenultimo(df, df_r, current_frame, trayectorias_vivas, trayectorias_a_continuar)

                curv = calcular_curvaturas_potenciales(df, pre_prev_pool, previous_pool, current_pool, Nro_candidatos,
                        indice_particulas, indice_candidatos, Nro_trayectorias, N_trayectorias_vivas, numero_de_frames_validos, trayectorias_vivas)
                if use_predictive == True:
                    previous_pool = predecir_el_previous_pool(df, numero_de_frames_validos, primer_frame_valido, ultimo_frame_valido, current_frame, trayectorias_vivas, previous_pool)
                    previous_pool_r = df_r.iloc[-1,:].values.astype(float).flatten()
                    previous_pool[2*N_trayectorias_vivas:] = previous_pool_r
            #----------------------------------------------------------
            # Caso 2: Hay trayectorias vivas con al menos dos elementos en la historia.
            #         *NO* hay trayectorias continuadas sintéticamente.
            elif (condicion_1 > 0 and condicion_2 == False):
                pre_prev_pool = pool_antepenultimo(df, df_r, current_frame, trayectorias_vivas, trayectorias_a_continuar)

                curv = calcular_curvaturas_potenciales(df, pre_prev_pool, previous_pool, current_pool, Nro_candidatos,
                        indice_particulas, indice_candidatos, Nro_trayectorias, N_trayectorias_vivas, numero_de_frames_validos, trayectorias_vivas)
                if use_predictive == True:
                    previous_pool = predecir_el_previous_pool(df, numero_de_frames_validos, primer_frame_valido, ultimo_frame_valido, current_frame, trayectorias_vivas, previous_pool)
            #----------------------------------------------------------
            # Caso 3: *NO* hay trayectorias vivas con al menos dos elementos en la historia
            #         pero SÍ existen trayectorias continuadas sintéticamente.
            elif (condicion_1==0 and condicion_2 == True):
                
                pre_prev_pool = np.nan*np.ones(2*N_trayectorias_vivas)

                pre_prev_pool_r = df_r.iloc[-2, :].values.flatten()
                pre_prev_pool = np.append(pre_prev_pool, pre_prev_pool_r)

                curv = calcular_curvaturas_potenciales(df,pre_prev_pool, previous_pool, current_pool, Nro_candidatos,
                        indice_particulas, indice_candidatos, Nro_trayectorias, N_trayectorias_vivas, numero_de_frames_validos, trayectorias_vivas)

                if use_predictive == True:
                    previous_pool = predecir_el_previous_pool(df, numero_de_frames_validos, primer_frame_valido, ultimo_frame_valido, current_frame, trayectorias_vivas, previous_pool)
                    previous_pool_r = df_r.iloc[-1,:].values.astype(float).flatten()
                    previous_pool[2*N_trayectorias_vivas:] = previous_pool_r
            #----------------------------------------------------------
            # Caso 4: Ninguna de las dos condiciones.
            #         Se crea un 'curv' vacío.
            else:
                curv = pd.DataFrame(0, index=indice_particulas, columns=indice_candidatos)
            #----------------------------------------------------------
            #       FIN DE EDICIÓN

            if use_curvature_history == True:
                curv_hist = calcular_curvatura_historica(current_frame, df, df_r, indice_particulas, trayectorias_vivas, N_trayectorias_vivas, Nro_trayectorias_a_continuar, Nro_trayectorias, numero_de_frames_validos, trayectorias_a_continuar, frames_perdidos, indice_particulas_r)
            else:
                curv_hist = pd.DataFrame(index = ['curv_historica'], columns=indice_particulas)
            

            if df.index.get_loc(current_frame) == 1:
                    prox_tol = proximity_tolerance*2
            else:
                    prox_tol = proximity_tolerance
            
            asoc = asociar(previous_pool, current_pool, prox_tol, indice_particulas, indice_candidatos, curv, df, current_frame, dfo, curv_hist, use_curvature_history)

            df = asignar_candidatos_a_trayectorias_nuevo(asoc, df, current_frame, df_r, current_pool, indice_particulas_r, labels_particulas_eliminadas, use_predictive=True)
            
            df, labels_particulas_eliminadas = limpiar_df(df, umbral_trayectorias_muertas, current_frame, Nframes, output_path, datasetID, max_number_nans, min_number_frames, labels_particulas_eliminadas)
       
    escribir_df(df, output_path, datasetID, current_frame+1)
       
   
    return df

# -----------------------------------
def escribir_df(df, output_path,\
                datasetID,\
                current_frame,\
                remove_noise = 1,\
                min_duration = 10):
    """
    Escribe a disco, en el path indicado, el DataFrame
    """
    
    indice_inicial = df.index.values[0]
    
    if remove_noise == 1:
        L = df.count().values > min_duration
    
    else:
        L = np.ones(df.shape[1], dtype=bool)

    if np.sum(L) > 0:
        df.iloc[:,L].to_pickle(os.path.join(output_path, 'DF_X_' + datasetID + '_{:06d}.out').format(current_frame))
    
    return None 

# -----------------------------------
def limpiar_df(df, umbral_trayectorias_muertas,\
               current_frame,\
               Nframes,\
               output_path,\
               datasetID,\
               max_number_nans,\
               min_number_frames,\
               labels_particulas_eliminadas):
    """
    Remueve del DataFrame las trayectorias que ya no pueden ser continuadas y las escribe a disco.

    """
    #---------------------
    # SA:
    #    En este caso ya se asignaron al DataFrame las nuevas trayectorias.
    #    Hay que buscar qué trayectorias no se van a poder continuar en el
    #    frame que sigue.
    #    Sacamos el -1, comparado con el cálculo de frames_perdidos en la función
    #    ' analizar_trayectorias_a_continuar '.
    #---------------------

    frames_perdidos = current_frame - get_last_non_null_vectorial( df.loc[:current_frame, :] ) 

    trayectorias_vivas = (frames_perdidos == 0)

    ultimo_frame_valido = get_last_non_null_vectorial(df.loc[:current_frame, :])
    primer_frame_valido = get_first_non_null_vectorial(df.loc[:current_frame, :])
    
    numero_de_frames_validos = ultimo_frame_valido - primer_frame_valido + 1

    # lo asignamos a 2*Nframes para que los nans no molesten al comparar
    frames_perdidos[np.isnan(frames_perdidos)] = 2 * Nframes
    # Ahora encuentro qué trayectorias debería continuar sinteticamente.
    trayectorias_a_continuar = ( (frames_perdidos > 0) * (frames_perdidos <= max_number_nans)
        * (numero_de_frames_validos >= min_number_frames) )
    
    Trayectorias_muertas = ~(trayectorias_vivas + trayectorias_a_continuar)

    N_trayectorias_muertas = int(np.sum(Trayectorias_muertas) / 2)

    if N_trayectorias_muertas >= umbral_trayectorias_muertas:
        
        escribir_df(df.loc[:, Trayectorias_muertas], output_path, datasetID, current_frame)

        labels = sorted(np.array([int(s[2:]) for s in list(df.loc[:, Trayectorias_muertas].columns)[::2]]))
        
        labels_particulas_eliminadas = sorted(np.append(labels_particulas_eliminadas, labels))

        df = df.loc[:, ~Trayectorias_muertas]

    else:
        pass

    return df, labels_particulas_eliminadas

# -----------------------------------
def analizar_trayectorias_a_continuar(df,\
                                      current_frame,\
                                      Nframes,\
                                      maximo_de_nans_tolerables=3,\
                                      minimo_de_frames_validos=2):
    """
    Analiza si hay trayectorias que continuar sinteticamente.
    """

   # Busca en el df hasta (inclusive) el frame inmediatamente anterior al current
    # cuál es el primer y último frame válido de cada trayectoria.
    ultimo_frame_valido = get_last_non_null_vectorial(df.loc[:current_frame-1, :])
    primer_frame_valido = get_first_non_null_vectorial(df.loc[:current_frame-1, :])
    numero_de_frames_validos = ultimo_frame_valido - primer_frame_valido + 1
    
    frames_perdidos = current_frame - ultimo_frame_valido - 1 
    
    # PJC: le agregué el -1 al final porque asi se puede interpretar
    #      correctamente este valor como la cantidad de frames perdidos

    # Donde 'frames_perdidos' tiene valores igual a 0, tenemos trayectorias vivas
    # es decir, que tuvieron un punto asignado en el frame previo
    trayectorias_vivas = (frames_perdidos == 0)
    # PJC: cambié el == 1 original por == 0 al cambiar la definicion de
    #      frames_perdidos

    N_trayectorias_vivas = int(np.sum(trayectorias_vivas) / 2)
    
    
    # tomamos las posiciones ocupadas en el frame inmediato anterior
    # por las trayectorias aun vivas, y las ponemos en previous_pool
    previous_pool = df.loc[current_frame - 1, trayectorias_vivas ].values.flatten()
    # el Nro_trayectorias aqui corresponde a cuantas trayectorias llegan a
    # al current_frame desde el frame anterior
    Nro_trayectorias = len(previous_pool) // 2

    # lo asignamos a 2*Nframes para que los nans no molesten al comparar
    frames_perdidos[np.isnan(frames_perdidos)] = 2 * Nframes
    # Ahora encuentro qué trayectorias debería continuar sinteticamente.
    trayectorias_a_continuar = ( (frames_perdidos > 0) * (frames_perdidos <= maximo_de_nans_tolerables)
        * (numero_de_frames_validos >= minimo_de_frames_validos) )
    # PJC: cambie la condicion frames_perdidos > 1 a frames_perdidos > 0,
    #      consistente con la interpretacion, luego de cambiar la definicion
    #      de frames_perdidos mas arriba.

    # trayectorias_a_continuar es entonces un booleano con la lista de
    # aquellas trayectorias que hay que continuar sintéticamente

    # genero lista con los nombres de las particulas que estan vivas;
    # tipo 'P1', 'P100', etc.
    indice_particulas = [ s[1:] for s in list(df.columns[trayectorias_vivas][::2].values) ]


    # Creo un DataFrame reducido 'df_r' vacio, y si es necesario lo lleno.
    df_r = pd.DataFrame(index=[], columns=[])
    if (trayectorias_a_continuar == False).all() == True:
        # NO hay trayectorias a continuar sinteticamente
        pass
    else:
        # SI hay trayectorias a continuar sinteticamente

        # esto estaba fuera del 'if' anterior, pero no tiene sentido ahi
        # df_r = pd.DataFrame(index=[], columns=[])
        
        # veamos cuantos frames hemos perdido en cada trayectoria,
        # para saber cuantos valores debemos extrapolar de cada trayectoria
        max_frames_perdidos = int(np.max(frames_perdidos[trayectorias_a_continuar]))

        # generamos el dataframe reducido y lo continuamos
        # SA: Cambio para qué tiempos se toma el df_r. 
        #     Va a ir siempre hasta current_frame inclusive.
        #     De esta forma, cuando se activa el predecir trayectoria ya está la predicción calculada.
        df_r = df.loc[ (current_frame - max_frames_perdidos - minimo_de_frames_validos) : current_frame, trayectorias_a_continuar ]
        # df_r = df.iloc[ (current_frame - max_frames_perdidos) - 1 : current_frame, trayectorias_a_continuar ]

        df_r = extrapolar_trayectoria(df_r)

        previous_pool_r = df_r.iloc[-2, :].values.flatten()
        previous_pool = np.append(previous_pool, previous_pool_r)

    Nro_trayectorias_a_continuar = df_r.shape[1]//2

    Nro_trayectorias = Nro_trayectorias + Nro_trayectorias_a_continuar
    
        
    indice_particulas_r = [s[1:] for s in list(df_r.columns[::2].values)]
    for s in indice_particulas_r:
        indice_particulas.append(s)


    return frames_perdidos, trayectorias_vivas, N_trayectorias_vivas, ultimo_frame_valido, primer_frame_valido, numero_de_frames_validos, previous_pool, Nro_trayectorias, trayectorias_a_continuar, indice_particulas, indice_particulas_r, df_r, Nro_trayectorias_a_continuar

# -----------------------------------
def extrapolar_trayectoria(df_r):
    """
    Continuar una trayectoria dada. 
    """
    n_lin_df = df_r.shape[0]
    
    # Vamos barriendo de a una trayectoria a continuar
    # por vez para extrapolar 
    for i in range(df_r.shape[1] // 2):
        pos_validos = ~np.isnan(df_r.iloc[:, 2*i].values.astype(float))

        # Cálculo de cuántos elementos no NaN tiene la tray. a continuar
        ni = np.sum(pos_validos)
        #ni = np.sum(~np.isnan(df_r.iloc[:, 2 * i].values.astype(float)))
        idx_ultimo_valido = np.max(np.where(pos_validos==True)[0])

        interp_order = np.min((ni - 1, 2))

        if ni > 1:
            t = np.arange(ni)
            x = df_r.iloc[pos_validos, 2 * i    ].values.astype(float)
            y = df_r.iloc[pos_validos, 2 * i + 1].values.astype(float)
            xsp = InterpolatedUnivariateSpline(t, x, ext=0, k=interp_order)
            ysp = InterpolatedUnivariateSpline(t, y, ext=0, k=interp_order)
            # original
            t_int = np.arange(idx_ultimo_valido + 1 , n_lin_df)
            xc = xsp(t_int)
            yc = ysp(t_int)
            df_r.iloc[(idx_ultimo_valido+1):, 2 * i    ] = xc
            df_r.iloc[(idx_ultimo_valido+1):, 2 * i + 1] = yc
            # t_int = t[-1]
            # xc = xsp(t_int)
            # yc = ysp(t_int)
            # df_r.iloc[ni+1, 2 * i] = xc
            # df_r.iloc[ni+1, 2 * i + 1] = yc
    return df_r

# -----------------------------------
def crear_dataframe_destino(dfo):
    """
    Crea el dataframe destino para las trayectorias a partir del primer frame.
    """

    # df, Nframes, N_inic_particulas, indice_frames, indice_columnas = crear_dataframe_destino(dfo)
    # CREAMOS EL DATAFRAME DESTINO (inicial*)
    # [* = porque puede crecer!]
    # Creamos el dataframe de pandas con columnas iniciales:
    # index | (frame) | xnp1 | ynp1 | xnp2 | ynp2 | ... | ... etc
    # normalmente sabemos cual es el numero inicial maximo de particulas
    # que tendremos en el dataset 

    # Tomamos las posiciones de TODAS las partículas detectadas
    # en el frame inicial
    nframe = 0
    #### SA: Búsqueda del primer frame con partículas detectadas
    #        Se guardan en un array xori, yori
    xori, yori = get_xy_positions_of_frame(dfo, nframe)
    while int(np.sum(~np.isnan(xori)))==0:
        nframe += 1
        xori, yori = get_xy_positions_of_frame(dfo, nframe)

    # El numero inicial de particulas corresponde a la cantidad de valores
    # no-nan (valores validos) de xori.
    N_inic_particulas = int(np.sum(~np.isnan(xori)))
    Nframes = dfo.shape[0]

    # generamos los indices de lineas y headers de columnas
    # ATENCION
    # - los frames     se enumeran desde 0
    # - las particulas se enumeran desde 1
    #   y cada columna lleva el prefijo 'xP' e 'yP', e.g.: xP5

    indice_frames = np.array(dfo.index)

    indice_columnas = generar_nombre_columnas(N_inic_particulas)

    # generamos el dataframe destino (con el tamano inicial)
    df = pd.DataFrame(index=indice_frames, columns=indice_columnas, dtype=float)

    # El primer paso es asignar_candidatos_a_trayectorias a cualquier columna de las
    # primer fila (indice o frame = 0) las diferentes particulas que
    # detectamos en el instante inicial
    indice = 0
    for i in range(len(xori)):
        if np.isnan(xori[i]) == False:
            indice += 1
            df.at[indice_frames[nframe], "xP" + str(indice)] = xori[i]
            df.at[indice_frames[nframe], "yP" + str(indice)] = yori[i]

    return df, Nframes, N_inic_particulas, indice_frames, indice_columnas, nframe 

# -----------------------------------
def get_first_non_null_vectorial(df):
    """
    Faster way for getting the index of the first non-nan entry
    in every column of dataframe df.
    (Must return nan for those columns which only hold nans)
    Returns _pandas index_, not array index!
    """
    row_index = df.apply(pd.Series.first_valid_index).values.astype('float')
    return row_index

# -----------------------------------
def get_last_non_null_vectorial(df):
    """
    Faster way for getting the index of the last non-nan entry
    in every column of dataframe df.
    (Must return nan for those columns which only hold nans)
    Returns _pandas index_, not array index!
    """
    row_index = df.apply(pd.Series.last_valid_index).values.astype('float')
    return row_index

# -----------------------------------
def get_xy_positions_of_frame(df, frame):
    """
    Get x and y positions (two vectors) of positions of particles
    in disordered dataframe 'df', for a given frame index
    (i.e., not a frame label).
    """
    #-----------------------------------------------------------
    #   Atencion: el segundo indice arranca en 1 (y luego en 2) porque la
    #             columna 0 de df original (desordenado) contiene el Nro de particulas,
    #             las demas tienen  alternadamente las posiciones x e y de particulas
    #-----------------------------------------------------------
    x = df.iloc[frame, 1::2].values.astype(float).flatten()
    y = df.iloc[frame, 2::2].values.astype(float).flatten()
    return x, y

# -----------------------------------
def generar_nombre_columnas(N_inic_particulas):
    indice_columnas = []
    for i in range(1, N_inic_particulas + 1):
        indice_columnas.append("xP" + str(i))
        indice_columnas.append("yP" + str(i))
    return indice_columnas

# -----------------------------------
def crear_indice_candidatos(Nro_candidatos):
    indice_candidatos = []
    for i in range(1, int(Nro_candidatos) + 1):
        indice_candidatos.append("C" + str(i))
    return indice_candidatos

# -----------------------------------
def crear_indice_particulas(Nro_trayectorias):
    indice_particulas = []
    for i in range(1, int(Nro_trayectorias) + 1):
        indice_particulas.append("P" + str(i))
    return indice_particulas

# -----------------------------------
def calcular_distancias_particulas_candidatos(previous_pool, current_pool):
    """
    Calculates distance matrix between particles and candidates.
    """
    Nro_trayectorias = len(previous_pool) // 2
    Nro_candidatos = len(current_pool) // 2
    set_particulas = np.reshape(previous_pool[0 : 2 * Nro_trayectorias + 2], (-1, 2))
    set_candidatos = np.reshape(current_pool[0 : 2 * Nro_candidatos + 2], (-1, 2))
    distancias_entre_sets = cdist(set_particulas, set_candidatos)
    return distancias_entre_sets

# -----------------------------------
def asociar(
                previous_pool,
                current_pool,
                proximity_tolerance,
                indice_particulas,
                indice_candidatos,
                curv, df, current_frame, dforig, curv_hist, use_curvature_history,
                ):
    """ 
    Genera la matriz/dataframe de asociacion asoc.
    """
    # CALCULAMOS DISTANCIA ENTRE TRAYECTORIAS PREEXISTENTES Y CANDIDATOS
    distancia_entre_particulas_y_candidatos = calcular_distancias_particulas_candidatos(previous_pool, current_pool)
    dist = pd.DataFrame(
                    distancia_entre_particulas_y_candidatos,
                    index=indice_particulas,
                    columns=indice_candidatos,
                    )
    asoc = dist.astype(float).idxmin()

    dist2 = dist.copy()
    indices_de_asoc = asoc.index # lista de candidatos
    for elem in range(len(asoc)):
        dist2.loc[asoc[elem], indices_de_asoc[elem]] = np.nan
    asoc_2 = dist2.idxmin()

    for i in range(len(asoc)):
        if isinstance(asoc[i], str):
            if dist.loc[asoc[i], "C" + str(i + 1)] > proximity_tolerance:
                asoc[i] = np.nan
            else:  
                pass

    for i in range(len(asoc_2)):
        if isinstance(asoc_2[i], str):
            if dist.loc[asoc_2[i], "C" + str(i + 1)] > proximity_tolerance:
                asoc_2[i] = np.nan
            else: 
                pass

    for elem in range(len(asoc)):
        particula_de_asoc = asoc[elem]
        particula_de_asoc2 = asoc_2[elem]
        
        # tiene nans en ambos?
        if is_str_nan(particula_de_asoc) and is_str_nan(particula_de_asoc2):
            pass
        elif is_str_nan(particula_de_asoc) and ~is_str_nan(particula_de_asoc2):
            asoc[elem] = asoc_2[elem]
        elif ~is_str_nan(particula_de_asoc) and is_str_nan(particula_de_asoc2):
            pass
        else:
            curvatura_de_asoc = np.abs(curv.loc[asoc[elem], asoc.index[elem]])
            curvatura_de_asoc2 = np.abs(curv.loc[asoc_2[elem], asoc_2.index[elem]])            

            if use_curvature_history == True:
                asoc = comparar_curvaturas(curvatura_de_asoc, curvatura_de_asoc2, asoc, asoc_2, elem, curv_hist)

            # dirimimos por menor curvatura (siempre se lleva uno de aca!)
            elif curvatura_de_asoc > curvatura_de_asoc2:
                asoc[elem] = asoc_2[elem]

    # Mientras que existan repeticiones:
    while np.max(asoc.value_counts(ascending=False)) > 1:
        repeticiones = asoc.value_counts(ascending=False)
        # Buscamos en 'asoc' el primero que tenga repeticiones.
        # Armamos la lista de distancias a comparar.
        particula_repetida = repeticiones.index[0]
        candidatos_a_dirimir = []
        for j in range(repeticiones[0]):
            candidatos_a_dirimir.append(asoc[asoc == repeticiones.index[0]].index[j])

        curvaturas_a_dirimir = curv.loc[particula_repetida][candidatos_a_dirimir]
        mejor_candidato = curvaturas_a_dirimir.astype(float).idxmin()
        # comparamos y dejamos solo esa, las demas le ponemos NaNs en dist.
        # la forma facil de hacer esto es remover el candidato mas proximo de
        # la lista de candidatos_a_dirimir, y luego usar los candidatos que
        # quedaron alli para poner nans en el dataframe dist.
        if mejor_candidato is not np.nan:
            candidatos_a_dirimir.remove(mejor_candidato)
        asoc.loc[candidatos_a_dirimir] = np.nan
        # ACA TERMINA DE ASOCIAR
    return asoc

# -----------------------------------
def is_str_nan(part):
    """
    Checks if `part' is str or not (is NaN).
    returns True if NaN
    returns False if not NaN
    """
    return ~np.array(isinstance(part, str))

# -----------------------------------
def comparar_curvaturas(curvatura_de_asoc, curvatura_de_asoc2, asoc, asoc_2, elem, curv_hist):
    
    curvatura_historica_de_asoc  = curv_hist.loc['curv_historica', asoc[elem]  ]
    curvatura_historica_de_asoc2 = curv_hist.loc['curv_historica', asoc_2[elem]]

    comparacion   = np.abs(curvatura_de_asoc  - curvatura_historica_de_asoc )
    comparacion_2 = np.abs(curvatura_de_asoc2 - curvatura_historica_de_asoc2)

    # Se dirime por qué curvatura 'nueva' se parece más a la 'histórica'
    comparaciones = np.array([comparacion, comparacion_2])

    trayectoria_para_candidato = np.where(comparaciones = np.nanmin(comparaciones))
    # dirimimos por menor curvatura (siempre se lleva uno de aca!)
    if len(trayectoria_para_candidato != 0):
        if trayectoria_para_candidato == 1:
            asoc[elem] = asoc_2[elem]
        else:
            pass
    return asoc

# -----------------------------------
def asignar_candidatos_a_trayectorias_r(asoc, df, current_frame, df_r, current_pool_r):
    """ 
    Dada una asociacion ya calculada, hace la asignacion de candidatos a
    trayectorias.
    PARA DATAFRAMES REDUCIDOS
    """
    # Ahora realizamos la asociacion, pasando las coordenadas actuales de
    # los candidatos a las trayectorias preexistentes.
    for i in range(len(asoc)):
        # chequea si tiene asignada una particula
        if isinstance(asoc[i], str):
            lin = df_r.shape[0] - 1
            df.loc[current_frame - lin : current_frame - 1, "x" + asoc[i]] = df_r.loc[
                :, "x" + asoc[i]
            ].values.astype(float)
            df.loc[current_frame - lin : current_frame - 1, "y" + asoc[i]] = df_r.loc[
                :, "y" + asoc[i]
            ].values.astype(float)
            df.loc[current_frame, "x" + asoc[i]] = current_pool_r[2 * i]
            df.loc[current_frame, "y" + asoc[i]] = current_pool_r[2 * i + 1]
    return df

# -----------------------------------
def asignar_candidatos_a_trayectorias_nuevo(asoc, df, current_frame, df_r, current_pool, indice_particulas_r, labels_particulas_eliminadas, use_predictive=True):
    """ 
    Dada una asociacion ya calculada, hace la asignacion de candidatos a
    trayectorias.
    """
    # Ahora realizamos la asociacion, pasando las coordenadas actuales de
    # los candidatos a las trayectorias preexistentes.
    indice_inicial = df.index.values[0]

    candidatos_no_asignados = []
    
    labels_particulas = sorted(np.array([int(s[2:]) for s in list(df.columns)[::2]]))

    labels_particulas = sorted(np.append(labels_particulas, labels_particulas_eliminadas))
    # print('  ', 'trayectorias extendidas', trayectorias_extendidas, sep='\n')
    for i in range(len(asoc)):
        # chequea si tiene asignada una particula
        if isinstance(asoc[i], str):
            if asoc[i] in indice_particulas_r:

                #-------------------------------------------------------------- 
                # SA: 
                #     El df_r ahora tiene una fila más, para cubrir el caso que 
                #     'use_predictive=True'.
                #     El numero de líneas que importa entonces es 
                #               lin = df_r.shape[0] - 1
                #     En consecuencia, en el df_r hay que buscar las 
                #     trayectorias hasta la anteúltima fila inclusive.
                #-------------------------------------------------------------- 

                lin = df_r.shape[0] - 1
                
                # First assign the points that were extrapolated to the *real* trajectory.
                df.loc[current_frame - lin : current_frame - 1, 'x' + asoc[i] ] = df_r.loc[:(current_frame - 1), 'x' + asoc[i]].values.astype(float)
                df.loc[current_frame - lin : current_frame - 1, 'y' + asoc[i] ] = df_r.loc[:(current_frame - 1), 'y' + asoc[i]].values.astype(float)

                # Now assign the new *real* candidate
                df.loc[current_frame, 'x' + asoc[i]] = current_pool[2 * i]
                df.loc[current_frame, 'y' + asoc[i]] = current_pool[2 * i + 1]
            else:
                df.loc[current_frame, 'x' + asoc[i]] = current_pool[2 * i]
                df.loc[current_frame, 'y' + asoc[i]] = current_pool[2 * i + 1]
                
        else:
            idx_new = int(labels_particulas[-1] + 1)
            labels_particulas.append(idx_new)
            df['xP' + str(idx_new)] = np.nan
            df['yP' + str(idx_new)] = np.nan
            df.loc[current_frame, 'xP' + str(idx_new)] = current_pool[2 * i]
            df.loc[current_frame, 'yP' + str(idx_new)] = current_pool[2 * i + 1]

    return df

# -----------------------------------
def asignar_candidatos_a_trayectorias(asoc, df, current_frame, current_pool):
    """ 
    Dada una asociacion ya calculada, hace la asignacion de candidatos a
    trayectorias.
    """
    # Ahora realizamos la asociacion, pasando las coordenadas actuales de
    # los candidatos a las trayectorias preexistentes.
    candidatos_no_asignados = []
    for i in range(len(asoc)):
        # chequea si tiene asignada una particula
        if isinstance(asoc[i], str):
            df.at[current_frame, "x" + asoc[i]] = current_pool[2 * i]
            df.at[current_frame, "y" + asoc[i]] = current_pool[2 * i + 1]
        # En todos los lugares donde asoc tiene NaNs es porque no se encontró
        # ninguna trayectoria preexistente para ese candidato.
        # Ese candidato tiene que iniciar una trayectoria nueva en ese frame.
        else:
            candidatos_no_asignados.append(i)

    return df, candidatos_no_asignados

# -----------------------------------
def pool_antepenultimo(df, df_r, current_frame, trayectorias_vivas, trayectorias_a_continuar): 
    """
    Construye el pool anterior al previo.
    """

    # el anterior al previo
    pre_prev_pool = df.loc[ current_frame - 2, trayectorias_vivas ].values.flatten()

    if (trayectorias_a_continuar == False).all() == True:
        # NO hay trayectorias a continuadas sintéticamente
        pass
    else:
        # SI hay trayectorias continuadas sintéticamente
        pre_prev_pool_r = df_r.iloc[-2, :].values.flatten()
        pre_prev_pool = np.append(pre_prev_pool, pre_prev_pool_r)

    return pre_prev_pool

# -----------------------------------
def calcular_curvaturas_potenciales(df,\
                                    pre_prev_pool,\
                                    previous_pool,\
                                    current_pool,\
                                    Nro_candidatos,\
                                    indice_particulas,\
                                    indice_candidatos,\
                                    Nro_trayectorias,\
                                    N_trayectorias_vivas,\
                                    numero_de_frames_validos,\
                                    trayectorias_vivas):
    """ 
    Cálculo de las curvaturas potenciales que cada candidato le daria a las
    trayectorias preexistentes.
    """

    curv = pd.DataFrame(index=indice_particulas, columns=indice_candidatos)
    tray_cortas = numero_de_frames_validos[trayectorias_vivas][::2] < 2
    # para que los tamaños den, a revisar // feb 2020
    tray_cortas = np.append( tray_cortas, False * np.ones(Nro_trayectorias - N_trayectorias_vivas)).astype(bool)

    xypp = pre_prev_pool[~np.repeat(tray_cortas,2)]
    xyp  = previous_pool[~np.repeat(tray_cortas,2)]
    
    indice_particulas_a_calcular_curv = curv.index[~tray_cortas]
    
    xp_yp_a_calcular_curv = []
    [xp_yp_a_calcular_curv.extend(['x'+part, 'y'+part]) for part in indice_particulas_a_calcular_curv];

    df_curvature = pd.DataFrame(index=np.arange(3), columns=xp_yp_a_calcular_curv)
    df_curvature.iloc[0, :] = xypp
    df_curvature.iloc[1, :] = xyp

    for j in range(Nro_candidatos):
        xc, yc = current_pool[2 * j : 2 * j + 2]
        df_curvature.iloc[2, 0::2] = xc
        df_curvature.iloc[2, 1::2] = yc

        K = curv_vect(df_curvature)

        curv.loc[indice_particulas_a_calcular_curv, indice_candidatos[j] ] = K

    return curv

# -----------------------------------
def curv_vect(df):
    x = 1j*(df.iloc[:, 1::2].values.astype(float))
    x += df.iloc[:, 0::2].values.astype(float)

    # array de posiciones complejas listo.
    # tiene tamaño 3 lineas x N particulas.
    
    # Agregamos eps para evitar dividir por cero
    dpx = x[2,:]-x[1,:] 
    dpxa = np.abs(dpx + np.finfo(float).eps)
    dmx = x[1,:]-x[0,:]
    dmxa = np.abs(dmx + np.finfo(float).eps)

    k = 2*np.abs( dpx/dpxa - dmx/dmxa )/( dpxa + dmxa ) 
    return k

# -----------------------------------
def calcular_curvatura_historica(current_frame,\
                                 df, df_r,\
                                 indice_particulas,\
                                 trayectorias_vivas,\
                                 N_trayectorias_vivas,\
                                 Nro_trayectorias_a_continuar,\
                                 Nro_trayectorias,\
                                 numero_de_frames_validos,\
                                 trayectorias_a_continuar,\
                                 frames_perdidos,\
                                 indice_particulas_r):

    """
    Calcula una curvatura 'historica' para *todas* las trayectorias.
    [incluye trayectorias continuadas sintéticamente]
    Se calcula la curvatura punto a punto para cada trayectoria
    y luego se considera el valor máximo de la misma, con una tolerancia
    dada por la desviación estándar.
    """
    
    curv_hist = pd.DataFrame(index = ['curv_historica'], columns=indice_particulas)
    # SA:
    #    Sólamente las trayectorias con al menos 3 frames
    #    se toman para calcular una curvatura historica.
    tray_vivas_a_calcular = numero_de_frames_validos[trayectorias_vivas][::2] > 2

    todas_las_tray_a_calcular = np.append( tray_vivas_a_calcular, True * np.ones(Nro_trayectorias_a_continuar)).astype(bool)
    
    # A las trayectorias muy cortas se les asigna un 0
    curv_hist.loc['curv_historica', ~todas_las_tray_a_calcular] = 0

    indice_particulas_a_calcular = curv_hist.columns[todas_las_tray_a_calcular]

    tiempos_df_r = df_r.index
    frames_perdidos_r = pd.DataFrame(index = ['Num_frames_perdidos'], columns = indice_particulas_r)
    frames_perdidos_r.loc['Num_frames_perdidos', :] = frames_perdidos[trayectorias_a_continuar][::2]
    #frames_perdidos_r = pd.DataFrame(frames_perdidos[trayectorias_a_continuar][::2], index=['Num_frames_perdidos'], columns=indice_particulas_r)

    xp_yp_a_calcular = []
    [xp_yp_a_calcular.extend(['x'+part, 'y'+part]) for part in indice_particulas_a_calcular];

    for npart, PART in enumerate(indice_particulas_a_calcular):
        if PART in indice_particulas_r:
            # Se toman los x, y de la trayectoria real
            xy_hist = df.loc[:, ['x'+PART, 'y'+PART]].dropna().values.flatten().astype(float) #Esto tiene que ser un array plano de forma x1 y1 x2 y2 x3 y3 etc

            # A estos x, y hay que agregarle la parte sintética
            # SA: corrijo [].astype(int) por int( [] )
            frames_perdidos_n = int(frames_perdidos_r.loc['Num_frames_perdidos', PART])
            xy_hist_r = df_r.loc[ tiempos_df_r[ (-frames_perdidos_n-1) :-1] , ['x'+PART, 'y'+PART] ].dropna().values.flatten().astype(float)
            xy_hist = np.append(xy_hist, xy_hist_r)
        else:
            xy_hist = df.loc[:, ['x'+PART, 'y'+PART]].dropna().values.flatten().astype(float) 
    
        XY_curvatura_hist_n = np.vstack((xy_hist, np.roll(xy_hist, shift=-2), np.roll(xy_hist, shift=-4)))
        
        DF_curv_hist_n = pd.DataFrame(XY_curvatura_hist_n)

        K_n = curv_vect(DF_curv_hist_n)

        curv_hist.loc['curv_historica', PART] = np.nanmax(K_n[:-2]) + 3*np.nanstd(K_n[:-2])

    return curv_hist

# -----------------------------------
def predecir_el_previous_pool(df, numero_de_frames_validos, primer_frame_valido, ultimo_frame_valido, current_frame, trayectorias_vivas, previous_pool):
    """
    Predice la ubicacion de los puntos del previous_pool
    """
    
    col_names = df.columns
    for j in range(df.shape[1] // 2):
        if (
            numero_de_frames_validos[2 * j] >= 2
            and ultimo_frame_valido[2 * j] == current_frame - 1
        ):
            spline_order = 1
            previous_points = df.loc[
                int(primer_frame_valido[2 * j]) : current_frame - 1,
                col_names[[2 * j,2 * j + 1]],
            ].values.astype(float)
            t = np.arange(numero_de_frames_validos[2 * j])
            xsp = InterpolatedUnivariateSpline(
                t, previous_points[:, 0].flatten(), ext=0, k=spline_order
            )
            ysp = InterpolatedUnivariateSpline(
                t, previous_points[:, 1].flatten(), ext=0, k=spline_order
            )
            xe = xsp(t[-1] + 1)
            ye = ysp(t[-1] + 1)
            indice = np.sum(trayectorias_vivas[: 2 * j]) // 2
            previous_pool[2 * indice] = xe
            previous_pool[2 * indice + 1] = ye

    return previous_pool

# -----------------------------------
def generar_current_pool(dfo, current_frame):
    """
    Genera el current_pool para el current_frame.
    """

    # current_pool lo tomamos de los datos de entrada, disponiendolo en la misma
    # forma que previous_pool: con columnas adyacentes que tienen (x,y) de cada particula.

    idx_current = dfo.index.get_loc(current_frame)
    xori = dfo.iloc[idx_current, 1::2].dropna().values.astype(float).flatten()
    yori = dfo.iloc[idx_current, 2::2].dropna().values.astype(float).flatten()
    Nro_candidatos = len(xori) 

    current_pool = np.empty(Nro_candidatos * 2)
    current_pool[0::2] = xori
    current_pool[1::2] = yori

    indice_candidatos = crear_indice_candidatos(Nro_candidatos)

    return current_pool, Nro_candidatos, indice_candidatos 

