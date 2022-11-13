# per[pendicular]P[article]T[racking]V[elocimetry]

**perPTV** is a Python code to detect and track individual particles in 3D.
From a pair of movies, capturing two perpendicular views of the same process, it first performs the detection and reconstruction of 2D tracks.
Then, using the redudant information (as one coordinate of 3D space is recorded simultaneously by each view), the three-dimensional evolution of the particles is recovered.

- For the **detection** stage, processing parameters should be input using the `PTV_parameters.yaml` file (see template).
- Parameters for **2D tracking** and **3D reconstruction** are passed when calling the corresponding functions.

To run the detection and 2D tracking,
```Python
import PTV
# Define parameters for detection and 2D tracking
# See docs for a detailed explanation
path              = 'path to PTV_parameters.yaml file'
tol               = 20
max_nan	          = 5
min_frames        = 3
writing_threshold = 20
  
DF_objects,\
DF_trajectories,\
datasetID = PTV.analyze_by_PTV(path, tol=tol, max_nan_max_nan,\
                               min_frames=min_frames,\
                               writing_threshold=writing_threshold,\
                               use_predictive=True,\
                               use_curvature_history=False,\
                               print_progress=True)
```

Once the 2D tracking was performed in each view, having trajectories `(X_v1_i, Y_v1_i)` and `(X_v2_j, Y_v2_j)` (where `i,j` are trajectory labels and `v1` and `v2` indicate view 1 or 2, respectively), and assuming the redudant coordinate corresponds to the `Y` coordinate, to run the 3D reconstruction:
```Python
import pandas as pd
import matching_3D

DF_v1 = pd.read_pickle('path_to_view1_trajectories_DataFrame')
DF_v2 = pd.read_pickle('path_to_view2_trajectories_DataFrame')

# Parameters for trajectory matching
maxdif_len  = 100 # Max. value of length difference between two trajectories
dist_umbral = 10 # distance threshold for finding candidates.

DF_3D = matching_3D.get_df_3D(df_X=DF_v1, df_Y=DF_v2, maxdif_len=maxdif_len, dist_umbral=dist_umbral)

# Note this function *does not* save the 3D DataFrame.
``` 

---
The code was developed to track point-like particles and spherical, finite-size objects in turbulence experiments. 
It has been mostly used to study particle dynamics in a turbulent von Kármán experiment, set up in the Fluids and Plasmas laboratory of the Physics Department, University of Buenos Aires. Some of the results can be found in the following references:

1. S. Angriman, A. Ferran, F. Zapata, P. J. Cobelli, M. Obligado & P. D. Mininni. [_Clustering in
laboratory and numerical turbulent swirling flows_](https://doi.org/10.1103/PhysRevFluids.5.064605). J. Fluid Mech., Volume 948, A30 (2022).

2. S. Angriman, P. D. Mininni & P.J. Cobelli. [_Multitime structure functions and the Lagrangian scaling of turbulence_](https://doi.org/10.1103/PhysRevFluids.7.064603). Phys. Rev. Fluids 7, 064603 (2022).

3. S. Angriman, P. J. Cobelli, M. Bourgoin, S. G. Huisman, R. Volk & P. D. Mininni. [_Broken mirror symmetry of tracer's trajectories in turbulence_](https://doi.org/10.1103/PhysRevLett.127.254502). Phys. Rev. Lett. 127, 254502 (2021).
