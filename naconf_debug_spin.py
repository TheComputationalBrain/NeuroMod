import os
os.environ["OMP_NUM_THREADS"] = "1" 
os.environ["OPENBLAS_NUM_THREADS"] = "1"  
os.environ["MKL_NUM_THREADS"] = "1" 
os.environ["VECLIB_MAXIMUM_THREADS"] = "1" 
os.environ["NUMEXPR_NUM_THREADS"] = "1"
import glob
import numpy as np
import pandas as pd
import pickle
import nibabel as nib
from neuromaps import transforms
from math import log
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from scipy.stats import zscore, sem, ttest_rel, ttest_ind
from scipy.stats import ttest_ind
import utils.main_funcs as mf
from params_and_paths import Paths, Params, Receptors
import matplotlib.pyplot as plt 
from matplotlib import cm
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from nilearn import plotting, surface
from nilearn import datasets
from scipy.stats import pearsonr
from neuromaps import nulls, transforms


model_type = 'linear'# 'linear', 'poly2', 'lin+quad', 'lin+interact'
n_spins = 1000
ON_SURFACE = True
WEIGHT_MODE = "subjects"   # options: "voxels" or "subjects" 


if ON_SURFACE:
    proj = '_on_surf'
else:
    proj = ''

suffix = ''

paths = Paths()
params = Params()
rec = Receptors()

output_dir = os.path.join(paths.home_dir, 'variance_explained')
if not os.path.exists(output_dir):
        os.makedirs(output_dir) 

fmri_dir = {'NAConf': os.path.join('/neurospin/unicog/protocols/IRMf', 'MeynielMazancieux_NACONF_prob_2021', 'derivatives'),
            'EncodeProb': os.path.join('/neurospin/unicog/protocols/IRMf', 'EncodeProb_BounmyMeyniel_2020', 'derivatives'),
            'Explore': os.path.join('/neurospin/unicog/protocols/IRMf', 'Explore_Meyniel_Paunov_2021', 'bids/derivatives/fmriprep-23.1.3_MAIN'),
            'PNAS': os.path.join('/neurospin/unicog/protocols/IRMf', 'Meyniel_MarkovGuess_2014', 'MRI_data/analyzed_data'),
            'lanA': '/home_local/alice_hodapp/language_localizer/'}

ignore = {'NAConf': [3, 5, 6, 9, 15, 30, 36, 40, 42, 43, 51, 59], #30 and 43 are removed because of their low coverage (also 15, 40, 42, 59)
            'EncodeProb': [1, 4, 12, 20],
            'Explore': [9, 17, 46],
            'PNAS': [],
            'lanA' : [88]} 


#get Naconf common mask
task = 'NAConf'
beta_dir  = os.path.join(paths.home_dir,task,params.mask,'first_level')

subjects =  mf.get_subjects(task, fmri_dir[task])
subjects = [subj for subj in subjects if subj not in ignore[task]] 

if task in ['NAConf']:
    add_info = '_firstTrialsRemoved'
elif not params.zscore_per_session:
    add_info = '_zscoreAll'
else:
    add_info = ""    

receptor_dir = os.path.join(paths.home_dir, 'receptors', rec.source) 
receptor_density =zscore(np.load(os.path.join(receptor_dir,f'receptor_density_{params.mask}_surf.pickle'), allow_pickle=True))
mask_comb = params.mask 

n_features = receptor_density.shape[1]

latent_var = 'surprise'
r2_scores = [] 
if ON_SURFACE:
    if task == 'lanA':
        fmri_files = []
        for subj in subjects:
            subj_id = f"{subj:03d}"  
            pattern = os.path.join(beta_dir, 'subjects', subj_id, 'SPM', 'spmT_*.nii')
            fmri_files.extend(glob.glob(pattern))
    else:
        fmri_files_all = sorted(glob.glob(os.path.join(beta_dir,f'sub-*_{latent_var}_{mask_comb}_effect_size_map{add_info}.nii.gz')))
        fmri_files = []
        for file in fmri_files_all:
            basename = os.path.basename(file)
            subj_str = basename.split('_')[0]  # 'sub-XX'
            subj_id = int(subj_str.split('-')[1])  # XX as integer
            if subj_id in subjects:
                fmri_files.append(file)
    fmri_activity = []
    for file in fmri_files:
        data_vol = nib.load(file)
        effect_data = transforms.mni152_to_fsaverage(data_vol, fsavg_density='41k')
        data_gii = []
        for img in effect_data:
            data_hemi = img.agg_data()
            data_hemi = np.asarray(data_hemi).T
            data_gii += [data_hemi]
            effect_array = np.hstack(data_gii)    
        fmri_activity.append(effect_array) 
    valid_counts = []
    for idx, arr in enumerate(fmri_activity):
        mask_valid = ~np.logical_or(np.isnan(arr), np.isclose(arr, 0)).flatten()
        n_valid = np.sum(mask_valid)
        valid_counts.append(n_valid)
else:    
    fmri_files = sorted(glob.glob(os.path.join(beta_dir,f'sub-*_{latent_var}_{mask_comb}_effect_size{add_info}.pickle')))
    fmri_activity = []
    for file in fmri_files:
        with open(file, 'rb') as f:
            fmri_activity.append(pickle.load(f))  

all_valid_masks = [
    ~np.logical_or(np.isnan(arr), np.isclose(arr, 0)).flatten()
    for arr in fmri_activity
]
common_mask = np.logical_and.reduce(all_valid_masks)

#apply Naconf mask to EncodeProb

task = 'EncodeProb'

beta_dir  = os.path.join(paths.home_dir,task,params.mask,'first_level')

subjects =  mf.get_subjects(task, fmri_dir[task])
subjects = [subj for subj in subjects if subj not in ignore[task]]   

receptor_dir = os.path.join(paths.home_dir, 'receptors', rec.source) 
receptor_density =zscore(np.load(os.path.join(receptor_dir,f'receptor_density_{params.mask}_surf.pickle'), allow_pickle=True))
mask_comb = params.mask 

with open(os.path.join(output_dir,f'debug_encode_with_naconf_mask.txt'), "a") as outfile:
    outfile.write(f'{task}: variance explained in analysis with {rec.source} as predictor:\n\n')
    n_features = receptor_density.shape[1]
    r2_scores = [] 
    if ON_SURFACE:
        if task == 'lanA':
            fmri_files = []
            for subj in subjects:
                subj_id = f"{subj:03d}"  
                pattern = os.path.join(beta_dir, 'subjects', subj_id, 'SPM', 'spmT_*.nii')
                fmri_files.extend(glob.glob(pattern))
        else:
            fmri_files_all = sorted(glob.glob(os.path.join(beta_dir,f'sub-*_{latent_var}_{mask_comb}_effect_size_map.nii.gz')))
            fmri_files = []
            for file in fmri_files_all:
                basename = os.path.basename(file)
                subj_str = basename.split('_')[0]  # 'sub-XX'
                subj_id = int(subj_str.split('-')[1])  # XX as integer
                if subj_id in subjects:
                    fmri_files.append(file)
        fmri_activity = []
        for file in fmri_files:
            data_vol = nib.load(file)
            effect_data = transforms.mni152_to_fsaverage(data_vol, fsavg_density='41k')
            data_gii = []
            for img in effect_data:
                data_hemi = img.agg_data()
                data_hemi = np.asarray(data_hemi).T
                data_gii += [data_hemi]
                effect_array = np.hstack(data_gii)    
            fmri_activity.append(effect_array) 
    
    #sanity checks for NAConf
    fmri_activity_masked = [arr.flatten()[common_mask] for arr in fmri_activity]
    receptor_density_masked = receptor_density[common_mask]

    for i in range(len(subjects)):
        X_train, y_train, weights = [], [], []

        for j in range(len(subjects)):
            if j == i:
                continue
            Xj = zscore(receptor_density_masked)
            yj = zscore(fmri_activity_masked[j])

            # weight = 1 / n_voxels for this subject
            subj_weight = np.ones(len(yj)) / len(yj)

            X_train.append(Xj)
            y_train.append(yj)
            if WEIGHT_MODE == "subjects":
                subj_weight = np.ones(len(yj)) / len(yj)   # scale subject to weight=1
                weights.append(subj_weight)

        # Concatenate everything
        X_train = np.concatenate(X_train)
        y_train = np.concatenate(y_train)
        if WEIGHT_MODE == "subjects":
            weights = np.concatenate(weights)
        else:
            weights = None

        # Test subject
        X_test = zscore(receptor_density_masked)
        y_test = zscore(fmri_activity_masked[i])

        # Fit with sample weights
        model = LinearRegression()
        model.fit(X_train, y_train, sample_weight=weights)
        y_pred = model.predict(X_test)
        # r2 = r2_score(y_test, y_pred)
        # r2_scores.append(r2)
        r, _ = pearsonr(y_test, y_pred)
        r2_scores.append(r**2)

    average_r2 = np.mean(r2_scores)
    sem_r2 = sem(r2_scores)
    outfile.write(f'{latent_var}: {average_r2}, sem: {sem_r2}\n\n\n')


    #run spin test
    receptor_dir = os.path.join(paths.home_dir, 'receptors', rec.source)
    receptor_data = zscore(np.load(os.path.join(receptor_dir, f'receptor_density_{params.mask}_surf.pickle'), allow_pickle=True))
    index_array = np.arange(receptor_data.shape[0])
    spins = nulls.alexander_bloch(index_array, atlas='fsaverage', density='41k', n_perm=n_spins)

    beta_dir  = os.path.join(paths.home_dir,task,params.mask,'first_level')    
        
    subjects = [subj for subj in mf.get_subjects(task, fmri_dir[task]) if subj not in ignore[task]]

    add_info = '_firstTrialsRemoved' if task == 'NAConf' else ''
    
    latent_vars = ['surprise']
    outfile.write(f'Null model results\n\n')

    for latent_var in latent_vars:
        print(f"--- Spin test for {task} and {latent_var} ---")
        if task == 'lanA':
            fmri_files = []
            for subj in subjects:
                subj_id = f"{subj:03d}"  
                pattern = os.path.join(beta_dir, 'subjects', subj_id, 'SPM', 'spmT_*.nii')
                fmri_files.extend(glob.glob(pattern))
        else:
            fmri_files = sorted(glob.glob(os.path.join(beta_dir,f'sub-*_{latent_var}_{params.mask}_effect_size_map{add_info}.nii.gz')))        

        fmri_activity = []
        all_null = []
        for file in fmri_files:
            data_vol = nib.load(file)
            effect_data = transforms.mni152_to_fsaverage(data_vol, fsavg_density='41k')
            data_gii = [img.agg_data().T for img in effect_data]
            fmri_activity.append(np.hstack(data_gii))

        for s in range(spins.shape[1]):
            receptor_spin = receptor_data[spins[:, s], :]
            r2_scores = []

            #sanity checks for NAConf
            fmri_activity_masked = [arr.flatten()[common_mask] for arr in fmri_activity]
            receptor_density_masked = receptor_spin[common_mask]

            for i in range(len(subjects)):
                X_train, y_train, weights = [], [], []

                for j in range(len(subjects)):
                    if j == i:
                        continue
                    Xj = zscore(receptor_density_masked)
                    yj = zscore(fmri_activity_masked[j])

                    # weight = 1 / n_voxels for this subject
                    subj_weight = np.ones(len(yj)) / len(yj)

                    X_train.append(Xj)
                    y_train.append(yj)
                    if WEIGHT_MODE == "subjects":
                        subj_weight = np.ones(len(yj)) / len(yj)   # scale subject to weight=1
                        weights.append(subj_weight)

                # Concatenate everything
                X_train = np.concatenate(X_train)
                y_train = np.concatenate(y_train)
                if WEIGHT_MODE == "subjects":
                    weights = np.concatenate(weights)
                else:
                    weights = None

                # Test subject
                X_test = zscore(receptor_density_masked)
                y_test = zscore(fmri_activity_masked[i])

                # Fit with sample weights
                model = LinearRegression()
                model.fit(X_train, y_train, sample_weight=weights)
                y_pred = model.predict(X_test)
                # r2 = r2_score(y_test, y_pred)
                # r2_scores.append(r2)
                r, _ = pearsonr(y_test, y_pred)
                r2_scores.append(r**2)
            
            all_null.append(r2_scores)
            null_df = pd.DataFrame(all_null)
            null_means = null_df.mean().tolist()
            r2 = sum(null_means) / len(null_means)
            outfile.write(f'Null for Encode: {r2}')
