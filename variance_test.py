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


FROM_BETA = False
FROM_RECEPTOR = True
COMP_NULL = False
COMPARE_LANG_LEARN = False
COMPARE_EXPL_VAR = False
PLOT_VAR_EXPLAINED = False
PLOT_PERCENT_VAR_EXPLAINED = False

model_type = 'linear'# 'linear', 'poly2', 'lin+quad', 'lin+interact'

ON_SURFACE = True

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

tasks = ['EncodeProb', 'NAConf', 'PNAS', 'Explore'] 

latent_vars = ['surprise', 'confidence']

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

comparison_latent = 'language'
comparison_task = 'lanA'

df = pd.DataFrame(index=tasks, columns=latent_vars)

for task in tasks: 
    if task == 'Explore':
        beta_dir  = os.path.join(paths.home_dir,task,params.mask,'first_level', 'noEntropy_noER')
    elif task == 'lanA':
        beta_dir = fmri_dir['lanA']
    else:
        beta_dir  = os.path.join(paths.home_dir,task,params.mask,'first_level')

    subjects =  mf.get_subjects(task, fmri_dir[task])
    subjects = [subj for subj in subjects if subj not in ignore[task]] 

    if task in ['NAConf']:
        add_info = '_firstTrialsRemoved'
    elif not params.zscore_per_session:
        add_info = '_zscoreAll'
    else:
        add_info = ""    

    if ON_SURFACE: 
        receptor_dir = os.path.join(paths.home_dir, 'receptors', rec.source) 
        receptor_density =zscore(np.load(os.path.join(receptor_dir,f'receptor_density_{params.mask}_surf.pickle'), allow_pickle=True))
        mask_comb = params.mask 
    else:                                            
        if params.parcelated:
            receptor_dir = os.path.join(paths.home_dir, 'receptors', rec.source)  
            mask_comb = params.mask + '_' + params.mask_details 
            receptor_density = zscore(np.load(os.path.join(receptor_dir,f'receptor_density_{mask_comb}.pickle'), allow_pickle=True), nan_policy='omit') 
            text = 'by region'
        else:
            receptor_dir = os.path.join(paths.home_dir, 'receptors', 'PET2') #vertex level analyis can only be run on PET data densities 
            receptor_density = zscore(np.load(os.path.join(receptor_dir,f'receptor_density_{params.mask}.pickle'), allow_pickle=True))
            mask_comb = params.mask 
            text = 'by voxel'

        if rec.source == 'autorad_zilles44':
            #autoradiography dataset is only one hemisphere 
            receptor_density = np.concatenate((receptor_density, receptor_density))

    with open(os.path.join(output_dir,f'predict_from_receptor{proj}.txt'), "a") as outfile:
        outfile.write(f'{task}: variance explained in analysis with {rec.source} as predictor:\n\n')
        n_features = receptor_density.shape[1]
        for latent_var in latent_vars:
            print(f'latent variable: {latent_var}')
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
            
            #sanity checks for NAConf
            #coverage, all_subjects_mask = plot_coverage_direct(fmri_activity, task, latent_var)
                
            all_valid_masks = [
                ~np.logical_or(np.isnan(arr), np.isclose(arr, 0)).flatten()
                for arr in fmri_activity
            ]
            common_mask = np.logical_and.reduce(all_valid_masks)

            # Apply the mask to all subjects and receptor density
            fmri_activity_masked = [arr.flatten()[common_mask] for arr in fmri_activity]
            receptor_density_masked = receptor_density[common_mask]

            print(f"Common mask retains {common_mask.sum()} voxels")

            for i in range(len(subjects)):
                X_train, y_train = [], []

                for j in range(len(subjects)):
                    if j != i:
                        X_train.append(receptor_density_masked)
                        y_train.append(zscore(fmri_activity_masked[j]))

                # Concatenate training data
                X_train = np.tile(receptor_density_masked, (len(y_train), 1))
                y_train = np.concatenate(y_train)

                # Test subject
                X_test = receptor_density_masked
                y_test = zscore(fmri_activity_masked[i])

                # Model
                if model_type == 'linear':
                    model = LinearRegression()
                elif model_type == 'poly2':
                    poly = PolynomialFeatures(degree=2, include_bias=False)
                    model = make_pipeline(poly, LinearRegression())
                else:
                    poly = PolynomialFeatures(degree=2, include_bias=False)
                    X_train_poly = poly.fit_transform(X_train)
                    X_test_poly = poly.transform(X_test)
                    feature_names = poly.get_feature_names_out(input_features=rec.receptor_names)

                    if model_type == 'lin+quad':
                        mask = [(" " not in name) or ("^" in name) for name in feature_names]
                    elif model_type == 'lin+interact':
                        mask = ["^" not in name for name in feature_names]

                    X_train = X_train_poly[:, mask]
                    X_test = X_test_poly[:, mask]
                    model = LinearRegression()

                # Fit & predict
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                r2 = r2_score(y_test, y_pred)
                r2_scores.append(r2)

            average_r2 = np.mean(r2_scores)
            sem_r2 = sem(r2_scores)
            outfile.write(f'{latent_var}: {average_r2}, sem: {sem_r2}\n')
            df.loc[task, latent_var] = average_r2

            if model_type == 'linear':
                with open(os.path.join(output_dir,f'{task}_{latent_var}_all_regression_cv_r2{proj}{suffix}.pickle'), "wb") as fp:   
                    pickle.dump(r2_scores, fp)
            else:
                with open(os.path.join(output_dir,f'{task}_{latent_var}_all_regression_cv_r2{proj}_{model_type}.pickle'), "wb") as fp:   
                    pickle.dump(r2_scores, fp)

        outfile.write('\n\n')
if model_type == 'linear':
    df.to_csv(os.path.join(output_dir,f'overview_regression_cv{proj}_commonmask.csv'))
else:
    df.to_csv(os.path.join(output_dir,f'overview_regression_cv{proj}_{model_type}_commonmask.csv'))