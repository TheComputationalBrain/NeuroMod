
import os
#specify the number of threads before importing numpy to limit the amount of ressources that are taken up by numpy.
os.environ["OMP_NUM_THREADS"] = "1" 
os.environ["OPENBLAS_NUM_THREADS"] = "1"  
os.environ["MKL_NUM_THREADS"] = "1" 
os.environ["VECLIB_MAXIMUM_THREADS"] = "1" 
os.environ["NUMEXPR_NUM_THREADS"] = "1"
import numpy as np
from scipy.stats import zscore
import concurrent.futures
from scipy.spatial.distance import cdist
from sklearn.linear_model import LinearRegression
from neuromaps import nulls, transforms
from params_and_paths import Paths, Params, Receptors
import fmri_funcs as fun
from sklearn.metrics import r2_score
from scipy.stats import zscore
import utils.main_funcs as mf
import glob
import pickle
import pandas as pd
import nibabel as nib

paths = Paths()
params = Params()
rec = Receptors()

n_spins = 1000
EXPLORE_MODEL = 'noEntropy_noER'

tasks = ['EncodeProb', 'NAConf', 'PNAS', 'Explore']  
#tasks = ['NAConf']
output_dir = os.path.join(paths.home_dir, 'variance_explained')
os.makedirs(output_dir, exist_ok=True)

fmri_dir = {'NAConf': os.path.join('/neurospin/unicog/protocols/IRMf', 'MeynielMazancieux_NACONF_prob_2021', 'derivatives'),
            'EncodeProb': os.path.join('/neurospin/unicog/protocols/IRMf', 'EncodeProb_BounmyMeyniel_2020', 'derivatives'),
            'Explore': os.path.join('/neurospin/unicog/protocols/IRMf', 'Explore_Meyniel_Paunov_2021', 'bids/derivatives/fmriprep-23.1.3_MAIN'),
            'PNAS': os.path.join('/neurospin/unicog/protocols/IRMf', 'Meyniel_MarkovGuess_2014', 'MRI_data/analyzed_data')}


ignore = {'NAConf': [3, 5, 6, 9, 15, 30, 36, 40, 42, 43, 51, 59], #30 and 43 are removed because of their low coverage (also 15, 40, 42, 59)
            'EncodeProb': [1, 4, 12, 20],
            'Explore': [9, 17, 46],
            'PNAS': []}

receptor_dir = os.path.join(paths.home_dir, 'receptors', rec.source)
receptor_data = zscore(np.load(os.path.join(receptor_dir, f'receptor_density_{params.mask}_surf.pickle'), allow_pickle=True))
index_array = np.arange(receptor_data.shape[0])
spins = nulls.alexander_bloch(index_array, atlas='fsaverage', density='41k', n_perm=n_spins)
latent_vars = ['surprise', 'confidence']

def process_task(task):
    beta_dir = os.path.join(paths.home_dir, task, params.mask, 'first_level', EXPLORE_MODEL if task == 'Explore' else '')
    subjects = [subj for subj in mf.get_subjects(task, fmri_dir[task]) if subj not in ignore[task]]
    add_info = '_firstTrialsRemoved' if task == 'NAConf' else ''
    
    for latent_var in latent_vars:
        print(f"--- Spin test for {task} and {latent_var} ---")
        fmri_files_all = sorted(glob.glob(os.path.join(beta_dir,f'sub-*_{latent_var}_{params.mask}_effect_size_map{add_info}.nii.gz')))
        fmri_files = []
        for file in fmri_files_all:
            basename = os.path.basename(file)
            subj_str = basename.split('_')[0]  # 'sub-XX'
            subj_id = int(subj_str.split('-')[1])  # XX as integer
            if subj_id in subjects:
                fmri_files.append(file)        
        
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
            
            for i in range(len(subjects)):
                X_train, y_train = [], []
                for j in range(len(subjects)):
                    if j != i:
                        mask = ~np.logical_or(np.isnan(fmri_activity[j]), np.isclose(fmri_activity[j], 0)).flatten()
                        X_train.append(receptor_spin[mask])
                        y_train.append(zscore(fmri_activity[j].flatten()[mask]))
                X_train, y_train = np.concatenate(X_train), np.concatenate(y_train)
                mask_test = ~np.logical_or(np.isnan(fmri_activity[i]), np.isclose(fmri_activity[i], 0)).flatten()
                X_test, y_test = receptor_spin[mask_test], zscore(fmri_activity[i].flatten()[mask_test])
                model = LinearRegression(n_jobs=1)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                r2_scores.append(r2_score(y_test, y_pred))
            
            all_null.append(r2_scores)
        
        with open(os.path.join(output_dir, f'{task}_{latent_var}_all_regression_null_cv_r2.pickle'), "wb") as fp:
            pickle.dump(all_null, fp)

with concurrent.futures.ProcessPoolExecutor() as executor:
    executor.map(process_task, tasks)

    

