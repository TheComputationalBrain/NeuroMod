"""
This script explores the pverlap of significant clusters to explore the invariance of confidence and suprise effects across datasets
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nibabel as nib
from pathlib import Path
from nilearn import plotting, datasets, surface
from matplotlib.colors import ListedColormap, BoundaryNorm
from scipy.stats import pearsonr
from neuromaps import transforms
parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))
import utils.main_funcs as mf
from config.loader import load_config


# --- Global configuration ---
RESOLUTION = 'fsaverage'
FWHM = 5
TRESH = -np.log10(0.05)
VMAX = -np.log10(1 / 100000)
fsaverage = datasets.fetch_surf_fsaverage(mesh=RESOLUTION)


# Plot individual significant clusters per dataset & contrast
def plot_individual_clusters(tasks, contrasts, output_dir):
    for contrast in contrasts:
        cmap = 'Blues' if '_neg' in contrast else 'Reds'
        for task in tasks:
            params, paths, _ = load_config(task, return_what='all')
            _, add_info = mf.get_beta_dir_and_info(task, params, paths)
            group_dir = os.path.join(paths.home_dir, task, 'second_level')

            img_path = os.path.join(group_dir, f'{contrast}_logp_max_mass{add_info}_{FWHM}.nii.gz')
            img = nib.load(img_path)

            # Whole brain (both hemispheres)
            plotting.plot_img_on_surf(
                img, surf_mesh=RESOLUTION,
                hemispheres=['left', 'right'], views=['lateral', 'medial'],
                threshold=TRESH, vmax=VMAX, title=f"{task} {contrast}",
                colorbar=True, inflate=True, cmap=cmap, cbar_tick_format='%.2f'
            )
            out_path = os.path.join(output_dir, 'plots_by_task')
            os.makedirs(out_path, exist_ok=True)
            plt.savefig(os.path.join(out_path, f'{task}_{contrast}_cluster_mass.png'), dpi=300)
            plt.close('all')

            # Right hemisphere detailed
            texture = surface.vol_to_surf(img, fsaverage.pial_right)
            fig = plotting.plot_surf_stat_map(
                fsaverage.infl_right, texture, bg_map=fsaverage.sulc_right,
                hemi='right', threshold=TRESH, vmax=VMAX, title=f"{task} {contrast}",
                colorbar=True, cmap=cmap, cbar_tick_format='%.2f'
            )
            for ax in plt.gcf().axes:
                for coll in ax.collections:
                    coll.set_rasterized(True)
            
            mf.save_figure(fig, output_dir, f'{task}_{contrast}_cluster_mass_right')


# Plot overlap across all datasets
def plot_cluster_overlap_all(tasks, contrasts, output_dir):
    for contrast in contrasts:
        for hemi in ['right', 'left']:
            task_masks = []
            for task in tasks:
                params, paths, _ = load_config(task, return_what='all')
                _, add_info = mf.get_beta_dir_and_info(task, params, paths)
                group_dir = os.path.join(paths.home_dir, task, 'second_level')
                img_path = os.path.join(group_dir, f'{contrast}_logp_max_mass{add_info}_{FWHM}.nii.gz')
                img = nib.load(img_path)
                texture = surface.vol_to_surf(img, getattr(fsaverage, f'pial_{hemi}'))
                task_masks.append(texture > TRESH)

            # compute overlap
            conditions_array = np.sum(task_masks, axis=0)

            cmap = ListedColormap(['#4793AF', '#FFC470', '#6C946F', '#DD5746'])

            # Save plots
            for view in ['lateral', 'medial']:
                mesh = getattr(fsaverage, f'infl_{hemi}')
                bg_map = getattr(fsaverage, f'sulc_{hemi}')
                fig = plotting.plot_surf_roi(
                    surf_mesh=mesh, roi_map=conditions_array, bg_map=bg_map,
                    cmap=cmap, vmin=1, vmax=4, hemi=hemi, view=view
                )
                for ax in plt.gcf().axes:
                    for coll in ax.collections:
                        coll.set_rasterized(True)
                mf.save_figure(fig, output_dir, f'{contrast}_allOverlap_{hemi}_{view}')


# Plot colorbar legend for overlap
def plot_colorbar_overlap():
    categories = [0, 1, 2, 3]
    labels = ["no overlap", "2 overlap", "3 overlap", "all overlap"]
    cmap = ListedColormap(['#4793AF', '#FFC470', '#6C946F', '#DD5746'])
    norm = BoundaryNorm(categories + [len(categories)], cmap.N)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    fig, ax = plt.subplots() 
    cbar = plt.colorbar(sm, cax=ax, orientation='vertical')
    cbar.set_ticks([0.5, 1.5, 2.5, 3.5])
    cbar.ax.set_yticklabels(labels)
    cbar.ax.tick_params(labelsize=24)

    return fig, ax

#correalte effect maps 
def run_correlations(tasks, output_dir):
    data_list = []
    for task in tasks:
        params, paths, _ = load_config(task, return_what='all')
        _, add_info = mf.get_beta_dir_and_info(task, params, paths)
        base_dir = os.path.join(paths.home_dir, task, 'second_level')
        for fname in os.listdir(base_dir):
            for contrast in ['confidence', 'surprise']:
                if fname.endswith(f"{contrast}_effect_map{add_info}.nii.gz"):
                    img = nib.load(os.path.join(base_dir, fname))
                    surf_data = transforms.mni152_to_fsaverage(img, fsavg_density='41k')
                    data = np.hstack([d.agg_data().T for d in surf_data])
                    data_list.append({'task': task, 'contrast': contrast, 'data': data})

    df = pd.DataFrame(data_list)
    n = len(df)
    corr_matrix = np.zeros((n, n))

    # --- Compute correlations ---
    for i in range(n):
        for j in range(n):
            d1, d2 = df.iloc[i]['data'], df.iloc[j]['data']
            mask = (d1 != 0) & (d2 != 0)
            if np.any(mask):
                corr_matrix[i, j] = pearsonr(d1[mask], d2[mask])[0]
            else:
                corr_matrix[i, j] = np.nan

    labels = [f"{r.task}_{r.contrast}" for _, r in df.iterrows()]
    corr_df = pd.DataFrame(corr_matrix, index=labels, columns=labels)
    corr_df.to_csv(os.path.join(output_dir, 'correlation_df.csv'))


# Group-level correlation analysis (modular)
def plot_correlations(plots_to_generate=["all"], cmap="RdYlBu", paths = None, output_dir=""):
    """
    Generate correlation plots between second-level maps across tasks and contrasts.

    plots_to_generate: list of strings controlling which plots are created.
        Options include:
        - "all": full correlation heatmap (default)
        - "contrast_sorted": heatmap sorted by contrast/task
        - "confidence": submatrix for confidence contrasts
        - "surprise": submatrix for surprise contrasts
        - "cross": cross correlation between confidence and surprise
        - "lower_triangles": masked lower-triangle plots for publication-style visuals
    """
    # --- get results ---
    corr_df = pd.read_csv(os.path.join(paths.home_dir, 'domain_general', 'correlation_df.csv'), index_col = [0])
    labels = corr_df.index

    # --- Helper for saving ---
    def save_heatmap(
        matrix,
        fname,
        title=None,
        mask=None,
        annot=True,
        rename_tasks=False,
    ):
        """
        Saves a heatmap for a matrix.
        
        If rename_tasks=True, any label containing a task name from
        params.study_mapping will be replaced with the corresponding study name.
        """
        # Copy to avoid modifying original
        matrix_to_plot = matrix.copy()

        if rename_tasks:
            new_index = [
                next((params.study_mapping[t] for t in params.study_mapping if t in str(lbl)), lbl)
                for lbl in matrix_to_plot.index
            ]
            new_columns = [
                next((params.study_mapping[t] for t in params.study_mapping if t in str(lbl)), lbl)
                for lbl in matrix_to_plot.columns
            ]
            matrix_to_plot.index = new_index
            matrix_to_plot.columns = new_columns

        data = matrix_to_plot.values
        max_abs = np.max(np.abs(data))
        vmin = -max_abs
        vmax = max_abs

        fig = plt.figure()
        ax = sns.heatmap(
            matrix_to_plot,
            annot=annot,
            fmt=".2f",
            cmap=cmap,
            center=0,
            square=True,
            mask=mask,
            linewidth=1,
            vmin=vmin, 
            vmax=vmax,   
            cbar_kws=dict(label = "Pearson's correlation", shrink= 0.6)
        )

        # Get colorbar
        cbar = ax.collections[0].colorbar  

        cbar = ax.collections[0].colorbar
        cbar.set_ticks([vmin, vmax])
        cbar.set_ticklabels([f"{vmin:.2f}", f"{vmax:.2f}"])
        cbar.ax.tick_params(pad=1)

        cbar.set_label("Pearson's correlation", labelpad=-12)  
        cbar.ax.yaxis.label.set_verticalalignment('center')

        # Rotate tick labels to avoid overlap
        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=45)

        if title:
            plt.title(title)

        mf.save_figure(fig, output_dir, fname)
        plt.close(fig)  

    # --- 1. Full heatmap ---
    if "all" in plots_to_generate:
        save_heatmap(corr_df, 'correlation_heatmap_all', 'All contrasts & tasks', annot=True)

    # --- 2. Contrast-sorted ---
    if "contrast_sorted" in plots_to_generate:
        order = sorted(labels, key=lambda x: ('confidence' not in x, x))
        corr_sorted = corr_df.loc[order, order]
        save_heatmap(corr_sorted, 'correlation_heatmap_contrast_sorted', 'Contrast-sorted correlations')

    # --- 3. Confidence-only ---
    if "confidence" in plots_to_generate:
        conf_labels = [l for l in labels if "confidence" in l]
        conf_df = corr_df.loc[conf_labels, conf_labels]
        save_heatmap(conf_df, 'correlation_confidence_only', 'Confidence contrasts')

    # --- 4. Surprise-only ---
    if "surprise" in plots_to_generate:
        surpr_labels = [l for l in labels if "surprise" in l]
        surpr_df = corr_df.loc[surpr_labels, surpr_labels]
        save_heatmap(surpr_df, 'correlation_surprise_only', 'Surprise contrasts')

    # --- 5. Cross correlation between confidence and surprise ---
    if "cross" in plots_to_generate:
        conf_labels = [l for l in labels if "confidence" in l]
        surpr_labels = [l for l in labels if "surprise" in l]
        cross_df = corr_df.loc[conf_labels, surpr_labels]
        save_heatmap(cross_df, 'correlation_conf_surprise_cross', None, mask = None, rename_tasks=True)

    # --- 6. Lower-triangle versions ---
    if "lower_triangles" in plots_to_generate:
        conf_labels = [l for l in labels if "confidence" in l]
        surpr_labels = [l for l in labels if "surprise" in l]

        conf_df = corr_df.loc[conf_labels, conf_labels]
        mask_conf = np.triu(np.ones_like(conf_df, dtype=bool))
        save_heatmap(conf_df, 'correlation_confidence_lower_triangle', None, mask=mask_conf, rename_tasks=True)

        surpr_df = corr_df.loc[surpr_labels, surpr_labels]
        mask_surpr = np.triu(np.ones_like(surpr_df, dtype=bool))
        save_heatmap(surpr_df, 'correlation_surprise_lower_triangle', None, mask=mask_surpr, rename_tasks=True)


def run_analysis(
    run_individual=False,
    run_overlap=False,
    run_colorbar=False,
    run_correlations=False,
    params = None,
    paths = None):
    """wrapper to run selected analyses."""

    tasks=params.tasks
    contrasts = ['surprise', 'confidence', 'surprise_neg', 'confidence_neg']

    OUTPUT_DIR = os.path.join(paths.home_dir, 'domain_general')
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    mf.set_publication_style(font_size=8, layout="2-across")

    if run_individual:
        plot_individual_clusters(tasks, contrasts, OUTPUT_DIR) #saves internally by task and latent variable

    if run_overlap:
        plot_cluster_overlap_all(tasks, contrasts, OUTPUT_DIR) #saves internally by hemisphere

    if run_colorbar:
        fig, ax = plot_colorbar_overlap()
        mf.save_figure(fig, OUTPUT_DIR, 'colorbar_overlap_4cat')

    if run_correlations:
        run_correlations(tasks,OUTPUT_DIR)
        cmap_div = mf.get_custom_colormap('diverging')
        plot_correlations(["cross","lower_triangles"], cmap_div, paths, OUTPUT_DIR)

if __name__ == "__main__":
    params, paths, _ = load_config('all', return_what='all')

    run_analysis(run_individual=True, run_overlap=True ,run_colorbar=True, run_correlations=True, params=params, paths=paths)
