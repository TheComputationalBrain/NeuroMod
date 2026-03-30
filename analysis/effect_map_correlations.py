"""
This script contains all functions to explore the correlations between group level effect maps to explore the invariance of confidence and suprise representations across datasets.
"""

import os
import sys
import itertools
import numpy as np
import pandas as pd
import nibabel as nib
from nilearn.surface import SurfaceImage
from statsmodels.stats.multitest import multipletests
import utils.main_funcs as mf

# Local import

# Load local debugged version of neuromaps
sys.path.insert(0, os.path.abspath("."))
from neuromaps import nulls, transforms, stats

from config.loader import load_config

# Null model generation
def generate_null_model(surf_map, n_perm=1000, seed=1234):
    """Generate a random-spin null model for one surface map."""
    return nulls.alexander_bloch([surf_map], atlas="fsaverage", density="41k", n_perm=n_perm, seed=seed)


# Within-contrast spin tests (confidence-confidence, surprise-surprise)
def run_spin_test_within(tasks, contrasts, paths, explore_model="noEntropy_noER", n_perm=1000, seed=1234):
    """Run spin tests between all task pairs for each contrast separately."""
    spin_results = []

    for contrast in contrasts:
        surf_maps, null_models = {}, {}

        # Prepare surface maps and nulls
        for task in tasks:
            surf_map = mf.load_group_surface_map(task, contrast, paths, explore_model)
            surf_maps[task] = surf_map
            null_models[task] = generate_null_model(surf_map, n_perm, seed)

        # Pairwise comparisons
        for task1, task2 in itertools.combinations(tasks, 2):
            print(f"--- Spin test for {contrast}: {task1} vs {task2} ---")
            surf_map1, surf_map2 = [surf_maps[task1]], [surf_maps[task2]]
            rotated = null_models[task1]
            corr, pval = stats.compare_images(surf_map1, surf_map2, nulls=rotated)
            spin_results.append({
                "task1": task1,
                "task2": task2,
                "contrast": contrast,
                "corr": corr,
                "pval": pval,
            })

    results_df = pd.DataFrame(spin_results)
    reject, pvals_fdr, _, _ = multipletests(results_df["pval"], method="fdr_bh")
    results_df["pval_fdr"] = pvals_fdr
    return results_df

# Cross-contrast spin tests (confidence-surprise and vice versa)
def run_spin_test_across(tasks, contrasts, paths, explore_model="noEntropy_noER", n_perm=1000, seed=1234):
    """Run spin tests between confidence and surprise maps across all tasks."""
    surf_maps, null_models = {}, {}

    # Load all maps and generate nulls
    for task in tasks:
        for contrast in contrasts:
            surf_map = mf.load_group_surface_map(task, contrast, paths, explore_model)
            key = f"{task}_{contrast}"
            surf_maps[key] = surf_map
            null_models[key] = generate_null_model(surf_map, n_perm, seed)

    # Compare across contrasts
    spin_results = []
    for task1, task2 in itertools.product(tasks, repeat=2):
        # confidence(task1) vs surprise(task2)
        for c1, c2 in [("confidence", "surprise"), ("surprise", "confidence")]:
            map1_key, map2_key = f"{task1}_{c1}", f"{task2}_{c2}"
            surf_map1, surf_map2 = [surf_maps[map1_key]], [surf_maps[map2_key]]
            rotated = null_models[map1_key]
            corr, pval = stats.compare_images(surf_map1, surf_map2, nulls=rotated)

            spin_results.append({
                "task1": task1, "contrast1": c1,
                "task2": task2, "contrast2": c2,
                "corr": corr, "pval": pval,
            })

    results_df = pd.DataFrame(spin_results)
    reject, pvals_fdr, _, _ = multipletests(results_df["pval"], method="fdr_bh")
    results_df["pval_fdr"] = pvals_fdr
    return results_df


def run_all_spin_tests(
    within=True,
    paths = None,
    output_dir=None,
    tasks=None,
    contrasts=None,
    explore_model="noEntropy_noER",
    n_perm=1000,
    seed=1234,
):
    """wrapper to run spin tests (within or across contrasts)."""

    if tasks is None:
        tasks = ["EncodeProb", "NAConf", "PNAS", "Explore"]
    if contrasts is None:
        contrasts = ["surprise", "confidence"]

    if output_dir is None:
        output_dir = os.path.join(paths.home_dir, "domain_general")
    os.makedirs(output_dir, exist_ok=True)

    if within:
        results_df = run_spin_test_within(tasks, contrasts, paths, explore_model, n_perm, seed)
        fname = "results_map_spin_within.csv"
    else:
        results_df = run_spin_test_across(tasks, contrasts, paths, explore_model, n_perm, seed)
        fname = "results_map_spin_across.csv"

    results_df.to_csv(os.path.join(output_dir, fname), index=False)
    return results_df


if __name__ == "__main__":
    paths = load_config('all', return_what='paths')
    # Run within-contrast correlations (surprise-surprise, confidence-confidence)
    run_all_spin_tests(within=True, paths=paths)

    # Run cross-contrast correlations (confidence-surprise)
    run_all_spin_tests(within=False, paths=paths)
