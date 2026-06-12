"""IsoTree JSON-routed feature selectors for Arborration benchmarks."""

from __future__ import annotations

import numpy as np
import pandas as pd
from isotree import IsolationForest


def select_features_for_task(X, y, *, task, **kwargs):
    """Dispatch to the task-appropriate feature selector."""
    if task == "classification":
        kwargs = dict(kwargs)
        kwargs.pop("regression_target_bins", None)
        kwargs.pop("n_target_permutations", None)
        kwargs.pop("n_permutation_refits", None)
        kwargs.pop("permutation_mode", None)
        kwargs.pop("permutation_scope", None)
        kwargs.pop("permutation_usage_margin", None)
        kwargs.pop("min_permutation_adjusted_usage", None)
        kwargs.pop("competitive_real_target_weight", None)
        kwargs.pop("feature_attribution_mode", None)
        kwargs.pop("leaf_signal_quantile", None)
        kwargs.pop("leaf_min_samples", None)
        kwargs.pop("leaf_depth_decay", None)
        return select_features_by_class_contrast_isotree_json_usage(X, y, **kwargs)
    if task == "regression":
        kwargs = dict(kwargs)
        kwargs.pop("max_samples_per_side", None)
        kwargs.pop("min_class_samples", None)
        kwargs.pop("combine_classes", None)
        return select_features_by_target_weighted_isotree_json_usage(
            X,
            y,
            task="regression",
            **kwargs,
        )
    raise ValueError("task must be 'classification' or 'regression'.")


def select_features_by_target_weighted_isotree_json_usage(
    X,
    y,
    *,
    task,
    ntrees=100,
    sample_size=256,
    ndim=2,
    max_iter=20,
    min_usage_count=1,
    n_refits=1,
    require_used_in_any_refit=True,
    target_draw_probability=0.25,
    regression_target_bins=10,
    feature_attribution_mode="routing_flip",
    leaf_signal_quantile=0.25,
    leaf_min_samples=2,
    leaf_depth_decay=0.0,
    random_state=42,
    nthreads=-1,
    min_features_to_keep=1,
    max_refit_usage_fraction_for_removal=0.0,
    max_remove_fraction_per_iter=0.10,
    validation_scorer=None,
    score_tolerance=0.0,
    n_target_permutations=0,
    n_permutation_refits=1,
    permutation_mode="separate",
    permutation_scope="borderline",
    permutation_usage_margin=1,
    min_permutation_adjusted_usage=0.0,
    competitive_real_target_weight=0.60,
    standardize_X=True,
    isotree_kwargs=None,
    verbose=True,
):
    """Select features that change routing at IsoTree splits anchored to targets."""
    if task not in {"classification", "regression"}:
        raise ValueError("task must be 'classification' or 'regression'.")
    if not (0.0 < target_draw_probability < 1.0):
        raise ValueError("target_draw_probability must be between 0 and 1.")
    if ndim < 2:
        raise ValueError("This target-anchored method is intended for ndim >= 2.")
    if feature_attribution_mode not in {"routing_flip", "leaf_backtrack"}:
        raise ValueError("feature_attribution_mode must be 'routing_flip' or 'leaf_backtrack'.")
    if not (0.0 < leaf_signal_quantile <= 1.0):
        raise ValueError("leaf_signal_quantile must be in (0, 1].")
    if leaf_min_samples < 1:
        raise ValueError("leaf_min_samples must be at least 1.")
    if leaf_depth_decay < 0:
        raise ValueError("leaf_depth_decay must be nonnegative.")
    if score_tolerance < 0:
        raise ValueError("score_tolerance must be nonnegative.")
    _validate_permutation_params(
        n_target_permutations=n_target_permutations,
        n_permutation_refits=n_permutation_refits,
        permutation_mode=permutation_mode,
        permutation_scope=permutation_scope,
        permutation_usage_margin=permutation_usage_margin,
        competitive_real_target_weight=competitive_real_target_weight,
    )

    isotree_kwargs = isotree_kwargs or {}
    _validate_conservative_removal_params(
        n_refits=n_refits,
        min_features_to_keep=min_features_to_keep,
        max_refit_usage_fraction_for_removal=max_refit_usage_fraction_for_removal,
        max_remove_fraction_per_iter=max_remove_fraction_per_iter,
    )

    X_raw = _to_frame(X)
    y = pd.Series(y).reset_index(drop=True)
    if len(X_raw) != len(y):
        raise ValueError("X and y must have the same number of rows.")

    X_work = _standardize_frame(X_raw) if standardize_X else X_raw.copy()
    original_features = list(X_work.columns)
    current_features = list(X_work.columns)
    Y_target = _make_target_frame(
        y,
        task=task,
        n_bins=regression_target_bins,
        existing_columns=original_features,
    )
    target_features = list(Y_target.columns)

    history = []
    removed_features_all = []

    for iteration in range(max_iter):
        X_current = X_work[current_features].reset_index(drop=True)
        Y_current = Y_target.reset_index(drop=True)
        usage_by_refit = []
        decoy_usage_by_refit = []

        for refit_ix in range(n_refits):
            seed = random_state + 10_000 * iteration + refit_ix
            if permutation_mode == "competitive":
                decoy_target = _make_permuted_decoy_target_frame(
                    Y_target=Y_current,
                    target_features=target_features,
                    seed=random_state + 1_000_000 + 10_000 * iteration + refit_ix,
                )
                decoy_features = list(decoy_target.columns)
                X_aug = pd.concat([X_current, Y_current, decoy_target], axis=1)
                column_weights = _make_competitive_augmented_column_weights(
                    x_feature_names=current_features,
                    real_target_feature_names=target_features,
                    decoy_target_feature_names=decoy_features,
                    target_draw_probability=target_draw_probability,
                    real_target_weight=competitive_real_target_weight,
                )
            else:
                decoy_features = None
                X_aug = pd.concat([X_current, Y_current], axis=1)
                column_weights = _make_augmented_column_weights(
                    x_feature_names=current_features,
                    target_feature_names=target_features,
                    target_draw_probability=target_draw_probability,
                )
            model = IsolationForest(
                ntrees=ntrees,
                sample_size=min(sample_size, len(X_aug)),
                ndim=ndim,
                missing_action="fail",
                penalize_range=False,
                random_seed=seed,
                nthreads=nthreads,
                **isotree_kwargs,
            )
            model.fit(X_aug, column_weights=column_weights)
            usage_by_refit.append(
                _target_anchored_attribution_from_isotree_json(
                    model,
                    X_aug=X_aug,
                    x_feature_names=current_features,
                    target_feature_names=target_features,
                    feature_attribution_mode=feature_attribution_mode,
                    leaf_signal_quantile=leaf_signal_quantile,
                    leaf_min_samples=leaf_min_samples,
                    leaf_depth_decay=leaf_depth_decay,
                )
            )
            if decoy_features is not None:
                decoy_usage_by_refit.append(
                    _target_anchored_attribution_from_isotree_json(
                        model,
                        X_aug=X_aug,
                        x_feature_names=current_features,
                        target_feature_names=decoy_features,
                        feature_attribution_mode=feature_attribution_mode,
                        signal_target_feature_names=decoy_features,
                        leaf_signal_quantile=leaf_signal_quantile,
                        leaf_min_samples=leaf_min_samples,
                        leaf_depth_decay=leaf_depth_decay,
                    )
                )

        usage_matrix = np.vstack(usage_by_refit)
        selection_usage = usage_matrix.max(axis=0) if require_used_in_any_refit else usage_matrix.sum(axis=0)
        usage_label = _feature_attribution_usage_label(feature_attribution_mode)
        usage = (
            pd.DataFrame(
                {
                    "feature": current_features,
                    f"{usage_label}_for_selection": selection_usage,
                    f"{usage_label}_mean": usage_matrix.mean(axis=0),
                    f"{usage_label}_min": usage_matrix.min(axis=0),
                    f"{usage_label}_max": usage_matrix.max(axis=0),
                    "n_refits_used_with_target": (usage_matrix > 0).sum(axis=0),
                }
            )
            .sort_values(
                [f"{usage_label}_for_selection", f"{usage_label}_mean"],
                ascending=False,
            )
            .reset_index(drop=True)
        )

        raw_proposed_to_remove = _conservative_removal_candidates(
            usage,
            usage_column=f"{usage_label}_for_selection",
            tie_breaker_columns=[f"{usage_label}_mean"],
            current_features=current_features,
            min_usage_count=min_usage_count,
            n_refits=n_refits,
            min_features_to_keep=min_features_to_keep,
            max_refit_usage_fraction_for_removal=max_refit_usage_fraction_for_removal,
            max_remove_fraction_per_iter=max_remove_fraction_per_iter,
        )
        permutation_result = None
        proposed_to_remove = raw_proposed_to_remove
        if permutation_mode == "competitive":
            decoy_usage_matrix = np.vstack(decoy_usage_by_refit)
            decoy_selection_usage = (
                decoy_usage_matrix.max(axis=0)
                if require_used_in_any_refit
                else decoy_usage_matrix.sum(axis=0)
            )
            usage, permutation_result = _add_competitive_permutation_calibration(
                usage,
                observed_column=f"{usage_label}_for_selection",
                decoy_selection_usage=decoy_selection_usage,
                decoy_usage_matrix=decoy_usage_matrix,
                feature_names=current_features,
                observed_label=usage_label,
                decoy_label=f"permutation_decoy_{usage_label}",
            )
            proposed_to_remove = _permutation_calibrated_removal_candidates(
                usage,
                raw_proposed_to_remove=raw_proposed_to_remove,
                observed_column=f"{usage_label}_for_selection",
                adjusted_column="permutation_adjusted_usage_for_selection",
                tie_breaker_columns=[f"{usage_label}_mean"],
                current_features=current_features,
                min_usage_count=min_usage_count,
                min_permutation_adjusted_usage=min_permutation_adjusted_usage,
                permutation_scope=permutation_scope,
                permutation_usage_margin=permutation_usage_margin,
                min_features_to_keep=min_features_to_keep,
                max_remove_fraction_per_iter=max_remove_fraction_per_iter,
            )
        elif n_target_permutations:
            null_usage_matrix = _target_permutation_usage_matrix(
                X_current=X_current,
                y=y,
                task=task,
                current_features=current_features,
                target_features=target_features,
                target_draw_probability=target_draw_probability,
                regression_target_bins=regression_target_bins,
                n_target_permutations=n_target_permutations,
                n_permutation_refits=n_permutation_refits,
                base_seed=random_state + 1_000_000 + 10_000 * iteration,
                ntrees=ntrees,
                sample_size=sample_size,
                ndim=ndim,
                nthreads=nthreads,
                isotree_kwargs=isotree_kwargs,
                feature_attribution_mode=feature_attribution_mode,
                leaf_signal_quantile=leaf_signal_quantile,
                leaf_min_samples=leaf_min_samples,
                leaf_depth_decay=leaf_depth_decay,
            )
            usage, permutation_result = _add_permutation_calibration(
                usage,
                observed_column=f"{usage_label}_for_selection",
                null_usage_matrix=null_usage_matrix,
                feature_names=current_features,
                null_label=f"permutation_null_{usage_label}",
            )
            proposed_to_remove = _permutation_calibrated_removal_candidates(
                usage,
                raw_proposed_to_remove=raw_proposed_to_remove,
                observed_column=f"{usage_label}_for_selection",
                adjusted_column="permutation_adjusted_usage_for_selection",
                tie_breaker_columns=[f"{usage_label}_mean"],
                current_features=current_features,
                min_usage_count=min_usage_count,
                min_permutation_adjusted_usage=min_permutation_adjusted_usage,
                permutation_scope=permutation_scope,
                permutation_usage_margin=permutation_usage_margin,
                min_features_to_keep=min_features_to_keep,
                max_remove_fraction_per_iter=max_remove_fraction_per_iter,
            )
        to_remove, validation_result = _apply_validation_veto(
            X_raw=X_raw,
            y=y,
            current_features=current_features,
            proposed_to_remove=proposed_to_remove,
            validation_scorer=validation_scorer,
            score_tolerance=score_tolerance,
        )

        history.append(
            {
                "iteration": iteration,
                "n_features_before": len(current_features),
                "n_proposed_removed": len(proposed_to_remove),
                "n_removed": len(to_remove),
                "proposed_removed_features": proposed_to_remove,
                "removed_features": to_remove,
                "validation": validation_result,
                "permutation": permutation_result,
                "usage": usage,
                "target_draw_probability": target_draw_probability,
                "target_features": target_features,
                "permutation_mode": permutation_mode,
                "feature_attribution_mode": feature_attribution_mode,
            }
        )

        if verbose:
            print(
                f"iter={iteration} | features={len(current_features)} | "
                f"refits={n_refits} | propose={len(proposed_to_remove)} | "
                f"remove={len(to_remove)} | target_p={target_draw_probability:g} | "
                f"permutation_mode={permutation_mode}"
            )
        if not to_remove:
            break

        removed_features_all.extend(to_remove)
        remove_set = set(to_remove)
        current_features = [f for f in current_features if f not in remove_set]
        if len(current_features) <= min_features_to_keep:
            break

    result = {
        "selected_features": current_features,
        "removed_features": removed_features_all,
        "history": history,
        "final_usage": history[-1]["usage"] if history else None,
        "original_features": original_features,
        "target_features": target_features,
        "task": task,
        "method": f"target_weighted_json_{feature_attribution_mode}_usage_ndim2",
        "target_draw_probability": target_draw_probability,
        "permutation_mode": permutation_mode,
        "feature_attribution_mode": feature_attribution_mode,
    }
    return X_raw[current_features], result


def select_features_by_class_contrast_isotree_json_usage(
    X,
    y,
    *,
    ntrees=100,
    sample_size=256,
    ndim=2,
    max_iter=20,
    min_usage_count=1,
    n_refits=1,
    require_used_in_any_refit=True,
    target_draw_probability=0.25,
    max_samples_per_side=3000,
    min_class_samples=10,
    combine_classes="max",
    random_state=42,
    nthreads=-1,
    min_features_to_keep=1,
    max_refit_usage_fraction_for_removal=0.0,
    max_remove_fraction_per_iter=0.10,
    validation_scorer=None,
    score_tolerance=0.0,
    standardize_X=True,
    isotree_kwargs=None,
    verbose=True,
):
    """Classification selector using balanced one-vs-rest target-anchored splits."""
    if ndim < 2:
        raise ValueError("This method is intended for ndim >= 2.")
    if not (0.0 < target_draw_probability < 1.0):
        raise ValueError("target_draw_probability must be between 0 and 1.")
    if combine_classes not in {"max", "mean"}:
        raise ValueError("combine_classes must be 'max' or 'mean'.")
    if score_tolerance < 0:
        raise ValueError("score_tolerance must be nonnegative.")

    isotree_kwargs = isotree_kwargs or {}
    _validate_conservative_removal_params(
        n_refits=n_refits,
        min_features_to_keep=min_features_to_keep,
        max_refit_usage_fraction_for_removal=max_refit_usage_fraction_for_removal,
        max_remove_fraction_per_iter=max_remove_fraction_per_iter,
    )

    X_raw = _to_frame(X)
    y = pd.Series(y).reset_index(drop=True)
    if len(X_raw) != len(y):
        raise ValueError("X and y must have the same number of rows.")

    classes = np.array(sorted(y.unique()))
    if len(classes) < 2:
        raise ValueError("Need at least two classes.")

    X_work = _standardize_frame(X_raw) if standardize_X else X_raw.astype(float).copy()
    original_features = list(X_work.columns)
    current_features = list(X_work.columns)
    history = []
    removed_features_all = []

    for iteration in range(max_iter):
        X_current = X_work[current_features].reset_index(drop=True)
        y_current = y.reset_index(drop=True)
        rng = np.random.default_rng(random_state + 100_000 * iteration)
        usage_by_refit = []
        score_by_refit = []

        for class_ix, cls in enumerate(classes):
            pos_idx = np.flatnonzero(y_current.to_numpy() == cls)
            rest_idx = np.flatnonzero(y_current.to_numpy() != cls)
            if len(pos_idx) < min_class_samples or len(rest_idx) < min_class_samples:
                continue

            m = min(len(pos_idx), len(rest_idx), max_samples_per_side)
            for refit_ix in range(n_refits):
                seed = random_state + 100_000 * iteration + 10_000 * class_ix + refit_ix
                pos_sample = rng.choice(pos_idx, size=m, replace=False)
                rest_sample = rng.choice(rest_idx, size=m, replace=False)
                sample_idx = np.concatenate([pos_sample, rest_sample])
                rng.shuffle(sample_idx)

                X_sub = X_current.iloc[sample_idx].reset_index(drop=True)
                target_name = f"__target_ovr__{class_ix}"
                y_sub = (y_current.iloc[sample_idx].to_numpy() == cls).astype(float)
                Y_sub = _standardize_frame(pd.DataFrame({target_name: y_sub}))
                X_aug = pd.concat([X_sub, Y_sub], axis=1)
                column_weights = _make_augmented_column_weights(
                    x_feature_names=current_features,
                    target_feature_names=[target_name],
                    target_draw_probability=target_draw_probability,
                )
                model = IsolationForest(
                    ntrees=ntrees,
                    sample_size=min(sample_size, len(X_aug)),
                    ndim=ndim,
                    missing_action="fail",
                    penalize_range=False,
                    random_seed=seed,
                    nthreads=nthreads,
                    **isotree_kwargs,
                )
                model.fit(X_aug, column_weights=column_weights)

                counts = _anchored_usage_from_isotree_json(
                    model,
                    x_feature_names=current_features,
                    target_feature_names=[target_name],
                ).astype(float)
                usage_by_refit.append(counts)
                denom = counts.sum()
                score_by_refit.append(counts / denom if denom > 0 else np.zeros_like(counts))

        if not usage_by_refit:
            break

        usage_matrix = np.vstack(usage_by_refit)
        score_matrix = np.vstack(score_by_refit)
        selection_usage = usage_matrix.max(axis=0) if require_used_in_any_refit else usage_matrix.sum(axis=0)
        target_contrast_score = score_matrix.max(axis=0) if combine_classes == "max" else score_matrix.mean(axis=0)
        usage = (
            pd.DataFrame(
                {
                    "feature": current_features,
                    "target_contrast_usage_for_selection": selection_usage,
                    "target_contrast_score": target_contrast_score,
                    "target_contrast_usage_mean": usage_matrix.mean(axis=0),
                    "target_contrast_usage_max": usage_matrix.max(axis=0),
                    "n_refits_used_with_target": (usage_matrix > 0).sum(axis=0),
                }
            )
            .sort_values(
                ["target_contrast_usage_for_selection", "target_contrast_score"],
                ascending=False,
            )
            .reset_index(drop=True)
        )

        proposed_to_remove = _conservative_removal_candidates(
            usage,
            usage_column="target_contrast_usage_for_selection",
            tie_breaker_columns=["target_contrast_score"],
            current_features=current_features,
            min_usage_count=min_usage_count,
            n_refits=len(usage_by_refit),
            min_features_to_keep=min_features_to_keep,
            max_refit_usage_fraction_for_removal=max_refit_usage_fraction_for_removal,
            max_remove_fraction_per_iter=max_remove_fraction_per_iter,
        )
        to_remove, validation_result = _apply_validation_veto(
            X_raw=X_raw,
            y=y,
            current_features=current_features,
            proposed_to_remove=proposed_to_remove,
            validation_scorer=validation_scorer,
            score_tolerance=score_tolerance,
        )

        history.append(
            {
                "iteration": iteration,
                "n_features_before": len(current_features),
                "n_proposed_removed": len(proposed_to_remove),
                "n_removed": len(to_remove),
                "proposed_removed_features": proposed_to_remove,
                "removed_features": to_remove,
                "validation": validation_result,
                "usage": usage,
                "classes": classes.tolist(),
                "target_draw_probability": target_draw_probability,
                "combine_classes": combine_classes,
            }
        )

        if verbose:
            print(
                f"iter={iteration} | features={len(current_features)} | "
                f"classes={len(classes)} | refits={n_refits} | "
                f"propose={len(proposed_to_remove)} | remove={len(to_remove)} | "
                f"target_p={target_draw_probability:g}"
            )
        if not to_remove:
            break

        removed_features_all.extend(to_remove)
        remove_set = set(to_remove)
        current_features = [f for f in current_features if f not in remove_set]
        if len(current_features) <= min_features_to_keep:
            break

    result = {
        "selected_features": current_features,
        "removed_features": removed_features_all,
        "history": history,
        "final_usage": history[-1]["usage"] if history else None,
        "original_features": original_features,
        "classes": classes.tolist(),
        "method": "class_contrast_json_usage_ndim2",
        "target_draw_probability": target_draw_probability,
    }
    return X_raw[current_features], result


def _to_frame(X):
    if isinstance(X, pd.DataFrame):
        out = X.copy()
    else:
        X = np.asarray(X)
        out = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
    out.columns = [str(c) for c in out.columns]
    if len(set(out.columns)) != len(out.columns):
        raise ValueError("X must have unique column names.")
    return out


def _standardize_frame(X):
    X = X.astype(float)
    return (X - X.mean(axis=0)) / (X.std(axis=0, ddof=0) + 1e-12)


def _make_target_frame(y, *, task, n_bins=10, existing_columns=None):
    y = pd.Series(y).reset_index(drop=True)
    existing_columns = set(existing_columns or [])
    prefix = "__target__"
    while any(str(c).startswith(prefix) for c in existing_columns):
        prefix = "_" + prefix

    if task == "classification":
        return _standardize_frame(pd.get_dummies(y.astype("category"), prefix=prefix, dtype=float))

    if task == "regression":
        y_float = y.astype(float)
        Y = pd.DataFrame({f"{prefix}z": y_float, f"{prefix}rank": y_float.rank(pct=True)})
        q = min(int(n_bins), y_float.nunique())
        if q >= 2:
            bins = pd.qcut(y_float, q=q, labels=False, duplicates="drop")
            Y = pd.concat([Y, pd.get_dummies(bins, prefix=f"{prefix}bin", dtype=float)], axis=1)
        return _standardize_frame(Y)

    raise ValueError("task must be 'classification' or 'regression'.")


def _make_augmented_column_weights(x_feature_names, target_feature_names, *, target_draw_probability=0.25):
    n_x = len(x_feature_names)
    n_t = len(target_feature_names)
    if n_x < 1:
        raise ValueError("Need at least one X feature.")
    if n_t < 1:
        raise ValueError("Need at least one target feature.")
    target_total_weight = target_draw_probability / (1.0 - target_draw_probability) * float(n_x)
    return np.concatenate([np.ones(n_x, dtype=float), np.full(n_t, target_total_weight / n_t)])


def _make_competitive_augmented_column_weights(
    x_feature_names,
    real_target_feature_names,
    decoy_target_feature_names,
    *,
    target_draw_probability=0.25,
    real_target_weight=0.60,
):
    n_x = len(x_feature_names)
    n_real = len(real_target_feature_names)
    n_decoy = len(decoy_target_feature_names)
    if n_x < 1:
        raise ValueError("Need at least one X feature.")
    if n_real < 1 or n_decoy < 1:
        raise ValueError("Need at least one real target and one decoy target feature.")

    target_total_weight = target_draw_probability / (1.0 - target_draw_probability) * float(n_x)
    real_total_weight = real_target_weight * target_total_weight
    decoy_total_weight = (1.0 - real_target_weight) * target_total_weight
    return np.concatenate(
        [
            np.ones(n_x, dtype=float),
            np.full(n_real, real_total_weight / n_real),
            np.full(n_decoy, decoy_total_weight / n_decoy),
        ]
    )


def _make_permuted_decoy_target_frame(
    *,
    Y_target,
    target_features,
    seed,
):
    rng = np.random.default_rng(seed)
    row_order = rng.permutation(len(Y_target))
    decoy = Y_target.iloc[row_order].reset_index(drop=True).reindex(columns=target_features)
    decoy.columns = [_decoy_target_feature_name(feature) for feature in target_features]
    return decoy


def _decoy_target_feature_name(feature):
    return f"__permuted_{feature}"


def _feature_attribution_usage_label(feature_attribution_mode):
    if feature_attribution_mode == "routing_flip":
        return "target_anchored_flip_count"
    if feature_attribution_mode == "leaf_backtrack":
        return "target_leaf_backtrack_score"
    raise ValueError("feature_attribution_mode must be 'routing_flip' or 'leaf_backtrack'.")


def _target_anchored_attribution_from_isotree_json(
    model,
    *,
    X_aug,
    x_feature_names,
    target_feature_names,
    feature_attribution_mode,
    signal_target_feature_names=None,
    leaf_signal_quantile=0.25,
    leaf_min_samples=2,
    leaf_depth_decay=0.0,
):
    if feature_attribution_mode == "routing_flip":
        return _anchored_flip_usage_from_isotree_json(
            model,
            X_aug=X_aug,
            x_feature_names=x_feature_names,
            target_feature_names=target_feature_names,
        )
    if feature_attribution_mode == "leaf_backtrack":
        return _leaf_backtrack_usage_from_isotree_json(
            model,
            X_aug=X_aug,
            x_feature_names=x_feature_names,
            target_feature_names=target_feature_names,
            signal_target_feature_names=signal_target_feature_names or target_feature_names,
            leaf_signal_quantile=leaf_signal_quantile,
            leaf_min_samples=leaf_min_samples,
            leaf_depth_decay=leaf_depth_decay,
        )
    raise ValueError("feature_attribution_mode must be 'routing_flip' or 'leaf_backtrack'.")


def _anchored_flip_usage_from_isotree_json(model, *, X_aug, x_feature_names, target_feature_names):
    trees = model.to_json(as_str=False)
    if isinstance(trees, dict):
        trees = [trees]

    x_feature_names = list(x_feature_names)
    counts = pd.Series(0, index=x_feature_names, dtype="int64")
    X_aug = X_aug.reset_index(drop=True)
    x_features = set(map(str, x_feature_names))
    target_features = set(map(str, target_feature_names))
    columns = set(map(str, X_aug.columns))
    stats = {"n_extractable_splits": 0}

    for tree in trees:
        _accumulate_tree_flip_usage(
            tree,
            X_aug=X_aug,
            active_rows=np.arange(len(X_aug)),
            counts=counts,
            x_features=x_features,
            target_features=target_features,
            columns=columns,
            stats=stats,
        )

    if stats["n_extractable_splits"] == 0:
        raise ValueError("Could not parse split coefficients and thresholds from IsoTree JSON.")

    return counts.loc[x_feature_names].to_numpy(dtype=np.int64)


def _leaf_backtrack_usage_from_isotree_json(
    model,
    *,
    X_aug,
    x_feature_names,
    target_feature_names,
    signal_target_feature_names,
    leaf_signal_quantile=0.25,
    leaf_min_samples=2,
    leaf_depth_decay=0.0,
):
    trees = model.to_json(as_str=False)
    if isinstance(trees, dict):
        trees = [trees]

    x_feature_names = list(x_feature_names)
    X_aug = X_aug.reset_index(drop=True)
    x_features = set(map(str, x_feature_names))
    target_features = set(map(str, target_feature_names))
    signal_target_feature_names = list(signal_target_feature_names)
    columns = set(map(str, X_aug.columns))
    stats = {"n_extractable_splits": 0}
    leaf_records = []

    for tree in trees:
        _collect_tree_leaf_paths(
            tree,
            X_aug=X_aug,
            active_rows=np.arange(len(X_aug)),
            path=[],
            leaf_records=leaf_records,
            x_features=x_features,
            target_features=target_features,
            signal_target_feature_names=signal_target_feature_names,
            columns=columns,
            stats=stats,
            leaf_min_samples=leaf_min_samples,
        )

    if stats["n_extractable_splits"] == 0:
        raise ValueError("Could not parse split coefficients and thresholds from IsoTree JSON.")

    eligible = [leaf for leaf in leaf_records if leaf["n_samples"] >= leaf_min_samples]
    counts = pd.Series(0.0, index=x_feature_names, dtype="float64")
    if not eligible:
        return counts.loc[x_feature_names].to_numpy(dtype=float)

    variances = np.array([leaf["target_variance"] for leaf in eligible], dtype=float)
    signal_threshold = float(np.quantile(variances, leaf_signal_quantile))
    for leaf in eligible:
        if leaf["target_variance"] > signal_threshold:
            continue
        leaf_weight = float(leaf["n_samples"])
        for depth, terms in leaf["path"]:
            depth_weight = 1.0 / (1.0 + leaf_depth_decay * float(depth))
            for feature, coef in terms.items():
                if feature in x_features:
                    counts.loc[feature] += abs(float(coef)) * leaf_weight * depth_weight

    return counts.loc[x_feature_names].to_numpy(dtype=float)


def _collect_tree_leaf_paths(
    tree,
    *,
    X_aug,
    active_rows,
    path,
    leaf_records,
    x_features,
    target_features,
    signal_target_feature_names,
    columns,
    stats,
    leaf_min_samples,
):
    if not isinstance(tree, dict):
        return

    if _looks_like_flat_tree(tree):
        raise ValueError("Leaf backtracking requires nested child nodes in IsoTree JSON.")

    _collect_node_leaf_paths(
        tree,
        X_aug=X_aug,
        active_rows=active_rows,
        path=path,
        leaf_records=leaf_records,
        x_features=x_features,
        target_features=target_features,
        signal_target_feature_names=signal_target_feature_names,
        columns=columns,
        stats=stats,
        leaf_min_samples=leaf_min_samples,
        depth=0,
    )


def _collect_node_leaf_paths(
    node,
    *,
    X_aug,
    active_rows,
    path,
    leaf_records,
    x_features,
    target_features,
    signal_target_feature_names,
    columns,
    stats,
    leaf_min_samples,
    depth,
):
    if not isinstance(node, dict) or len(active_rows) == 0:
        return

    split = _extract_oblique_split(node, columns=columns)
    children = _left_right_children(node)
    if split is None or children is None:
        if len(active_rows) >= leaf_min_samples:
            y_leaf = X_aug.iloc[active_rows][signal_target_feature_names].to_numpy(dtype=float)
            leaf_records.append(
                {
                    "n_samples": int(len(active_rows)),
                    "target_variance": float(np.mean(np.var(y_leaf, axis=0))),
                    "path": list(path),
                }
            )
        return

    terms, threshold = split
    stats["n_extractable_splits"] += 1
    path_terms = {feature: coef for feature, coef in terms.items() if feature in x_features}
    path_next = path + [(depth, path_terms)] if path_terms else path

    left_child, right_child = children
    X_node = X_aug.iloc[active_rows]
    routed_left = _evaluate_split_terms(X_node, terms) <= threshold
    left_rows = active_rows[routed_left]
    right_rows = active_rows[~routed_left]
    _collect_node_leaf_paths(
        left_child,
        X_aug=X_aug,
        active_rows=left_rows,
        path=path_next,
        leaf_records=leaf_records,
        x_features=x_features,
        target_features=target_features,
        signal_target_feature_names=signal_target_feature_names,
        columns=columns,
        stats=stats,
        leaf_min_samples=leaf_min_samples,
        depth=depth + 1,
    )
    _collect_node_leaf_paths(
        right_child,
        X_aug=X_aug,
        active_rows=right_rows,
        path=path_next,
        leaf_records=leaf_records,
        x_features=x_features,
        target_features=target_features,
        signal_target_feature_names=signal_target_feature_names,
        columns=columns,
        stats=stats,
        leaf_min_samples=leaf_min_samples,
        depth=depth + 1,
    )


def _accumulate_tree_flip_usage(tree, *, X_aug, active_rows, counts, x_features, target_features, columns, stats):
    if not isinstance(tree, dict):
        return

    if _looks_like_flat_tree(tree):
        for node in tree.values():
            _accumulate_node_flip_usage(
                node,
                X_aug=X_aug,
                active_rows=active_rows,
                counts=counts,
                x_features=x_features,
                target_features=target_features,
                columns=columns,
                stats=stats,
                route_children=False,
            )
        return

    _accumulate_node_flip_usage(
        tree,
        X_aug=X_aug,
        active_rows=active_rows,
        counts=counts,
        x_features=x_features,
        target_features=target_features,
        columns=columns,
        stats=stats,
        route_children=True,
    )


def _looks_like_flat_tree(tree):
    if not tree:
        return False
    values = list(tree.values())
    return values and all(isinstance(value, dict) for value in values)


def _accumulate_node_flip_usage(
    node,
    *,
    X_aug,
    active_rows,
    counts,
    x_features,
    target_features,
    columns,
    stats,
    route_children,
):
    if not isinstance(node, dict) or len(active_rows) == 0:
        return

    split = _extract_oblique_split(node, columns=columns)
    if split is None:
        if route_children:
            for child in _child_nodes(node):
                _accumulate_node_flip_usage(
                    child,
                    X_aug=X_aug,
                    active_rows=active_rows,
                    counts=counts,
                    x_features=x_features,
                    target_features=target_features,
                    columns=columns,
                    stats=stats,
                    route_children=True,
                )
        return

    terms, threshold = split
    stats["n_extractable_splits"] += 1
    split_x_features = [feature for feature in terms if feature in x_features]
    split_target_features = [feature for feature in terms if feature in target_features]
    if split_x_features and split_target_features:
        X_node = X_aug.iloc[active_rows]
        values = _evaluate_split_terms(X_node, terms)
        routed_left = values <= threshold
        for feature in split_x_features:
            counterfactual_values = values - terms[feature] * X_node[feature].to_numpy(dtype=float)
            counterfactual_left = counterfactual_values <= threshold
            counts.loc[feature] += int(np.count_nonzero(routed_left != counterfactual_left))

    if not route_children:
        return

    children = _left_right_children(node)
    if children is None:
        for child in _child_nodes(node):
            _accumulate_node_flip_usage(
                child,
                X_aug=X_aug,
                active_rows=active_rows,
                counts=counts,
                x_features=x_features,
                target_features=target_features,
                columns=columns,
                stats=stats,
                route_children=True,
            )
        return

    left_child, right_child = children
    X_node = X_aug.iloc[active_rows]
    routed_left = _evaluate_split_terms(X_node, terms) <= threshold
    left_rows = active_rows[routed_left]
    right_rows = active_rows[~routed_left]
    _accumulate_node_flip_usage(
        left_child,
        X_aug=X_aug,
        active_rows=left_rows,
        counts=counts,
        x_features=x_features,
        target_features=target_features,
        columns=columns,
        stats=stats,
        route_children=True,
    )
    _accumulate_node_flip_usage(
        right_child,
        X_aug=X_aug,
        active_rows=right_rows,
        counts=counts,
        x_features=x_features,
        target_features=target_features,
        columns=columns,
        stats=stats,
        route_children=True,
    )


def _extract_oblique_split(node, *, columns):
    terms = _extract_split_terms(node, columns=columns)
    threshold = _extract_split_threshold(node)
    if not terms or threshold is None:
        return None
    return terms, threshold


def _extract_split_terms(node, *, columns):
    feature_keys = ("features", "feature", "cols", "columns", "col", "column")
    coef_keys = ("coefs", "coef", "coefficients", "weights", "weight")

    for feature_key in feature_keys:
        for coef_key in coef_keys:
            if feature_key not in node or coef_key not in node:
                continue
            feature_value = node[feature_key]
            coef_value = node[coef_key]
            terms = _terms_from_feature_coef_values(feature_value, coef_value, columns=columns)
            if terms:
                return terms

    for coef_key in coef_keys:
        coef_value = node.get(coef_key)
        if isinstance(coef_value, dict):
            terms = {
                str(feature): float(coef)
                for feature, coef in coef_value.items()
                if str(feature) in columns and _is_number(coef)
            }
            if terms:
                return terms

    for key in ("coef_by_col", "coef_by_column", "weights_by_col", "weights_by_column"):
        value = node.get(key)
        if isinstance(value, dict):
            terms = {
                str(feature): float(coef)
                for feature, coef in value.items()
                if str(feature) in columns and _is_number(coef)
            }
            if terms:
                return terms

    return None


def _terms_from_feature_coef_values(feature_value, coef_value, *, columns):
    if isinstance(feature_value, str) and _is_number(coef_value) and feature_value in columns:
        return {feature_value: float(coef_value)}

    if isinstance(feature_value, dict) and isinstance(coef_value, dict):
        terms = {}
        for key, feature in feature_value.items():
            if key in coef_value and str(feature) in columns and _is_number(coef_value[key]):
                terms[str(feature)] = float(coef_value[key])
        return terms or None

    if not isinstance(feature_value, (list, tuple)) or not isinstance(coef_value, (list, tuple)):
        return None
    if len(feature_value) != len(coef_value):
        return None

    terms = {}
    for feature, coef in zip(feature_value, coef_value):
        feature = str(feature)
        if feature in columns and _is_number(coef):
            terms[feature] = float(coef)
    return terms or None


def _extract_split_threshold(node):
    for key in ("threshold", "split_point", "cutpoint", "cutoff", "t"):
        value = node.get(key)
        if _is_number(value):
            return float(value)
    return None


def _evaluate_split_terms(X, terms):
    values = np.zeros(len(X), dtype=float)
    for feature, coef in terms.items():
        values += coef * X[feature].to_numpy(dtype=float)
    return values


def _left_right_children(node):
    child_key_pairs = (
        ("left", "right"),
        ("left_child", "right_child"),
        ("lte", "gt"),
        ("less", "greater"),
        ("yes", "no"),
        ("true", "false"),
    )
    for left_key, right_key in child_key_pairs:
        left = node.get(left_key)
        right = node.get(right_key)
        if isinstance(left, dict) and isinstance(right, dict):
            return left, right
    return None


def _child_nodes(node):
    children = []
    for value in node.values():
        if isinstance(value, dict):
            children.append(value)
        elif isinstance(value, list):
            children.extend(item for item in value if isinstance(item, dict))
    return children


def _is_number(value):
    return isinstance(value, (int, float, np.integer, np.floating)) and np.isfinite(value)


def _anchored_usage_from_isotree_json(model, *, x_feature_names, target_feature_names):
    trees = model.to_json(as_str=False)
    if isinstance(trees, dict):
        trees = [trees]

    x_features = set(map(str, x_feature_names))
    target_features = set(map(str, target_feature_names))
    counts = pd.Series(0, index=list(x_feature_names), dtype="int64")

    for tree in trees:
        for node in _iter_isotree_node_dicts(tree):
            node_strings = set(_collect_strings(node))
            used_x = node_strings & x_features
            if used_x and (node_strings & target_features):
                for feature in used_x:
                    counts.loc[feature] += 1

    return counts.loc[list(x_feature_names)].to_numpy(dtype=np.int64)


def _iter_isotree_node_dicts(tree):
    if isinstance(tree, dict):
        values = list(tree.values())
        if values and all(isinstance(value, dict) for value in values):
            for node in values:
                yield node
        else:
            yield tree


def _collect_strings(obj):
    if isinstance(obj, str):
        return [obj]
    if isinstance(obj, dict):
        found = []
        for value in obj.values():
            found.extend(_collect_strings(value))
        return found
    if isinstance(obj, list):
        found = []
        for item in obj:
            found.extend(_collect_strings(item))
        return found
    return []


def _validate_conservative_removal_params(
    *,
    n_refits,
    min_features_to_keep,
    max_refit_usage_fraction_for_removal,
    max_remove_fraction_per_iter,
):
    if n_refits < 1:
        raise ValueError("n_refits must be at least 1.")
    if min_features_to_keep < 1:
        raise ValueError("min_features_to_keep must be at least 1.")
    if not (0.0 <= max_refit_usage_fraction_for_removal <= 1.0):
        raise ValueError("max_refit_usage_fraction_for_removal must be between 0 and 1.")
    if not (0.0 < max_remove_fraction_per_iter <= 1.0):
        raise ValueError("max_remove_fraction_per_iter must be in (0, 1].")


def _validate_permutation_params(
    *,
    n_target_permutations,
    n_permutation_refits,
    permutation_mode,
    permutation_scope,
    permutation_usage_margin,
    competitive_real_target_weight,
):
    if n_target_permutations < 0:
        raise ValueError("n_target_permutations must be nonnegative.")
    if n_permutation_refits < 1:
        raise ValueError("n_permutation_refits must be at least 1.")
    if permutation_mode not in {"separate", "competitive"}:
        raise ValueError("permutation_mode must be 'separate' or 'competitive'.")
    if permutation_scope not in {"all", "candidates", "borderline"}:
        raise ValueError("permutation_scope must be 'all', 'candidates', or 'borderline'.")
    if permutation_usage_margin < 0:
        raise ValueError("permutation_usage_margin must be nonnegative.")
    if not (0.5 <= competitive_real_target_weight < 1.0):
        raise ValueError("competitive_real_target_weight must be in [0.5, 1.0).")


def _target_permutation_usage_matrix(
    *,
    X_current,
    y,
    task,
    current_features,
    target_features,
    target_draw_probability,
    regression_target_bins,
    n_target_permutations,
    n_permutation_refits,
    base_seed,
    ntrees,
    sample_size,
    ndim,
    nthreads,
    isotree_kwargs,
    feature_attribution_mode="routing_flip",
    leaf_signal_quantile=0.25,
    leaf_min_samples=2,
    leaf_depth_decay=0.0,
):
    rows = []
    y_values = pd.Series(y).reset_index(drop=True).to_numpy()
    column_weights = _make_augmented_column_weights(
        x_feature_names=current_features,
        target_feature_names=target_features,
        target_draw_probability=target_draw_probability,
    )

    for permutation_ix in range(n_target_permutations):
        permutation_seed = base_seed + 10_000 * permutation_ix
        rng = np.random.default_rng(permutation_seed)
        y_permuted = pd.Series(rng.permutation(y_values))
        Y_permuted = _make_target_frame(
            y_permuted,
            task=task,
            n_bins=regression_target_bins,
            existing_columns=current_features,
        )
        Y_permuted = Y_permuted.reindex(columns=target_features, fill_value=0.0).reset_index(drop=True)
        X_aug = pd.concat([X_current.reset_index(drop=True), Y_permuted], axis=1)

        for refit_ix in range(n_permutation_refits):
            seed = permutation_seed + refit_ix
            model = IsolationForest(
                ntrees=ntrees,
                sample_size=min(sample_size, len(X_aug)),
                ndim=ndim,
                missing_action="fail",
                penalize_range=False,
                random_seed=seed,
                nthreads=nthreads,
                **isotree_kwargs,
            )
            model.fit(X_aug, column_weights=column_weights)
            rows.append(
                _target_anchored_attribution_from_isotree_json(
                    model,
                    X_aug=X_aug,
                    x_feature_names=current_features,
                    target_feature_names=target_features,
                    feature_attribution_mode=feature_attribution_mode,
                    leaf_signal_quantile=leaf_signal_quantile,
                    leaf_min_samples=leaf_min_samples,
                    leaf_depth_decay=leaf_depth_decay,
                )
            )

    return np.vstack(rows)


def _add_permutation_calibration(
    usage,
    *,
    observed_column,
    null_usage_matrix,
    feature_names,
    null_label="permutation_null_usage",
):
    usage = usage.copy()
    null_mean = null_usage_matrix.mean(axis=0)
    null_std = null_usage_matrix.std(axis=0, ddof=0)
    null_frame = pd.DataFrame(
        {
            "feature": list(feature_names),
            f"{null_label}_mean": null_mean,
            f"{null_label}_std": null_std,
            f"{null_label}_max": null_usage_matrix.max(axis=0),
        }
    )
    empirical_p_values = {}
    for feature_ix, feature in enumerate(feature_names):
        observed = float(usage.loc[usage["feature"] == feature, observed_column].iloc[0])
        empirical_p_values[feature] = (1.0 + float((null_usage_matrix[:, feature_ix] >= observed).sum())) / (
            1.0 + null_usage_matrix.shape[0]
        )

    usage = usage.merge(null_frame, on="feature", how="left")
    observed = usage[observed_column].to_numpy(dtype=float)
    usage["permutation_adjusted_usage_for_selection"] = observed - usage[f"{null_label}_mean"]
    usage["permutation_usage_z_score"] = (
        observed - usage[f"{null_label}_mean"]
    ) / (usage[f"{null_label}_std"] + 1e-12)
    usage["permutation_empirical_p_value"] = usage["feature"].map(empirical_p_values)
    usage = usage.sort_values(
        ["permutation_adjusted_usage_for_selection", observed_column],
        ascending=False,
    ).reset_index(drop=True)
    return usage, {
        "mode": "separate",
        "n_null_fits": int(null_usage_matrix.shape[0]),
        "null_usage_mean_mean": float(null_mean.mean()),
        "null_usage_mean_max": float(null_mean.max()),
        "null_label": null_label,
    }


def _add_competitive_permutation_calibration(
    usage,
    *,
    observed_column,
    decoy_selection_usage,
    decoy_usage_matrix,
    feature_names,
    observed_label="target_anchored_usage",
    decoy_label="permutation_decoy_usage",
):
    usage = usage.copy()
    decoy_frame = pd.DataFrame(
        {
            "feature": list(feature_names),
            f"{decoy_label}_for_selection": decoy_selection_usage,
            f"{decoy_label}_mean": decoy_usage_matrix.mean(axis=0),
            f"{decoy_label}_min": decoy_usage_matrix.min(axis=0),
            f"{decoy_label}_max": decoy_usage_matrix.max(axis=0),
        }
    )
    usage = usage.merge(decoy_frame, on="feature", how="left")
    observed = usage[observed_column].to_numpy(dtype=float)
    decoy = usage[f"{decoy_label}_for_selection"].to_numpy(dtype=float)
    usage["permutation_adjusted_usage_for_selection"] = observed - decoy
    usage["permutation_competitive_ratio"] = (observed - decoy) / (observed + decoy + 1e-12)
    usage = usage.sort_values(
        ["permutation_adjusted_usage_for_selection", observed_column],
        ascending=False,
    ).reset_index(drop=True)
    return usage, {
        "mode": "competitive",
        "n_competitive_refits": int(decoy_usage_matrix.shape[0]),
        "decoy_usage_mean_mean": float(decoy_usage_matrix.mean(axis=0).mean()),
        "decoy_usage_mean_max": float(decoy_usage_matrix.mean(axis=0).max()),
        "observed_label": observed_label,
        "decoy_label": decoy_label,
    }


def _permutation_calibrated_removal_candidates(
    usage,
    *,
    raw_proposed_to_remove,
    observed_column,
    adjusted_column,
    tie_breaker_columns,
    current_features,
    min_usage_count,
    min_permutation_adjusted_usage,
    permutation_scope,
    permutation_usage_margin,
    min_features_to_keep,
    max_remove_fraction_per_iter,
):
    raw_set = set(raw_proposed_to_remove)
    if permutation_scope == "all":
        scoped = set(usage["feature"])
    elif permutation_scope == "candidates":
        scoped = raw_set
    else:
        borderline_limit = min_usage_count + permutation_usage_margin
        scoped = set(usage.loc[usage[observed_column] <= borderline_limit, "feature"]) | raw_set

    calibrated = set(
        usage.loc[
            usage["feature"].isin(scoped)
            & (usage[adjusted_column] <= min_permutation_adjusted_usage),
            "feature",
        ]
    )
    candidate_features = raw_set | calibrated
    if not candidate_features:
        return []

    sort_columns = [adjusted_column, observed_column] + list(tie_breaker_columns)
    low_usage = usage.sort_values(sort_columns, ascending=True)
    ordered = low_usage.loc[low_usage["feature"].isin(candidate_features), "feature"].tolist()
    return _limit_removal_candidates(
        ordered,
        current_features=current_features,
        min_features_to_keep=min_features_to_keep,
        max_remove_fraction_per_iter=max_remove_fraction_per_iter,
    )


def _limit_removal_candidates(
    candidate_features,
    *,
    current_features,
    min_features_to_keep,
    max_remove_fraction_per_iter,
):
    max_removable_for_floor = max(0, len(current_features) - min_features_to_keep)
    max_removable_for_budget = max(1, int(np.floor(max_remove_fraction_per_iter * len(current_features))))
    max_removable = min(max_removable_for_floor, max_removable_for_budget)
    if max_removable <= 0:
        return []
    return list(candidate_features)[:max_removable]


def _conservative_removal_candidates(
    usage,
    *,
    usage_column,
    tie_breaker_columns,
    current_features,
    min_usage_count,
    n_refits,
    min_features_to_keep,
    max_refit_usage_fraction_for_removal,
    max_remove_fraction_per_iter,
):
    usage = usage.copy()
    usage["refit_usage_fraction"] = usage["n_refits_used_with_target"] / float(n_refits)
    candidate_mask = (
        (usage[usage_column] < min_usage_count)
        & (usage["refit_usage_fraction"] <= max_refit_usage_fraction_for_removal)
    )
    candidate_features = usage.loc[candidate_mask, "feature"].tolist()
    if not candidate_features:
        return []

    sort_columns = [usage_column, "refit_usage_fraction"] + list(tie_breaker_columns)
    low_usage = usage.sort_values(sort_columns, ascending=True)
    ordered = low_usage.loc[low_usage["feature"].isin(candidate_features), "feature"].tolist()
    return _limit_removal_candidates(
        ordered,
        current_features=current_features,
        min_features_to_keep=min_features_to_keep,
        max_remove_fraction_per_iter=max_remove_fraction_per_iter,
    )


def _apply_validation_veto(
    *,
    X_raw,
    y,
    current_features,
    proposed_to_remove,
    validation_scorer,
    score_tolerance,
):
    if not proposed_to_remove or validation_scorer is None:
        return list(proposed_to_remove), None

    remove_set = set(proposed_to_remove)
    candidate_features = [feature for feature in current_features if feature not in remove_set]
    current_score = validation_scorer(X_raw[current_features], y)
    candidate_score = validation_scorer(X_raw[candidate_features], y)
    accepted = candidate_score >= current_score - score_tolerance
    validation_result = {
        "current_score": current_score,
        "candidate_score": candidate_score,
        "score_tolerance": score_tolerance,
        "vetoed": not accepted,
    }
    return (list(proposed_to_remove), validation_result) if accepted else ([], validation_result)
