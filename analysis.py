# analysis.py
import numpy as np
import pandas as pd
from skimage.measure import label, regionprops_table
from skimage.morphology import remove_small_objects


def seg_metrics(gt01: np.ndarray, pred01: np.ndarray) -> dict:
    """
    gt01/pred01: binary arrays where 1 = crack/pore, 0 = matrix
    """
    gt = (gt01 > 0).astype(np.uint8)
    pr = (pred01 > 0).astype(np.uint8)

    tp = int(np.sum((gt == 1) & (pr == 1)))
    fp = int(np.sum((gt == 0) & (pr == 1)))
    fn = int(np.sum((gt == 1) & (pr == 0)))
    tn = int(np.sum((gt == 0) & (pr == 0)))

    iou = tp / (tp + fp + fn + 1e-9)
    dice = 2 * tp / (2 * tp + fp + fn + 1e-9)
    prec = tp / (tp + fp + 1e-9)
    rec = tp / (tp + fn + 1e-9)
    f1 = 2 * prec * rec / (prec + rec + 1e-9)
    acc = (tp + tn) / (tp + tn + fp + fn + 1e-9)

    return {
        "IoU": float(iou),
        "Dice": float(dice),
        "Precision": float(prec),
        "Recall": float(rec),
        "F1": float(f1),
        "Accuracy": float(acc),
        "TP": tp, "FP": fp, "FN": fn, "TN": tn
    }


def _eq_diameter_px(area_px: np.ndarray) -> np.ndarray:
    return np.sqrt(4.0 * area_px / np.pi)


def morphology_report(mask01: np.ndarray, res_um_px: float, min_obj_px: int = 0):
    """
    mask01: binary where 1 = crack/pore, 0 = matrix
    res_um_px: microns per pixel

    Returns:
      summary_dict, df_objects
    """
    res = float(res_um_px)
    m = (mask01 > 0).astype(np.uint8)

    if min_obj_px and int(min_obj_px) > 0:
        m = remove_small_objects(m.astype(bool), min_size=int(min_obj_px)).astype(np.uint8)

    porosity_2d = float(np.mean(m))
    lab = label(m, connectivity=2)
    n_obj = int(lab.max())

    if n_obj == 0:
        summary = {
            "porosity_2d": porosity_2d,
            "n_objects": 0,
            "largest_component_frac": 0.0,
            "connectivity_proxy": 0.0,

            # unit-aware (µm / µm²) outputs
            "mean_eq_diam_um": 0.0,
            "median_eq_diam_um": 0.0,
            "max_eq_diam_um": 0.0,
            "mean_area_um2": 0.0,
            "mean_perimeter_um": 0.0,
            "mean_aperture_um": 0.0,
        }
        return summary, pd.DataFrame()

    props = regionprops_table(
        lab,
        properties=(
            "label",
            "area",               # px^2
            "perimeter",          # px
            "eccentricity",
            "major_axis_length",  # px
            "minor_axis_length",  # px
            "solidity",
        ),
    )
    df = pd.DataFrame(props)

    # ---------- Unit conversions (journal-friendly column names) ----------
    df["area_um2"] = df["area"].astype(float) * (res ** 2)
    df["perimeter_um"] = df["perimeter"].astype(float) * res
    df["major_axis_length_um"] = df["major_axis_length"].astype(float) * res
    df["minor_axis_length_um"] = df["minor_axis_length"].astype(float) * res

    # Equivalent diameter (from area)
    df["eq_diam_px"] = _eq_diameter_px(df["area"].values.astype(float))
    df["eq_diam_um"] = df["eq_diam_px"] * res

    # Aperture proxy: aperture ≈ area / major_axis_length (px) → µm
    maj = np.maximum(df["major_axis_length"].values.astype(float), 1e-9)
    df["aperture_px_proxy"] = df["area"].values.astype(float) / maj
    df["aperture_um_proxy"] = df["aperture_px_proxy"] * res

    # Largest component dominance (connectivity proxy)
    total_area_px2 = float(df["area"].sum())
    largest_area_px2 = float(df["area"].max())
    largest_frac = float(largest_area_px2 / (total_area_px2 + 1e-9))
    connectivity_proxy = float(largest_frac)

    summary = {
        "porosity_2d": porosity_2d,
        "n_objects": n_obj,
        "largest_component_frac": largest_frac,
        "connectivity_proxy": connectivity_proxy,

        # unit-aware summaries
        "mean_eq_diam_um": float(df["eq_diam_um"].mean()),
        "median_eq_diam_um": float(df["eq_diam_um"].median()),
        "max_eq_diam_um": float(df["eq_diam_um"].max()),
        "mean_area_um2": float(df["area_um2"].mean()),
        "mean_perimeter_um": float(df["perimeter_um"].mean()),
        "mean_aperture_um": float(df["aperture_um_proxy"].mean()),
    }
    return summary, df


def mask_to_png_black_crack(mask01: np.ndarray) -> np.ndarray:
    """
    Convert binary mask (1=crack/pore) to uint8 image where:
      crack/pore = black (0)
      matrix     = white (255)
    Returns uint8 HxW
    """
    m = (mask01 > 0).astype(np.uint8)
    return np.where(m > 0, 0, 255).astype(np.uint8)


def prob_to_png(prob: np.ndarray) -> np.ndarray:
    """
    prob: float [0,1] HxW
    Returns uint8 HxW (0..255)
    """
    p = np.clip(prob, 0.0, 1.0)
    return (p * 255).astype(np.uint8)

