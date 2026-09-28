"""
Predict concrete strength from fracture-surface images.
Python port of the MATLAB script, with the major methodological bugs fixed.

Run from the project root:   python src/fracture_strength.py
Requires: numpy pandas scikit-image scikit-learn scipy matplotlib pillow

Expected data/metadata.csv (edit the CONFIG block if yours differs):
    filename,strength_MPa[,specimen_id][,batch_id]
  - one row per image (48 rows), the two halves of a specimen share a strength
  - specimen_id (optional): if missing, consecutive files (sorted) are paired
  - batch_id    (optional): if missing, every BATCH_SIZE consecutive specimens
                            form a batch (the MATLAB strengths look like 8 x 3)
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from scipy import stats
from skimage import color, exposure, feature, filters, morphology, transform
from scipy.ndimage import gaussian_filter
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import LeaveOneGroupOut, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# ----------------------------- CONFIG ---------------------------------------
ROOT = Path(__file__).resolve().parent.parent if "__file__" in globals() else Path(".")
IMG_DIR = ROOT / "data" / "raw"
META_CSV = ROOT / "data" / "metadata.csv"
RESULTS_DIR = ROOT / "results"

COL_FILE, COL_STRENGTH = "filename", "strength_MPa"
COL_SPECIMEN, COL_BATCH = "specimen_id", "batch_id"   # optional columns
BATCH_SIZE = 3            # specimens per batch if batch_id is absent
SIGMAS = [1, 3, 6, 12]    # Gaussian blur scales for roughness
FEATURES = ["coarseR", "crackDensity"]   # any of the names computed below
COMMON_SIZE = None        # e.g. (1024, 1024); None = auto (smallest image size)
# -----------------------------------------------------------------------------


def natural_key(s):
    """Sort 'img2' before 'img10' (MATLAB's dir() does NOT do this)."""
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", str(s))]


def load_gray(path, size):
    img = np.asarray(Image.open(path))
    gray = color.rgb2gray(img[..., :3]) if img.ndim == 3 else img.astype(float)
    gray = gray.astype(float)
    if gray.max() > 1.0:
        gray = gray / 255.0 if gray.max() <= 255 else gray / gray.max()
    # FIX: put all images on the same pixel grid, otherwise every
    # scale-dependent feature (roughness, crack length) is biased by resolution.
    if gray.shape != size:
        gray = transform.resize(gray, size, anti_aliasing=True)
    gray = (gray - gray.min()) / (gray.max() - gray.min() + 1e-12)  # min-max stretch
    return exposure.equalize_adapthist(gray)                        # CLAHE


def image_features(gray):
    f = {}
    f["meanIntensity"] = gray.mean()
    f["stdIntensity"] = gray.std(ddof=1)

    # multi-scale roughness: mean Sobel gradient magnitude after Gaussian blur
    rough = [filters.sobel(gaussian_filter(gray, s)).mean() for s in SIGMAS]
    half = len(SIGMAS) // 2
    f["fineR"] = np.mean(rough[:half])
    f["coarseR"] = np.mean(rough[half:])

    # crack density: Canny edges -> 1px skeleton -> fraction of pixels
    edges = feature.canny(gray, sigma=np.sqrt(2))
    f["crackDensity"] = morphology.skeletonize(edges).sum() / edges.size

    # NOTE: this is the fraction of BRIGHT pixels (solid), not pores, unless the
    # pores are bright. Kept for parity with the MATLAB code.
    f["brightFraction"] = (gray > filters.threshold_otsu(gray)).mean()

    # GLCM contrast, horizontal neighbour (computed on 64 gray levels for speed)
    q = np.clip((gray * 63).astype(np.uint8), 0, 63)
    glcm = feature.graycomatrix(q, [1], [0], levels=64, symmetric=False, normed=True)
    f["glcmContrast"] = feature.graycoprops(glcm, "contrast")[0, 0]
    return f


def load_metadata():
    meta = pd.read_csv(META_CSV)
    meta = meta.iloc[sorted(range(len(meta)), key=lambda i: natural_key(meta[COL_FILE][i]))]
    meta = meta.reset_index(drop=True)
    if COL_SPECIMEN not in meta:
        meta[COL_SPECIMEN] = np.arange(len(meta)) // 2
    if COL_BATCH not in meta:
        meta[COL_BATCH] = meta[COL_SPECIMEN] // BATCH_SIZE
    # sanity checks: halves must agree on strength
    spread = meta.groupby(COL_SPECIMEN)[COL_STRENGTH].nunique()
    assert (spread == 1).all(), f"Specimens with mismatching halves: {spread[spread > 1].index.tolist()}"
    assert (meta.groupby(COL_SPECIMEN).size() == 2).all(), "Each specimen should have exactly 2 images"
    return meta


def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    meta = load_metadata()
    paths = [IMG_DIR / f for f in meta[COL_FILE]]   # order comes from metadata, not dir()
    missing = [p for p in paths if not p.exists()]
    assert not missing, f"Missing images: {missing[:3]}..."

    size = COMMON_SIZE or tuple(np.min([Image.open(p).size[::-1] for p in paths], axis=0))
    print(f"{len(paths)} images, common size (h, w) = {size}")

    rows = [image_features(load_gray(p, size)) for p in paths]
    per_image = pd.concat([meta, pd.DataFrame(rows)], axis=1)
    per_image.to_csv(RESULTS_DIR / "features_per_image.csv", index=False)

    # average the two halves -> one row per specimen
    spec = (per_image.groupby(COL_SPECIMEN)
            .agg({**{c: "mean" for c in rows[0]}, COL_STRENGTH: "first", COL_BATCH: "first"})
            .reset_index())
    spec.to_csv(RESULTS_DIR / "features_per_specimen.csv", index=False)
    y = spec[COL_STRENGTH].to_numpy()
    X = spec[FEATURES].to_numpy()
    batches = spec[COL_BATCH].to_numpy()

    # ---- correlations (all computed features, with p-values) ----
    print("\nPearson correlation with strength (n = %d specimens):" % len(y))
    for c in rows[0]:
        r, p = stats.pearsonr(spec[c], y)
        print(f"  {c:15s} r = {r:+.3f}   p = {p:.3f}")

    # ---- model: scaling lives INSIDE the pipeline so CV folds never leak ----
    model = make_pipeline(StandardScaler(), LinearRegression())
    model.fit(X, y)
    pred_in = model.predict(X)
    print(f"\nIn-sample  R2 = {r2_score(y, pred_in):.3f}  (optimistic, training fit only)")
    print("Standardised coefficients:",
          dict(zip(FEATURES, np.round(model[-1].coef_, 3))))

    # ---- honest validation ----
    # Leave-one-batch-out: whole batches (similar mix/strength) are held out,
    # so near-duplicates can't leak between train and test.
    pred_cv = cross_val_predict(model, X, y, groups=batches, cv=LeaveOneGroupOut())
    r2_cv = r2_score(y, pred_cv)
    rmse_cv = mean_squared_error(y, pred_cv) ** 0.5
    print(f"Leave-one-batch-out  R2 = {r2_cv:.3f}   RMSE = {rmse_cv:.2f} MPa "
          f"(predicting the mean gives RMSE = {y.std():.2f} MPa)")

    # Permutation test: how often does shuffling strengths give a CV R2 this good?
    rng = np.random.default_rng(0)
    null = []
    for _ in range(500):
        yp = rng.permutation(y)
        null.append(r2_score(yp, cross_val_predict(model, X, yp, groups=batches,
                                                   cv=LeaveOneGroupOut())))
    p_perm = (np.sum(np.array(null) >= r2_cv) + 1) / (len(null) + 1)
    print(f"Permutation-test p (CV R2) = {p_perm:.3f}")

    # ---- plot ----
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5), sharex=True, sharey=True)
    lims = [min(y.min(), pred_cv.min(), pred_in.min()) - 1, max(y.max(), pred_cv.max(), pred_in.max()) + 1]
    for a, pred, ttl in [(ax[0], pred_in, f"In-sample (R² = {r2_score(y, pred_in):.2f})"),
                         (ax[1], pred_cv, f"Leave-one-batch-out (R² = {r2_cv:.2f})")]:
        a.scatter(y, pred, c=batches, cmap="tab10", s=70, edgecolor="k")
        a.plot(lims, lims, "k--", lw=1)
        a.set(title=ttl, xlabel="Measured strength (MPa)", xlim=lims, ylim=lims)
        a.grid(alpha=.3)
    ax[0].set_ylabel("Predicted strength (MPa)")
    fig.suptitle("Coloured by batch: " + ", ".join(FEATURES))
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "predicted_vs_measured.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    main()