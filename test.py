import sys
import numpy as np
import pandas as pd

from dfm import DFM

def _transform(s, code):
    """Apply the stationarity transform FRED-MD specifies for a series."""
    if code == 1: return s                        # level
    if code == 2: return s.diff()                 # 1st difference
    if code == 3: return s.diff().diff()          # 2nd difference
    if code == 4: return np.log(s)                # log level
    if code == 5: return np.log(s).diff()         # log 1st difference
    if code == 6: return np.log(s).diff().diff()  # log 2nd difference
    if code == 7: return s.pct_change().diff()    # 1st diff of pct change
    return s
 
 
def load_fredmd(path="current.csv", series=None, standardize=True):
    """Read FRED-MD current.csv -> clean, stationary, standardized panel.
 
    Parameters
    ----------
    path : str
        Path to FRED-MD current.csv.
    series : list[str] or None
        Which series to keep. None = all of them.
    standardize : bool
        Z-score each series so scales are comparable (DFM expects this).
 
    Returns a (T x N) DataFrame: rows = months, columns = series.
    """
    raw = pd.read_csv(path)
 
    # row 0 is the per-series transform code (indexed by column name)
    tcodes = raw.iloc[0]
    data = raw.iloc[1:].reset_index(drop=True)   # drop the transform row
 
    # date -> index
    data["sasdate"] = pd.to_datetime(data["sasdate"])
    data = data.set_index("sasdate")
 
    # coerce EVERY series to numeric up front, while columns are still clean.
    # (per-column call; passing errors= through DataFrame.apply is flaky across
    #  pandas versions.) Doing this before the transforms is what keeps stray
    # object-dtype junk out of the math.
    data = data.apply(lambda c: pd.to_numeric(c, errors="coerce"))
 
    if series is not None:
        data = data[series]
 
    # apply each series' stationarity transform
    out = pd.DataFrame({c: _transform(data[c], int(tcodes[c])) for c in data.columns})
 
    out = out.iloc[2:]            # first 2 rows are NaN from differencing
    out = out.dropna(axis=1)      # drop series with gaps / short history
 
    if standardize:
        sd = out.std().replace(0, 1.0)   # guard constant columns (std=0)
        out = (out - out.mean()) / sd
 
    return out

def mean_abs_corr(X):
    """Average absolute cross-series correlation (the co-movement a DFM captures)."""
    C = np.corrcoef(np.asarray(X), rowvar=False)
    return np.abs(C[np.triu_indices(X.shape[1], 1)]).mean()
 
 
def _ar1(x):
    x = np.asarray(x)
    return np.corrcoef(x[1:], x[:-1])[0, 1]
 
 
def per_series_stats(X):
    """Per-series volatility (std) and persistence (lag-1 autocorrelation)."""
    X = np.asarray(X)
    stds = X.std(axis=0)
    ars = np.array([_ar1(X[:, i]) for i in range(X.shape[1])])
    return stds, ars
 
 
def diagnostics(real, panels):
    """Print a real-vs-synthetic comparison table."""
    r_corr = mean_abs_corr(real)
    s_corr = np.mean([mean_abs_corr(p) for p in panels])
 
    r_std, r_ar = per_series_stats(real)
    s_std = np.concatenate([per_series_stats(p)[0] for p in panels])
    s_ar = np.concatenate([per_series_stats(p)[1] for p in panels])
 
    print("=" * 52)
    print(f"{'metric':<24}{'real':>12}{'synth':>12}")
    print("-" * 52)
    print(f"{'mean |corr| (co-move)':<24}{r_corr:>12.3f}{s_corr:>12.3f}")
    print(f"{'median std (vol)':<24}{np.median(r_std):>12.3f}{np.median(s_std):>12.3f}")
    print(f"{'median AR(1) (persist)':<24}{np.median(r_ar):>12.3f}{np.median(s_ar):>12.3f}")
    print("=" * 52)


"""
Main
"""
if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "current.csv"
 
    print(f"loading {path} ...")
    real = load_fredmd(path)
    print(f"panel: {real.shape[0]} months x {real.shape[1]} series\n")
 
    dfm = DFM(k_factors=3, factor_order=2, maxiter=100).fit_dfm(real)
    panels = dfm.generate_df(n_panels=20, n_steps=real.shape[0], burn_in=60, seed=1)
 
    diagnostics(real, panels)