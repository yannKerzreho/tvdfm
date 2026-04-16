"""
main.py — TVDFM Out-of-Sample Validation
=========================================

Expanding-window nowcasting benchmark:
  Baseline  →  Dynamic Factor Model (DFM, static loadings)
  tvDFM     →  Time-Varying DFM driven by a GRU exposure module

The evaluation follows a strict real-time protocol:
  • For each test year Y, the model is trained on all data up to Dec (Y-1).
  • GDP growth for each quarter of Y is predicted at six information lags
    (h=1 … h=6 months before the official release).
  • No look-ahead: standardisation statistics, factor selection, and
    EM initialisation are all computed on the training window only.

Key empirical finding (FRED-MD, 2015-2025):
  ssm_fullEM_autolam achieves −27 / −31 % RMSE vs DFM during the COVID
  recession quarters at h=1 / h=2, and −12 to −30 % pre-COVID at short
  horizons.  λ is selected each year by validation GDP MSE — no manual
  tuning.  MCS eliminates DFM at h=2 (p=0.004) and grouped h=1-3
  (p=0.017) over the full 2015-2025 sample.

──────────────────────────────────────────────────────────────

Usage examples
--------------
  # Default run — DFM full vs ssm_fullEM_autolam, h=1..6, 2015-2025
  python main.py

  # Custom λ candidate grid
  python main.py --lam_candidates 0.001 0.01 0.1 1.0 10.0

  # Fixed-λ variant (skip per-year search)
  python main.py --model ssm_fullEM --lam 1.0

  # Change OOS window
  python main.py --test_start 2010 --test_end 2025

  # Short horizons only, quiet mode, save results
  python main.py --horizons 1 2 3 --quiet --output_dir results/quick

  # Use strict-training DFM as baseline (no val data in EM)
  python main.py --baseline train

  # GDP-only TV loadings variant
  python main.py --model ssm_fullEM_target --lam 0.01
"""

from __future__ import annotations

import argparse
import io
import sys
import time
import warnings
import contextlib
import textwrap
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ── paths ──────────────────────────────────────────────────────────────────
ROOT_DIR       = Path(__file__).resolve().parent
EXPERIMENT_DIR = ROOT_DIR / "experiment"
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(EXPERIMENT_DIR))

from tvdfm import TVDFModel

# Experiment utilities: data loading and MCS tests
try:
    from data_utils import (
        load_raw_data, standardize,
        make_horizon_dataset, extract_gdp_pred,
        get_quarter_ends, TARGET,
    )
    from mcs import run_mcs_analysis, run_mcs_grouped, mcs_summary_text
    _HAS_EXPERIMENT = True
except ImportError:
    _HAS_EXPERIMENT = False
    print("[warn] experiment/ utilities not found — data loading unavailable.")


# ══════════════════════════════════════════════════════════════════════════════
# Argument parsing
# ══════════════════════════════════════════════════════════════════════════════

def get_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="TVDFM out-of-sample validation: DFM vs time-varying DFM (GRU).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Models
            ------
            ssm_fullEM         TV-loadings + GRU exposure + SSM imputation
                               + full-history EM init (best overall config).
            ssm_fullEM_autolam Same as ssm_fullEM but selects λ from
                               --lam_candidates by validation GDP MSE each year.
            ssm_fullEM_target  Same as ssm_fullEM but TV loading restricted to
                               the GDP row only.
            short              TV-loadings + GRU + forward-fill imputation
                               (faster, no SSM step; competitive at h=1,2).

            Horizons
            --------
            h=1  2 of 3 quarter months observed  (late nowcast)
            h=2  1 of 3 quarter months observed
            h=3  0 months observed (quarter-start nowcast)
            h=4  1 month before target quarter
            h=5  2 months before
            h=6  end of previous quarter        (early forecast)
        """),
    )

    # ── data ────────────────────────────────────────────────────────────────
    data = p.add_argument_group("data")
    data.add_argument(
        "--data_dir", type=str,
        default=str(EXPERIMENT_DIR / "data"),
        help="Directory containing 2026-02-MD.csv and 2026-02-QD.csv "
             "(default: experiment/data/)",
    )
    data.add_argument(
        "--data_start", type=str, default="1990-01-01",
        metavar="YYYY-MM-DD",
        help="Start of training history (default: 1990-01-01)",
    )
    data.add_argument(
        "--test_start", type=int, default=2015,
        metavar="YYYY",
        help="First OOS test year (default: 2015)",
    )
    data.add_argument(
        "--test_end", type=int, default=2025,
        metavar="YYYY",
        help="Last OOS test year inclusive (default: 2025)",
    )
    data.add_argument(
        "--horizons", type=int, nargs="+", default=[1, 2, 3, 4, 5, 6],
        metavar="H",
        help="Evaluation horizons in months (default: 1 2 3 4 5 6)",
    )

    # ── models ───────────────────────────────────────────────────────────────
    mdl = p.add_argument_group("models")
    mdl.add_argument(
        "--baseline", type=str, choices=["full", "train"], default="full",
        help="DFM baseline: 'full' trains EM on the entire window including "
             "validation (val_n=0); 'train' holds out the last val_n months "
             "from EM (default: full)",
    )
    mdl.add_argument(
        "--model",
        type=str,
        choices=["ssm_fullEM_autolam", "ssm_fullEM", "ssm_fullEM_target", "short"],
        default="ssm_fullEM_autolam",
        help="tvDFM variant to compare against the DFM baseline "
             "(default: ssm_fullEM_autolam)",
    )
    mdl.add_argument(
        "--lam", type=float, default=1.0,
        metavar="λ",
        help="Lambda-deviation penalty weight (fixed). Ignored when "
             "--model ssm_fullEM_autolam is used. (default: 1.0)",
    )
    mdl.add_argument(
        "--lam_candidates", type=float, nargs="+",
        default=[0.01, 0.1, 1.0, 10.0],
        metavar="λ",
        help="Candidate λ values for ssm_fullEM_autolam. The one with lowest "
             "validation GDP MSE is selected each year. (default: 0.01 0.1 1.0 10.0)",
    )

    # ── architecture ─────────────────────────────────────────────────────────
    arch = p.add_argument_group("architecture")
    arch.add_argument("--n_factors",    type=int, default=3,  help="Number of latent factors (default: 3)")
    arch.add_argument("--n_select",     type=int, default=20, help="Covariates kept by iterative-PCA preselection (default: 20)")
    arch.add_argument("--factor_order", type=int, default=1,  help="VAR order of factor dynamics (default: 1)")
    arch.add_argument("--error_order",  type=int, default=1,  help="AR order of idiosyncratic noise (default: 1)")
    arch.add_argument("--hidden_size",  type=int, default=16, help="GRU hidden state size (default: 16)")
    arch.add_argument("--mlp_width",    type=int, default=64, help="MLP readout hidden width (default: 64)")
    arch.add_argument("--mlp_depth",    type=int, default=2,  help="MLP readout depth (default: 2)")
    arch.add_argument("--dropout",      type=float, default=0.1, help="GRU dropout (default: 0.1)")

    # ── training ──────────────────────────────────────────────────────────────
    train = p.add_argument_group("training")
    train.add_argument("--val_n",    type=int,   default=48,  help="Validation window in months (default: 48)")
    train.add_argument("--n_epochs", type=int,   default=200, help="Max GRU epochs (default: 200)")
    train.add_argument("--patience", type=int,   default=7,   help="Early stopping patience (default: 7)")
    train.add_argument("--em_iter",  type=int,   default=50,  help="EM iterations for DFM init (default: 50)")
    train.add_argument("--seed",     type=int,   default=42,  help="Random seed (default: 42)")

    # ── output ────────────────────────────────────────────────────────────────
    out = p.add_argument_group("output")
    out.add_argument(
        "--output_dir", type=str, default=None,
        metavar="DIR",
        help="Save CSV results to this directory (default: results/tvdfm_YYYYMMDD_HHMMSS/)",
    )
    out.add_argument(
        "--no_mcs", action="store_true",
        help="Skip Model Confidence Set tests (faster)",
    )
    out.add_argument(
        "--quiet", action="store_true",
        help="Suppress per-epoch training output",
    )

    return p.parse_args()


# ══════════════════════════════════════════════════════════════════════════════
# Model factories
# ══════════════════════════════════════════════════════════════════════════════

def build_dfm(args: argparse.Namespace, val_n: int) -> TVDFModel:
    """Static DFM: EM initialised, no GRU, no time-varying parameters."""
    return TVDFModel(
        n_factors         = args.n_factors,
        exposure          = None,
        factor_order      = args.factor_order,
        error_order       = args.error_order,
        dfm_init          = "preselected",
        target_col        = TARGET,
        n_select          = args.n_select,
        presel_method     = "iterative_pca",
        lag_x             = 1,
        em_iter           = args.em_iter,
        n_epochs          = 0,
        val_n             = val_n,
        random_state      = args.seed,
        verbose           = not args.quiet,
    )


def build_tvdfm(args: argparse.Namespace, lam_override: Optional[float] = None) -> TVDFModel:
    """Time-Varying DFM: GRU exposure with the config selected via --model.

    Parameters
    ----------
    lam_override : float, optional
        When provided, overrides args.lam (used by select_lambda for grid search).
    """
    # Imputation and EM strategy
    if args.model in ("ssm_fullEM", "ssm_fullEM_autolam", "ssm_fullEM_target"):
        imputation = "ssm"
        em_val_n   = 0      # EM sees the full training window
    else:  # "short"
        imputation = "forward"
        em_val_n   = None

    # GDP-only TV loading for ssm_fullEM_target
    lambda_indices = (0,) if args.model == "ssm_fullEM_target" else None

    lam = lam_override if lam_override is not None else args.lam

    return TVDFModel(
        n_factors             = args.n_factors,
        exposure              = "gru",
        tv_Lambda             = True,
        tv_A                  = False,
        factor_order          = args.factor_order,
        error_order           = args.error_order,
        hidden_size           = args.hidden_size,
        mlp_width             = args.mlp_width,
        mlp_depth             = args.mlp_depth,
        dropout               = args.dropout,
        imputation            = imputation,
        em_val_n              = em_val_n,
        lambda_indices        = lambda_indices,
        dfm_init              = "preselected",
        target_col            = TARGET,
        n_select              = args.n_select,
        presel_method         = "iterative_pca",
        lag_x                 = 1,
        em_iter               = args.em_iter,
        lr_ssm                = 1e-3,
        lr_exposure           = 3e-3,
        wd_exposure           = 0.0,
        wd_ssm                = 0.0,
        kf_ll_weight          = 0.0,
        lambda_dev_weight     = lam,
        horizon_augment_max   = 3,
        val_horizon           = 2,
        val_full_kf           = True,
        n_epochs              = args.n_epochs,
        patience              = args.patience,
        val_n                 = args.val_n,
        random_state          = args.seed,
        verbose               = not args.quiet,
    )


def select_lambda(
    args: argparse.Namespace,
    candidates: list,
    data_train_std: pd.DataFrame,
) -> tuple[TVDFModel, float]:
    """
    Train one model per λ candidate and return the best by validation GDP MSE.

    All candidates are trained quietly (regardless of --quiet) so the λ
    search doesn't flood the console.  The selected model is already fitted —
    no need to retrain.

    Parameters
    ----------
    candidates : list of float
        λ values to evaluate.
    data_train_std : pd.DataFrame
        Standardised training panel.

    Returns
    -------
    best_model : TVDFModel (fitted)
    best_lam   : float
    """
    best_lam   = candidates[0]
    best_val   = float("inf")
    best_model = None

    for lam in candidates:
        m = build_tvdfm(args, lam_override=lam)
        try:
            m, _ = _capture_fit(m, data_train_std, quiet=True)
        except Exception as e:
            print(f"    [autolam] λ={lam}: training failed ({e}), skipping.")
            continue
        val = getattr(getattr(m, "manager_", None), "best_val_loss_", float("inf"))
        print(f"    [autolam] λ={lam:<6}  val_GDP_MSE={val:.5f}")
        if val < best_val:
            best_val   = val
            best_lam   = lam
            best_model = m

    if best_model is None:
        raise RuntimeError("All λ candidates failed during training.")

    print(f"    [autolam] → selected λ={best_lam}  (val={best_val:.5f})")
    return best_model, best_lam


# ══════════════════════════════════════════════════════════════════════════════
# Training + evaluation helpers
# ══════════════════════════════════════════════════════════════════════════════

def _capture_fit(model: TVDFModel, data_std: pd.DataFrame, quiet: bool) -> tuple[TVDFModel, str]:
    """Fit model, optionally suppressing stdout. Returns (model, captured_log)."""
    buf = io.StringIO()
    ctx = contextlib.redirect_stdout(buf) if quiet else contextlib.nullcontext()
    with ctx:
        model.fit(data_std, target_series=TARGET)
    return model, buf.getvalue()


def _epochs_info(model: TVDFModel) -> str:
    mgr = getattr(model, "manager_", None)
    n   = getattr(mgr, "n_epochs_done_", 0)
    bv  = getattr(mgr, "best_val_loss_", float("nan"))
    if n == 0:
        return "EM only"
    bv_s = f"{bv:.5f}" if np.isfinite(float(bv)) else "n/a"
    return f"{n} ep · best_val={bv_s}"


def evaluate_model(
    model: TVDFModel,
    data_full_std: pd.DataFrame,
    test_qends: list,
    horizons: list,
) -> pd.DataFrame:
    rows = []
    for h in horizons:
        for q in test_qends:
            pred_data = make_horizon_dataset(data_full_std, q, horizon=h, keep_full=True)
            try:
                preds  = model.predict(pred_data)
                sliced = pred_data[model.selected_columns_]
                y_pred = extract_gdp_pred(preds, sliced, q)
            except Exception:
                y_pred = np.nan
            y_true = float(data_full_std.loc[q, TARGET])
            sq_err = (y_pred - y_true) ** 2 if np.isfinite(y_pred) else np.nan
            rows.append(dict(horizon=h, test_date=q, y_true=y_true,
                             y_pred=y_pred, sq_err=sq_err))
    return pd.DataFrame(rows)


def period_label(d: pd.Timestamp) -> str:
    COVID_START = pd.Timestamp("2020-01-01")
    COVID_END   = pd.Timestamp("2021-12-31")
    if d < COVID_START:    return "pre_covid"
    if d <= COVID_END:     return "covid"
    return "post_covid"


# ══════════════════════════════════════════════════════════════════════════════
# Pretty printing
# ══════════════════════════════════════════════════════════════════════════════

_LINE = "─" * 66
_DLINE = "═" * 66

def _banner(args: argparse.Namespace, test_years: list) -> None:
    baseline_name = f"DFM ({'full-history EM' if args.baseline == 'full' else 'strict-train EM'})"
    if args.model == "ssm_fullEM_autolam":
        tvdfm_name = f"{args.model}  λ∈{args.lam_candidates}"
    else:
        tvdfm_name = f"{args.model}  λ={args.lam}"
    oos_years = f"{test_years[0]} – {test_years[-1]}  ({len(test_years)} years)"

    print(f"\n{_DLINE}")
    print(f"  TVDFM  ·  Expanding-Window GDP Nowcast Benchmark")
    print(_DLINE)
    print(f"  Baseline  :  {baseline_name}")
    print(f"  tvDFM     :  {tvdfm_name}")
    print(f"  OOS       :  {oos_years}")
    print(f"  Horizons  :  h = {args.horizons}")
    print(f"  Data      :  FRED-MD  ({args.data_start} onwards)")
    print(_DLINE)


def _results_table(
    df_all: pd.DataFrame,
    baseline_label: str,
    tvdfm_label: str,
    horizons: list,
) -> None:
    print(f"\n{_DLINE}")
    print("  OUT-OF-SAMPLE RESULTS")
    print(_DLINE)

    # Overall RMSE
    rmse = (
        df_all.groupby(["config", "horizon"])["sq_err"]
        .mean().apply(np.sqrt).unstack("horizon")
    )
    print("\n  RMSE by horizon:")
    print(f"  {'':28s}", end="")
    for h in horizons:
        print(f"  h={h}", end="")
    print()
    for cfg in [baseline_label, tvdfm_label]:
        if cfg not in rmse.index:
            continue
        row = rmse.loc[cfg]
        print(f"  {cfg:<28s}", end="")
        for h in horizons:
            v = row.get(h, np.nan)
            print(f"  {v:.3f}" if np.isfinite(v) else "    —  ", end="")
        print()

    # Ratio row
    if baseline_label in rmse.index and tvdfm_label in rmse.index:
        print(f"  {'  Ratio (tvDFM / DFM)':<28s}", end="")
        for h in horizons:
            b = rmse.loc[baseline_label].get(h, np.nan)
            t = rmse.loc[tvdfm_label].get(h, np.nan)
            r = t / b if (np.isfinite(b) and b > 0) else np.nan
            print(f"  {r:.3f}" if np.isfinite(r) else "    —  ", end="")
        print()

    # Sub-period breakdown
    periods = [("pre_covid", "Pre-COVID  (up to 2019)"),
               ("covid",     "COVID      (2020–2021) ★"),
               ("post_covid","Post-COVID (2022+)    ")]

    print("\n  RMSE by sub-period (ratio tvDFM / DFM):")
    for period_key, period_label_str in periods:
        sub = df_all[df_all.period == period_key]
        if len(sub) == 0:
            continue
        rmse_sub = (
            sub.groupby(["config", "horizon"])["sq_err"]
            .mean().apply(np.sqrt).unstack("horizon")
        )
        if baseline_label not in rmse_sub.index or tvdfm_label not in rmse_sub.index:
            continue
        print(f"\n    {period_label_str}")
        print(f"    {'':28s}", end="")
        for h in horizons:
            print(f"  h={h}", end="")
        print()
        for cfg in [baseline_label, tvdfm_label]:
            row = rmse_sub.loc[cfg]
            print(f"    {cfg:<28s}", end="")
            for h in horizons:
                v = row.get(h, np.nan)
                print(f"  {v:.3f}" if np.isfinite(v) else "    —  ", end="")
            print()
        # ratio
        print(f"    {'  Ratio':<28s}", end="")
        for h in horizons:
            b = rmse_sub.loc[baseline_label].get(h, np.nan)
            t = rmse_sub.loc[tvdfm_label].get(h, np.nan)
            r = t / b if (np.isfinite(b) and b > 0) else np.nan
            print(f"  {r:.3f}" if np.isfinite(r) else "    —  ", end="")
        print()


def _covid_callout(df_all: pd.DataFrame, baseline_label: str, tvdfm_label: str, horizons: list) -> None:
    """Print a highlighted box if a meaningful COVID edge is detected."""
    covid = df_all[df_all.period == "covid"]
    if len(covid) == 0:
        return
    rmse_covid = (
        covid.groupby(["config", "horizon"])["sq_err"]
        .mean().apply(np.sqrt).unstack("horizon")
    )
    if baseline_label not in rmse_covid.index or tvdfm_label not in rmse_covid.index:
        return
    best_gain = 0.0
    best_h    = None
    for h in horizons:
        b = rmse_covid.loc[baseline_label].get(h, np.nan)
        t = rmse_covid.loc[tvdfm_label].get(h, np.nan)
        if np.isfinite(b) and np.isfinite(t) and b > 0:
            gain = 1.0 - t / b
            if gain > best_gain:
                best_gain, best_h = gain, h

    if best_gain >= 0.10:  # at least 10% gain somewhere
        print(f"\n  {'★ ' * 10}")
        print(f"  COVID STRUCTURAL-BREAK ADVANTAGE (2020–2021)")
        print(f"  {tvdfm_label} vs {baseline_label}:")
        for h in horizons:
            b = rmse_covid.loc[baseline_label].get(h, np.nan)
            t = rmse_covid.loc[tvdfm_label].get(h, np.nan)
            if np.isfinite(b) and b > 0:
                pct = (1.0 - t / b) * 100
                bar = "▓" * max(0, int(abs(pct) / 3))
                sign = "−" if pct > 0 else "+"
                print(f"    h={h}   DFM={b:.3f}  tvDFM={t:.3f}   "
                      f"{sign}{abs(pct):.1f}%  {bar}")
        print(f"  {'★ ' * 10}")


def _mcs_section(df_all: pd.DataFrame, horizons: list) -> None:
    print(f"\n{_DLINE}")
    print("  MODEL CONFIDENCE SET  (α=10%, circular block bootstrap, B=2000)")
    print(_DLINE)

    # Per-horizon MCS
    mcs_df  = run_mcs_analysis(df_all, horizons, alpha=0.10, n_boot=2000, seed=42)
    print(mcs_summary_text(mcs_df, "[per-horizon]"))

    # Grouped MCS
    groups = [[h for h in grp if h in horizons] for grp in [[1,2,3],[4,5,6]]]
    groups = [g for g in groups if len(g) > 0]
    if groups:
        mcs_grp = run_mcs_grouped(df_all, groups, alpha=0.10, n_boot=2000, seed=42)
        print(mcs_summary_text(mcs_grp, "[grouped h1-3 / h4-6]"))


# ══════════════════════════════════════════════════════════════════════════════
# Main loop
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    if not _HAS_EXPERIMENT:
        print("ERROR: experiment/ utilities are required. "
              "Run from the project root with: python main.py")
        sys.exit(1)

    args = parser_args = get_args()

    # ── output directory ──────────────────────────────────────────────────
    if args.output_dir is None:
        from datetime import datetime
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = str(ROOT_DIR / "results" / f"tvdfm_{ts}")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── load data ─────────────────────────────────────────────────────────
    data_start = pd.Timestamp(args.data_start)
    test_years = list(range(args.test_start, args.test_end + 1))

    print("\nLoading FRED-MD panel …")
    data, _ = load_raw_data(start_date=data_start)
    print(f"  Panel: {data.shape[0]} months × {data.shape[1]} series  "
          f"({data.index[0].date()} – {data.index[-1].date()})")

    _banner(args, test_years)

    baseline_label = "DFM_full" if args.baseline == "full" else "DFM_train"
    if args.model == "ssm_fullEM_autolam":
        tvdfm_label = "ssm_fullEM_autolam"
    else:
        tvdfm_label = f"{args.model}_lam{args.lam}"

    all_eval: list[pd.DataFrame] = []
    selected_lams: dict[int, float] = {}   # year → selected λ (autolam only)

    # ── expanding-window loop ─────────────────────────────────────────────
    for year in test_years:
        train_end  = pd.Timestamp(f"{year - 1}-12-31")
        data_train = data.loc[data.index <= train_end]
        T_train    = len(data_train)

        if T_train <= args.val_n + 12:
            print(f"\n  [skip] {year}: only {T_train} training months.")
            continue

        data_train_std, stats = standardize(data_train)
        data_full_std, _      = standardize(data, stats=stats)

        test_qends = [
            q for q in get_quarter_ends(
                pd.Timestamp(f"{year}-01-01"),
                pd.Timestamp(f"{year}-12-31"),
            ) if q in data_full_std.index
        ]
        if not test_qends:
            continue

        print(f"\n{_LINE}")
        print(f"  Year {year}  —  training on {T_train} months "
              f"({data_train.index[0].date()} … {data_train.index[-1].date()})")
        print(f"  Val window : last {args.val_n} months  "
              f"({data_train.index[-args.val_n].date()} … {data_train.index[-1].date()})")

        results_year: list[pd.DataFrame] = []

        # Build the tvDFM for this year (autolam runs a λ-grid search first)
        t0_tv = time.time()
        selected_lam = None
        if args.model == "ssm_fullEM_autolam":
            try:
                tv_model, selected_lam = select_lambda(
                    args, args.lam_candidates, data_train_std
                )
                selected_lams[year] = selected_lam
            except Exception as e:
                print(f"  [ERROR] {tvdfm_label} λ-search: {e}")
                tv_model = None
        else:
            try:
                tv_model, _ = _capture_fit(build_tvdfm(args), data_train_std, quiet=args.quiet)
            except Exception as e:
                print(f"  [ERROR] {tvdfm_label}: {e}")
                tv_model = None
        tv_elapsed = time.time() - t0_tv

        for tag, model, elapsed in [
            (baseline_label,
             None,   # fitted below
             None),
            (tvdfm_label, tv_model, tv_elapsed),
        ]:
            if tag == baseline_label:
                t0 = time.time()
                dfm_spec = build_dfm(args, val_n=0 if args.baseline == "full" else args.val_n)
                try:
                    model, log = _capture_fit(dfm_spec, data_train_std, quiet=args.quiet)
                    if not args.quiet and log.strip():
                        for line in log.splitlines():
                            if "| train" in line or "epoch" in line.lower():
                                print(f"    {line.strip()}")
                except Exception as e:
                    print(f"  [ERROR] {tag}: {e}")
                    continue
                elapsed = time.time() - t0
            else:
                if model is None:
                    continue  # training failed earlier
                if not args.quiet and args.model != "ssm_fullEM_autolam":
                    pass  # log already captured above

            df_eval = evaluate_model(model, data_full_std, test_qends, args.horizons)
            df_eval["config"]       = tag
            df_eval["year"]         = year
            df_eval["period"]       = df_eval["test_date"].apply(period_label)
            if tag == tvdfm_label and selected_lam is not None:
                df_eval["selected_lam"] = selected_lam
            results_year.append(df_eval)

            ep_info = _epochs_info(model)
            rmse_h1 = np.sqrt(df_eval[df_eval.horizon == 1]["sq_err"].mean())
            rmse_h3 = np.sqrt(df_eval[df_eval.horizon == 3]["sq_err"].mean()) if 3 in args.horizons else np.nan
            h1_s = f"h1={rmse_h1:.3f}" if np.isfinite(rmse_h1) else "h1=NaN"
            h3_s = f"h3={rmse_h3:.3f}" if np.isfinite(rmse_h3) else ""
            lam_s = f"  λ={selected_lam}" if selected_lam is not None and tag == tvdfm_label else ""
            print(f"  {tag:<32}  {elapsed:5.0f}s  {h1_s}  {h3_s}{lam_s}  [{ep_info}]")

        if len(results_year) == 2:
            df_base = results_year[0]
            df_tv   = results_year[1]
            all_eval.extend(results_year)

            # Year-level COVID flag
            covid_base = df_base[df_base.period == "covid"]["sq_err"].mean()
            covid_tv   = df_tv[df_tv.period == "covid"]["sq_err"].mean()
            if np.isfinite(covid_base) and np.isfinite(covid_tv) and covid_base > 0:
                gain_pct = (1.0 - np.sqrt(covid_tv) / np.sqrt(covid_base)) * 100
                if abs(gain_pct) >= 5:
                    sign = "−" if gain_pct > 0 else "+"
                    print(f"  ★ COVID quarters: tvDFM {sign}{abs(gain_pct):.0f}% RMSE vs DFM")
        elif results_year:
            all_eval.extend(results_year)

    # ── aggregate results ─────────────────────────────────────────────────
    if not all_eval:
        print("\n[warn] No results collected.")
        return

    df_all = pd.concat(all_eval, ignore_index=True)
    df_all["test_date"] = pd.to_datetime(df_all["test_date"])
    df_all.to_csv(out_dir / "eval_raw.csv", index=False)

    # Save selected-λ trajectory for autolam runs
    if selected_lams:
        lam_df = pd.DataFrame(
            sorted(selected_lams.items()), columns=["year", "selected_lam"]
        )
        lam_df.to_csv(out_dir / "selected_lam.csv", index=False)
        print(f"\n  λ selected per year (autolam):")
        for _, row in lam_df.iterrows():
            print(f"    {int(row.year)}  →  λ={row.selected_lam}")

    _results_table(df_all, baseline_label, tvdfm_label, args.horizons)
    _covid_callout(df_all, baseline_label, tvdfm_label, args.horizons)

    # RMSE summary CSVs
    rmse = (
        df_all.groupby(["config", "horizon"])["sq_err"]
        .mean().apply(np.sqrt).unstack("horizon")
    )
    rmse.to_csv(out_dir / "rmse_overall.csv")
    (
        df_all.groupby(["config", "horizon", "period"])["sq_err"]
        .mean().apply(np.sqrt).unstack("period")
    ).to_csv(out_dir / "rmse_subperiod.csv")
    if baseline_label in rmse.index and tvdfm_label in rmse.index:
        rmse.div(rmse.loc[baseline_label]).to_csv(out_dir / "ratio_to_baseline.csv")

    # ── MCS ──────────────────────────────────────────────────────────────
    if not args.no_mcs:
        _mcs_section(df_all, args.horizons)

    print(f"\n{_DLINE}")
    print(f"  Results saved to: {out_dir}/")
    print(_DLINE + "\n")


# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    main()
