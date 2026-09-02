"""Explicit, auditable promotion of a training/pipeline.py run to production.

Deliberately not automatic: promotion touches the live app's model artifacts,
and this project's own history includes retrains that looked promising by an
inner metric but regressed on the real evaluation (the 3-class U-Net retrain
attempt) - a human should look at the numbers before they go live. This
script's only job is to make that human decision fast and hard to get wrong:
it refuses by default if the candidate doesn't beat the currently-promoted
run's F1-macro, and requires --force to override with a stated reason.
"""
import argparse
import json
import shutil
from pathlib import Path

import joblib
import pandas as pd

TRAINING_DIR = Path(__file__).parent
REPO_ROOT = TRAINING_DIR.parent
RUNS_DIR = TRAINING_DIR / "runs"
PROD_MODEL_PATH = REPO_ROOT / "best_prognostic_model.pkl"
PROD_CSV_PATH = REPO_ROOT / "combined_radiomics_features.csv"
BASE_CSV = REPO_ROOT / "combined_radiomics_features_FIXED.csv"
INFARCT_TRAIN_CSV = TRAINING_DIR / "infarct_features_train.csv"
INFARCT_TEST_CSV = TRAINING_DIR / "infarct_features_test.csv"


def load_record(run_id: str) -> dict:
    path = RUNS_DIR / f"{run_id}.json"
    if not path.exists():
        raise SystemExit(f"No run record found at {path}")
    return json.loads(path.read_text())


def currently_promoted():
    records = [json.loads(p.read_text()) for p in RUNS_DIR.glob("*.json")]
    promoted = [r for r in records if isinstance(r, dict) and r.get("promoted")]
    if not promoted:
        return None
    return max(promoted, key=lambda r: r["timestamp"])


def build_production_csv(use_infarct_features: bool) -> pd.DataFrame:
    base_df = pd.read_csv(BASE_CSV, dtype={"PatientID": str})
    base_df["PatientID"] = base_df["PatientID"].str.zfill(3)
    if not use_infarct_features:
        return base_df
    infarct_train = pd.read_csv(INFARCT_TRAIN_CSV, dtype={"PatientID": str})
    infarct_test = pd.read_csv(INFARCT_TEST_CSV, dtype={"PatientID": str})
    infarct_all = pd.concat([infarct_train, infarct_test], ignore_index=True)
    infarct_all["PatientID"] = infarct_all["PatientID"].str.zfill(3)
    return base_df.merge(infarct_all, on="PatientID", how="inner")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_id")
    parser.add_argument("--force", action="store_true", help="promote even if the candidate regresses on F1-macro")
    args = parser.parse_args()

    candidate = load_record(args.run_id)
    baseline = currently_promoted()

    print(f"Candidate run: {candidate['run_id']} ({candidate['model_family']}, {candidate['n_features']} features)")
    cand_f1 = candidate["cv_metrics"]["f1_macro"]["mean"]
    print(f"  F1-macro: {cand_f1:.3f} +/- {candidate['cv_metrics']['f1_macro']['std']:.3f}")

    if baseline is None:
        print("No currently-promoted model on record - nothing to compare against, proceeding.")
    else:
        base_f1 = baseline["cv_metrics"]["f1_macro"]["mean"]
        print(f"Currently promoted: {baseline['run_id']} - F1-macro: {base_f1:.3f}")
        if cand_f1 <= base_f1 and not args.force:
            print(f"\nREFUSING to promote: candidate F1-macro ({cand_f1:.3f}) does not beat the "
                  f"currently-promoted run ({base_f1:.3f}). Re-run with --force to override.")
            raise SystemExit(1)
        if cand_f1 <= base_f1:
            print(f"\n--force set: promoting despite F1-macro not improving ({cand_f1:.3f} <= {base_f1:.3f}).")

    model_src = REPO_ROOT / candidate["model_artifact"]
    shutil.copy(model_src, PROD_MODEL_PATH)
    print(f"Copied {model_src} -> {PROD_MODEL_PATH}")

    prod_csv = build_production_csv(candidate["use_infarct_features"])
    prod_csv.to_csv(PROD_CSV_PATH, index=False)
    print(f"Wrote {PROD_CSV_PATH} ({prod_csv.shape})")

    # sanity check: the model's recorded feature_columns must all exist in the CSV we just wrote
    missing = [c for c in candidate["feature_columns"] if c not in prod_csv.columns]
    if missing:
        raise SystemExit(f"REFUSING: promoted model expects columns not present in the production CSV: {missing}")

    if baseline is not None:
        baseline_path = RUNS_DIR / f"{baseline['run_id']}.json"
        baseline["promoted"] = False
        baseline_path.write_text(json.dumps(baseline, indent=2))

    candidate["promoted"] = True
    (RUNS_DIR / f"{candidate['run_id']}.json").write_text(json.dumps(candidate, indent=2))
    print(f"\nPromoted {candidate['run_id']} to production. Test locally, then commit "
          f"best_prognostic_model.pkl, combined_radiomics_features.csv, and training/runs/ together.")


if __name__ == "__main__":
    main()
