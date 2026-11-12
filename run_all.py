"""Run the complete analysis, in order, on any operating system.

Usage:  python run_all.py --oulad PATH_TO_OULAD_FOLDER [--out results]
(OULAD_DIR / OUT_DIR environment variables are also respected.)
"""
import argparse, os, subprocess, sys, time

STEPS = [
    ("make_clicks.py", "cache the click data"),
    ("pipeline.py", "pre-specified analyses: RQ1 flags, RQ2 models and tests, RQ3 subgroups, robustness"),
    ("early.py", "RQ1 early-detection curves"),
    ("recent_popularity.py", "EXPLORATORY: recent-popularity baseline"),
    ("rq3_recent.py", "EXPLORATORY: RQ3 with the recent-popularity baseline"),
    ("blend_recent.py", "EXPLORATORY: recent popularity blended with CF"),
    ("recent_cohort_check.py", "ROBUSTNESS: contributors chosen from pre-cutoff data only"),
    ("time_aware_cf.py", "EXPLORATORY: time-aware CF"),
    ("audit.py", "participant flow, selection check, relaxed-threshold sensitivity"),
    ("make_figures.py", "Figures 1-3"),
]

ap = argparse.ArgumentParser()
ap.add_argument("--oulad", default=os.environ.get("OULAD_DIR"))
ap.add_argument("--out", default=os.environ.get("OUT_DIR", "results"))
a = ap.parse_args()
if not a.oulad:
    sys.exit("Give the OULAD folder: python run_all.py --oulad PATH_TO_OULAD_FOLDER")
env = dict(os.environ, OULAD_DIR=os.path.abspath(a.oulad), OUT_DIR=os.path.abspath(a.out))
os.makedirs(env["OUT_DIR"], exist_ok=True)
t0 = time.time()
for script, what in STEPS:
    print(f"\n=== {script}: {what}", flush=True)
    subprocess.run([sys.executable, script], env=env, check=True)
print(f"\nAll done in {time.time() - t0:.0f} s. Results are in {env['OUT_DIR']}")
