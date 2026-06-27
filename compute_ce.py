import os, sys, json, csv, re
import argparse
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import numpy as np
import pandas as pd

# -------------------- Constants --------------------
CHEXPERT_COLS = [
    "Atelectasis", "Cardiomegaly", "Consolidation", "Edema",
    "Enlarged Cardiomediastinum", "Fracture", "Lung Lesion",
    "Lung Opacity", "No Finding", "Pleural Effusion",
    "Pleural Other", "Pneumonia", "Pneumothorax", "Support Devices",
]

# -------------------- BLEU(1-4) --------------------
def _ngram_counts(tokens: List[str], n: int) -> Dict[Tuple[str, ...], int]:
    d = {}
    for i in range(len(tokens) - n + 1):
        ng = tuple(tokens[i:i+n])
        d[ng] = d.get(ng, 0) + 1
    return d

def _modified_precision(hyp: List[str], ref: List[str], n: int) -> float:
    hc = _ngram_counts(hyp, n)
    rc = _ngram_counts(ref, n)
    if not hc:
        return 0.0
    overlap = sum(min(c, rc.get(g, 0)) for g, c in hc.items())
    total = sum(hc.values())
    return (overlap + 1.0) / (total + 1.0)  

def _brevity_penalty(hlen: int, rlen: int) -> float:
    if hlen == 0:
        return 0.0
    if hlen > rlen:
        return 1.0
    return float(np.exp(1.0 - rlen / max(hlen, 1)))

def bleu_1_4(hyp: str, ref: str) -> Tuple[float, float, float, float]:
    h = hyp.strip().split()
    r = ref.strip().split()
    bp = _brevity_penalty(len(h), len(r))
    prec = [ _modified_precision(h, r, n) for n in (1,2,3,4) ]
    def _bleu_k(k: int) -> float:
        if k <= 0: return 0.0
        return float(bp * np.exp(np.mean([np.log(max(prec[i], 1e-12)) for i in range(k)])))
    return _bleu_k(1), _bleu_k(2), _bleu_k(3), _bleu_k(4)

# -------------------- CheXpert --------------------
def run_chexpert_labeler_on_reports(reports: List[str], chexpert_repo_dir: str,
                                    temp_dir: str = "./chexpert_temp") -> pd.DataFrame:
    """
    reports: List of texts
    Returns: A DataFrame containing CHEXPERT_COLS (with values ​​of -1/0/1)
    """
    from CheXpert_labeler.label import label
    from CheXpert_labeler.args import ArgParser

    temp_dir = os.path.abspath(temp_dir)
    os.makedirs(temp_dir, exist_ok=True)
    in_file  = os.path.join(temp_dir, "reports_input.csv")
    out_file = os.path.join(temp_dir, "labels_output.csv")

    with open(in_file, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, quoting=csv.QUOTE_ALL)
        for rpt in reports:
            w.writerow([(rpt or "").strip().replace("\n", " ")])

    old_argv = list(sys.argv)
    prev_cwd = os.getcwd()
    try:
        sys.argv = ["label.py", "--reports_path", in_file, "--output_path", out_file]
        os.chdir(chexpert_repo_dir)
        args = ArgParser().parse_args()
        label(args)
    finally:
        os.chdir(prev_cwd)
        sys.argv = old_argv

    df = pd.read_csv(out_file)
    try:
        os.remove(in_file); os.remove(out_file)
    except Exception:
        pass
    return df

# -------------------- id -> (subject_id, study_id) --------------------
def build_id_map_from_split_json(
    split_json_path: str,
    split_key: str = "test",
    id_key: str = "id",
    subject_key: str = "subject_id",
    study_key: str = "study_id",
) -> Dict[str, Tuple[str, str]]:
    """
    Construct a mapping from the dataset JSON (including the test split) to id -> (subject_id, study_id).
    """
    with open(split_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    items = data.get(split_key, [])
    lut: Dict[str, Tuple[str, str]] = {}
    for obj in items:
        _id = str(obj.get(id_key, ""))
        if not _id:
            continue
        sid = str(obj.get(subject_key, ""))
        tid = str(obj.get(study_key, ""))
        lut[_id] = (sid, tid)
    if not lut:
        raise ValueError(f"{split_json_path} -> split='{split_key}' No results found {id_key}.")
    return lut

# -------------------- Function 1: JSONL -> CheXpert CSV (using JSON to look up id mappings) --------------------
def jsonl_to_chexpert_csv_with_lookup(
    jsonl_path: str,
    split_json_path: str,     # JSON containing the test split, providing id->(subject_id, study_id)
    out_csv: str,
    chexpert_repo_dir: str,
    jsonl_id_field: str = "id",
    jsonl_pred_report_field: str = "pred",
    split_key: str = "test",
    split_id_key: str = "id",
    split_subject_key: str = "subject_id",
    split_study_key: str = "study_id",
) -> None:
    """
    Read JSONL: retrieve the ID and prediction report; use the `test split` function of the split JSON 
    to retrieve the `subject_id` and `study_id`;
    Call CheXpert; output CSV: id, subject_id, study_id, 14-column label, pred_report
    """
    # 1) Build id -> (subject_id, study_id) mapping
    id_map = build_id_map_from_split_json(
        split_json_path, split_key, split_id_key, split_subject_key, split_study_key
    )

    # 2) Read JSONL
    rows = []
    miss = 0
    with open(jsonl_path, "r", encoding="utf-8") as fin:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            _id = str(obj.get(jsonl_id_field, ""))
            pred = obj.get(jsonl_pred_report_field, "")
            sid, tid = id_map.get(_id, ("", ""))
            if not sid and not tid:
                miss += 1
            rows.append({
                "id": _id,
                "subject_id": str(sid),
                "study_id": str(tid),
                "pred_report": pred if isinstance(pred, str) else str(pred),
            })
    if not rows:
        raise ValueError("JSONL is empty or no samples were parsed.")
    if miss > 0:
        print(f"[WARN] {miss} rows of id were not found in split JSON for subject_id/study_id.")

    df = pd.DataFrame(rows)
    # 3) CheXpert
    chex_df = run_chexpert_labeler_on_reports(
        df["pred_report"].fillna("").astype(str).tolist(),
        chexpert_repo_dir=chexpert_repo_dir
    )
    lack = [c for c in CHEXPERT_COLS if c not in chex_df.columns]
    if lack:
        raise ValueError(f"CheXpert output missing columns: {lack}")

    out = pd.concat([df[["id","subject_id","study_id","pred_report"]].reset_index(drop=True),
                     chex_df[CHEXPERT_COLS].reset_index(drop=True)], axis=1)
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_csv, index=False)
    print(f"[OK] The prediction CSV is written out: {Path(out_csv).resolve()}, Total {len(out)} rows")

# -------------------- Helper: Binary Conversion (1/0), Only 1 Counts as Positive --------------------
def _to_binary(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    out = df.copy()
    out[cols] = out[cols].apply(pd.to_numeric, errors="coerce")
    out[cols] = (out[cols] == 1).astype(np.int8)
    return out

def _set_to_str(names: List[str]) -> str:
    return "|".join(sorted(names))

# -------------------- Optional: Normalize Keys to Avoid '123.0' / Whitespace / Inconsistent Prefixes --------------------
def normalize_keys_inplace(df, subject_col="subject_id", study_col="study_id",
                           auto_prefix=False, verbose=False):
    def strip_fix(x):
        s = str(x).strip()
        if s.endswith(".0"): s = s[:-2]
        return s
    before_sid = df[subject_col].astype(str).tolist()
    before_sty = df[study_col].astype(str).tolist()
    df[subject_col] = df[subject_col].map(strip_fix)
    df[study_col]   = df[study_col].map(strip_fix)
    if auto_prefix:
        num = re.compile(r"^\d+$")
        df[subject_col] = df[subject_col].map(lambda s: ("p"+s) if (num.match(s) and not s.startswith("p")) else s)
        df[study_col]   = df[study_col].map(lambda s: ("s"+s) if (num.match(s) and not s.startswith("s")) else s)
    if verbose:
        ch_sid = sum(a!=b for a,b in zip(before_sid, df[subject_col].astype(str).tolist()))
        ch_sty = sum(a!=b for a,b in zip(before_sty, df[study_col].astype(str).tolist()))
        print(f"[Normalize] subject_id changed {ch_sid} rows, study_id changed {ch_sty} rows")

# -------------------- Function 2: Evaluation + Per-Sample Details (Including BLEU) --------------------
def evaluate_pred_vs_gt(
    pred_csv: str,
    gt_csv: str,
    per_class_out_csv: Optional[str] = None,
    per_sample_out_csv: Optional[str] = None,
    subject_col: str = "subject_id",
    study_col: str = "study_id",
    gt_report_col: str = "report",      # If GT does not have a report text, simply provide the name of the non-existent column.
    pred_report_col: str = "pred_report",
    id_col: str = "id",
    key_normalization: str = "basic",   # "none" | "basic" | "auto_prefix"
) -> Tuple[pd.DataFrame, Dict[str,float], Dict[str,float]]:
    pred_df_raw = pd.read_csv(pred_csv, dtype={subject_col:str, study_col:str})
    gt_df_raw   = pd.read_csv(gt_csv,   dtype={subject_col:str, study_col:str})

    need_pred = [subject_col, study_col] + CHEXPERT_COLS
    need_gt   = [subject_col, study_col] + CHEXPERT_COLS

    miss_p = [c for c in need_pred if c not in pred_df_raw.columns]
    miss_g = [c for c in need_gt   if c not in gt_df_raw.columns]
    if miss_p: raise ValueError(f"pred_csv is missing columns:{miss_p}")
    if miss_g: raise ValueError(f"gt_csv is missing columns:{miss_g}")

    has_pred_report = pred_report_col in pred_df_raw.columns
    has_gt_report   = gt_report_col in gt_df_raw.columns
    compute_bleu    = has_pred_report and has_gt_report
    if (per_sample_out_csv is not None) and (not compute_bleu):
        print(f"[WARN] Missing report text column, skip BLEU:pred {has_pred_report}({pred_report_col}), gt {has_gt_report}({gt_report_col})")

    if key_normalization == "basic":
        normalize_keys_inplace(pred_df_raw, subject_col, study_col, auto_prefix=False, verbose=True)
        normalize_keys_inplace(gt_df_raw,   subject_col, study_col, auto_prefix=False, verbose=True)
    elif key_normalization == "auto_prefix":
        normalize_keys_inplace(pred_df_raw, subject_col, study_col, auto_prefix=True,  verbose=True)
        normalize_keys_inplace(gt_df_raw,   subject_col, study_col, auto_prefix=True,  verbose=True)

    keys = [subject_col, study_col]

    # GT deduplication (based on the maximum value after binary analysis)
    gt_bin = _to_binary(gt_df_raw[keys + CHEXPERT_COLS], CHEXPERT_COLS)
    gt_dedup = gt_bin.groupby(keys, as_index=False)[CHEXPERT_COLS].max(numeric_only=True)

    # Prediction binarization (without deduplication)
    pred_bin = _to_binary(pred_df_raw[keys + CHEXPERT_COLS], CHEXPERT_COLS)

    merged = pred_bin.merge(gt_dedup, on=keys, how="inner", suffixes=("_pred","_gt"))
    if merged.empty:
        raise ValueError("The alignment is empty, so it cannot be evaluated. (Please check if the keys are consistent.)")

    # —— Category-by-category Evaluation —— 
    rows = []
    micro_tp = micro_fp = micro_fn = micro_tn = 0
    for cls in CHEXPERT_COLS:
        yp = merged[f"{cls}_pred"].to_numpy(np.int32)
        yt = merged[f"{cls}_gt"].to_numpy(np.int32)
        tp = int(((yp==1)&(yt==1)).sum())
        fp = int(((yp==1)&(yt==0)).sum())
        fn = int(((yp==0)&(yt==1)).sum())
        tn = int(((yp==0)&(yt==0)).sum())

        prec = tp/(tp+fp) if (tp+fp) else 0.0
        rec  = tp/(tp+fn) if (tp+fn) else 0.0
        f1   = (2*prec*rec/(prec+rec)) if (prec+rec) else 0.0
        acc  = (tp+tn)/(tp+fp+fn+tn) if (tp+fp+fn+tn) else 0.0
        rows.append({"class":cls,"TP":tp,"FP":fp,"FN":fn,"TN":tn,
                     "support":int((yt==1).sum()),
                     "precision":prec,"recall":rec,"f1":f1,"accuracy":acc})
        micro_tp += tp; micro_fp += fp; micro_fn += fn; micro_tn += tn

    per_class_df = pd.DataFrame(rows, columns=[
        "class","TP","FP","FN","TN","support","precision","recall","f1","accuracy"
    ])

    micro_p = micro_tp/(micro_tp+micro_fp) if (micro_tp+micro_fp) else 0.0
    micro_r = micro_tp/(micro_tp+micro_fn) if (micro_tp+micro_fn) else 0.0
    micro_f1= (2*micro_p*micro_r/(micro_p+micro_r)) if (micro_p+micro_r) else 0.0
    micro_a = (micro_tp+micro_tn)/(micro_tp+micro_fp+micro_fn+micro_tn) if (micro_tp+micro_fp+micro_fn+micro_tn) else 0.0

    macro_p = float(per_class_df["precision"].mean())
    macro_r = float(per_class_df["recall"].mean())
    macro_f1= float(per_class_df["f1"].mean())
    macro_a = float(per_class_df["accuracy"].mean())

    micro = {"precision":micro_p,"recall":micro_r,"f1":micro_f1,"accuracy":micro_a}
    macro = {"precision":macro_p,"recall":macro_r,"f1":macro_f1,"accuracy":macro_a}

    print("\n[micro]  P={:.4f}  R={:.4f}  F1={:.4f}  Acc={:.4f}".format(micro_p, micro_r, micro_f1, micro_a))
    print("[macro]  P={:.4f}  R={:.4f}  F1={:.4f}  Acc={:.4f}".format(macro_p, macro_r, macro_f1, macro_a))

    if per_class_out_csv:
        Path(per_class_out_csv).parent.mkdir(parents=True, exist_ok=True)
        per_class_df.to_csv(per_class_out_csv, index=False)
        print(f"[OK] The per-class metric is written out: {Path(per_class_out_csv).resolve()}")

    # —— Per-Sample Details (Including BLEU and Chinese/English Column Name Requirements) ——
    if per_sample_out_csv:
        # Prepare Text
        pred_text_map = {}
        if pred_report_col in pred_df_raw.columns:
            pred_text_map = {(str(r[subject_col]), str(r[study_col]), str(r.get(id_col, ""))): str(r[pred_report_col] or "")
                             for _, r in pred_df_raw.iterrows()}
        gt_text_map = {}
        if gt_report_col in gt_df_raw.columns:
            gt_text_map = {(str(r[subject_col]), str(r[study_col])): str(r[gt_report_col] or "")
                           for _, r in gt_df_raw.iterrows()}

        # Use (sid, tid) to find a record with ID for pred.
        id_lookup = {}
        if id_col in pred_df_raw.columns:
            tmp = pred_df_raw[[subject_col, study_col, id_col]].copy()
            tmp = tmp.drop_duplicates(subset=[subject_col, study_col], keep="first")
            for _, rr in tmp.iterrows():
                id_lookup[(str(rr[subject_col]), str(rr[study_col]))] = str(rr[id_col])

        out_rows = []
        for i in range(len(merged)):
            sid = str(merged.loc[i, subject_col])
            tid = str(merged.loc[i, study_col])
            pid = id_lookup.get((sid, tid), "")

            pred_pos = [c for c in CHEXPERT_COLS if merged.loc[i, f"{c}_pred"] == 1]
            gt_pos   = [c for c in CHEXPERT_COLS if merged.loc[i, f"{c}_gt"]   == 1]

            correct = sorted(list(set(pred_pos) & set(gt_pos)))
            wrong   = sorted(list(set(pred_pos) ^ set(gt_pos)))  # Symmetrical difference

            # Text
            pred_text = ""
            key3 = (sid, tid, pid)
            if key3 in pred_text_map:
                pred_text = pred_text_map[key3]
            else:
                # Fallback: any (sid, tid)
                for k, v in pred_text_map.items():
                    if k[0] == sid and k[1] == tid:
                        pred_text = v; break
            gt_text = gt_text_map.get((sid, tid), "")

            # b1, b2, b3, b4 = bleu_1_4(pred_text, gt_text)
            if compute_bleu:
                b1, b2, b3, b4 = bleu_1_4(pred_text, gt_text)
            else:
                b1 = b2 = b3 = b4 = np.nan

            out_rows.append({
                "id": pid,
                "subject_id": sid,
                "study_id": tid,
                "pred cate": _set_to_str(pred_pos),
                "gt cate": _set_to_str(gt_pos),
                "pred report": pred_text,
                "gt report": gt_text,
                "correct pred cate": _set_to_str(correct),
                "wrong pred cate": _set_to_str(wrong),
                "BLEU-1": b1, "BLEU-2": b2, "BLEU-3": b3, "BLEU-4": b4
            })

        out_df = pd.DataFrame(out_rows, columns=[
            "id","subject_id","study_id","pred cate","gt cate",
            "pred report","gt report","correct pred cate","wrong pred cate",
            "BLEU-1","BLEU-2","BLEU-3","BLEU-4"
        ])
        Path(per_sample_out_csv).parent.mkdir(parents=True, exist_ok=True)
        out_df.to_csv(per_sample_out_csv, index=False)
        print(f"[OK] per-sample is saved in：{Path(per_sample_out_csv).resolve()}")

    return per_class_df, micro, macro

def latest_run_dir(exp_root: Path) -> Path:
    """Return the latest numeric run directory under exp_root (ignore tmp/ etc.), using directory mtime."""
    run_dirs = [p for p in exp_root.iterdir() if p.is_dir() and p.name.isdigit()]
    if not run_dirs:
        raise FileNotFoundError(f"No numeric run dirs under: {exp_root}")
    return max(run_dirs, key=lambda p: p.stat().st_mtime)

def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute CheXpert metrics (and BLEU if report texts exist) between prediction JSONL and GT CSV."
    )
    parser.add_argument(
        "--data_root",
        type=str,
        help=(
            "Dataset root path. It should contain 'mimic_test_split_chexpert_categories.csv'. "
            "Example: /mimic_dataset"
        ),
    )
    parser.add_argument(
        "--exp_name",
        type=str,
        default="REVA_stage3_MIMIC",
        help="Experiment folder name under <repo_root>/reva_po/output/.",
    )
    parser.add_argument(
        "--run_ts",
        type=str,
        default=None,
        help=(
            "Optional numeric run timestamp directory name under EXP_ROOT. "
            "If provided, the script will use this run folder instead of auto-picking the latest one."
        ),
    )
    parser.add_argument(
        "--epoch",
        type=str,
        default="0000",
        help="Epoch string used in filenames, e.g., 0000 in test_final_test_epoch0000.jsonl",
    )
    parser.add_argument(
        "--split_json",
        type=str,
        default="mimic_with_categories_sampled_10k.json",
        help="Split JSON filename located under <DATA_DIR>.",
    )
    parser.add_argument(
        "--gt_csv",
        type=str,
        default="mimic_test_split_chexpert_categories.csv",
        help="GT CSV filename located under <DATA_DIR>.",
    )
    return parser.parse_args()

def main():
    args = parse_args()

    # Repo root: this script is assumed to be located directly under REVA/
    REPO_ROOT = Path(__file__).resolve().parent

    # Experiment root containing run subfolders named by timestamps (or similar)
    EXP_ROOT = REPO_ROOT / "reva_po" / "output" / args.exp_name
    if not EXP_ROOT.exists():
        raise FileNotFoundError(f"EXP_ROOT not found: {EXP_ROOT}")

    # - If --run_ts is provided, use it directly (must be numeric).
    # - Otherwise, auto-pick the latest numeric run folder.
    if args.run_ts is not None:
        if not str(args.run_ts).isdigit():
            raise ValueError(f"--run_ts must be numeric, got: {args.run_ts}")
        run_dir = EXP_ROOT / str(args.run_ts)
        if not run_dir.exists() or not run_dir.is_dir():
            raise FileNotFoundError(f"Run directory not found: {run_dir}")
    else:
        run_dir = latest_run_dir(EXP_ROOT)

    # Resolve the prediction JSONL path inside the latest run folder
    jsonl_path = run_dir / "result" / f"test_final_test_epoch{args.epoch}.jsonl"
    if not jsonl_path.exists():
        raise FileNotFoundError(f"Prediction JSONL not found: {jsonl_path}")
    print("latest pred jsonl:", jsonl_path)

    # Create tmp folder if it does not exist
    TMP_DIR = EXP_ROOT / "tmp"
    TMP_DIR.mkdir(parents=True, exist_ok=True)

    # DATA_DIR is derived from the user-provided data_root
    data_root = Path(args.data_root).expanduser().resolve()
    DATA_DIR = data_root
    if not DATA_DIR.exists():
        raise FileNotFoundError(
            f"DATA_DIR not found: {DATA_DIR}\n"
            f"data_root = {data_root}\n"
            f"Expected: <data_root>/mimic_dataset/"
        )

    # Resolve split JSON and GT CSV paths
    split_json_path = DATA_DIR / args.split_json
    if not split_json_path.exists():
        raise FileNotFoundError(f"Split JSON not found: {split_json_path}")

    gt_csv_path = DATA_DIR / args.gt_csv
    print("gt csv:", gt_csv_path)
    if not gt_csv_path.exists():
        raise FileNotFoundError(f"GT CSV not found: {gt_csv_path}")

    # Write outputs under the current run folder to avoid overwriting other runs
    out_csv = run_dir / "result" / f"test_final_test_epoch{args.epoch}_chexpert.csv"

    # Step 1: Convert JSONL predictions to a CSV containing 14 CheXpert categories (and pred_report if available)
    print("Extracting CheXpert categories from the predicted reports. This may take several hours...")
    jsonl_to_chexpert_csv_with_lookup(
        jsonl_path=str(jsonl_path),
        split_json_path=str(split_json_path),
        out_csv=str(out_csv),
        chexpert_repo_dir=str(REPO_ROOT / "CheXpert_labeler"),
    )

    pred_chexpert_csv = out_csv.with_name(out_csv.stem + ".csv")
    if not pred_chexpert_csv.exists():
        raise FileNotFoundError(
            f"Expected CheXpert CSV not found: {pred_chexpert_csv}\n"
            f"Please check jsonl_to_chexpert_csv_with_lookup output naming."
        )

    # Step 2: Evaluate predictions vs GT and write per-class + per-sample outputs
    print("Evaluating predictions against ground truth...")
    per_class_df, micro, macro = evaluate_pred_vs_gt(
        pred_csv=str(pred_chexpert_csv),
        gt_csv=str(gt_csv_path),
        per_class_out_csv=str(TMP_DIR / "per_class_metrics.csv"),
        per_sample_out_csv=str(TMP_DIR / "per_sample_detail.csv"),
        key_normalization="basic",
    )

if __name__ == "__main__":
    main()
