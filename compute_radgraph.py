from __future__ import annotations

import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import pandas as pd
from tqdm import tqdm

from radgraph import F1RadGraph  # type: ignore


# -------------------- JSONL Reader --------------------
def read_pred_gt_from_jsonl(
    jsonl_path: str,
    id_field: str = "id",
    pred_field: str = "pred",
    gt_field: str = "gt",
    strict: bool = False,
) -> Tuple[List[str], List[str], List[str]]:
    """
    Read a JSONL file where each line is a dict containing {id, pred, gt}.

    Returns:
        ids:  List[str]
        hyps: List[str]
        refs: List[str]
    """
    ids: List[str] = []
    hyps: List[str] = []
    refs: List[str] = []

    bad = 0
    miss = 0

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for ln, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                bad += 1
                if strict:
                    raise ValueError(f"Invalid JSON at line {ln}: {line[:200]}")
                continue

            _id = obj.get(id_field, None)
            pred = obj.get(pred_field, None)
            gt = obj.get(gt_field, None)

            if _id is None or pred is None or gt is None:
                miss += 1
                if strict:
                    raise ValueError(
                        f"Missing field(s) at line {ln}. "
                        f"Need {id_field},{pred_field},{gt_field}. Got keys={list(obj.keys())}"
                    )
                continue

            ids.append(str(_id))
            hyps.append("" if pred is None else str(pred))
            refs.append("" if gt is None else str(gt))

    if len(ids) == 0:
        raise ValueError(f"No valid samples parsed from JSONL: {jsonl_path}")

    if bad > 0:
        print(f"[WARN] {bad} lines are not valid JSON and were skipped.")
    if miss > 0:
        print(f"[WARN] {miss} lines miss required fields and were skipped.")

    return ids, hyps, refs


# -------------------- RadGraph F1 from JSONL --------------------
def compute_radgraph_f1_from_jsonl(
    jsonl_path: str,
    id_field: str = "id",
    pred_field: str = "pred",
    gt_field: str = "gt",
    reward_level: str = "all",
    model_type: str = "radgraph-xl",
    show_progress: bool = True,
    batch_size: int = 32,
    strict_jsonl: bool = False,
) -> Dict[str, object]:
    """
    Compute RadGraph F1 scores from a JSONL file where each line contains: id, pred, gt.

    Returns a dict with:
      - num_samples
      - rg_e_mean, rg_er_mean
      - per_sample: DataFrame indexed by id with columns [rg_e, rg_er]
    """
    ids, hyps_all, refs_all = read_pred_gt_from_jsonl(
        jsonl_path=jsonl_path,
        id_field=id_field,
        pred_field=pred_field,
        gt_field=gt_field,
        strict=strict_jsonl,
    )

    num_samples = int(len(ids))
    scorer = F1RadGraph(reward_level=reward_level, model_type=model_type)

    per_rows = []
    rg_e_sum = 0.0
    rg_er_sum = 0.0
    rg_bar_er_sum = 0.0

    idx_iter = range(0, num_samples, batch_size)
    if show_progress:
        idx_iter = tqdm(
            list(idx_iter),
            desc="RadGraph scoring",
            total=(num_samples + batch_size - 1) // batch_size,
        )

    for start in idx_iter:
        end = min(start + batch_size, num_samples)
        hyps = hyps_all[start:end]
        refs = refs_all[start:end]

        mean_reward, reward_list, _, _ = scorer(hyps=hyps, refs=refs)

        # reward_list is 3 lists: [rg_e_list, rg_er_list, rg_bar_er_list]
        rg_e_list, rg_er_list, rg_bar_er_list = reward_list
        if not (len(rg_e_list) == len(rg_er_list) == len(rg_bar_er_list) == (end - start)):
            raise ValueError(
                f"Unexpected reward_list lengths: "
                f"{len(rg_e_list)}, {len(rg_er_list)}, {len(rg_bar_er_list)} vs batch {(end-start)}"
            )

        for j in range(end - start):
            sid = ids[start + j]

            rg_e_f = float(rg_e_list[j])
            rg_er_f = float(rg_er_list[j])
            rg_bar_er_f = float(rg_bar_er_list[j])

            rg_e_sum += rg_e_f
            rg_er_sum += rg_er_f
            rg_bar_er_sum += rg_bar_er_f

            per_rows.append(
                {
                    "id": sid,
                    "rg_e": rg_e_f,
                    "rg_er": rg_er_f,
                    "rg_bar_er": rg_bar_er_f,
                }
            )

    rg_e_mean = rg_e_sum / num_samples if num_samples > 0 else 0.0
    rg_er_mean = rg_er_sum / num_samples if num_samples > 0 else 0.0
    rg_bar_er_mean = rg_bar_er_sum / num_samples if num_samples > 0 else 0.0

    per_sample = pd.DataFrame(per_rows).set_index("id")

    return {
        "num_samples": num_samples,
        "rg_e_mean": rg_e_mean,
        "rg_er_mean": rg_er_mean,
        "rg_bar_er_mean": rg_bar_er_mean,
        "per_sample": per_sample,
    }


# -------------------- CLI (CE-like style) --------------------
def latest_run_dir(exp_root: Path) -> Path:
    run_dirs = [p for p in exp_root.iterdir() if p.is_dir() and p.name.isdigit()]
    if not run_dirs:
        raise FileNotFoundError(f"No numeric run dirs under: {exp_root}")
    return max(run_dirs, key=lambda p: p.stat().st_mtime)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute RadGraph F1 from prediction JSONL containing {id, pred, gt}."
    )
    parser.add_argument(
        "--repo_root",
        type=str,
        default=None,
        help="Optional repo root. If not set, use this script's parent directory.",
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
        help="Optional numeric run timestamp directory name under EXP_ROOT. If not set, auto-pick latest.",
    )
    parser.add_argument(
        "--epoch",
        type=str,
        default="0000",
        help="Epoch string used in filenames, e.g., 0000 in test_final_test_epoch0000.jsonl",
    )
    parser.add_argument("--id_field", type=str, default="id")
    parser.add_argument("--pred_field", type=str, default="pred")
    parser.add_argument("--gt_field", type=str, default="gt")

    parser.add_argument("--reward_level", type=str, default="all")
    parser.add_argument("--model_type", type=str, default="radgraph-xl")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--no_progress", action="store_true")

    parser.add_argument(
        "--out_csv",
        type=str,
        default=None,
        help="Optional path to save per-sample RadGraph scores as CSV.",
    )
    parser.add_argument(
        "--strict_jsonl",
        action="store_true",
        help="If set, invalid json/missing fields will raise error instead of skipping.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Repo root
    if args.repo_root is None:
        REPO_ROOT = Path(__file__).resolve().parent
    else:
        REPO_ROOT = Path(args.repo_root).expanduser().resolve()

    # Experiment root
    EXP_ROOT = REPO_ROOT / "reva_po" / "output" / args.exp_name
    if not EXP_ROOT.exists():
        raise FileNotFoundError(f"EXP_ROOT not found: {EXP_ROOT}")

    # pick run dir
    if args.run_ts is not None:
        if not str(args.run_ts).isdigit():
            raise ValueError(f"--run_ts must be numeric, got: {args.run_ts}")
        run_dir = EXP_ROOT / str(args.run_ts)
        if not run_dir.exists() or not run_dir.is_dir():
            raise FileNotFoundError(f"Run directory not found: {run_dir}")
    else:
        run_dir = latest_run_dir(EXP_ROOT)

    # jsonl path
    jsonl_path = run_dir / "result" / f"test_final_test_epoch{args.epoch}.jsonl"
    if not jsonl_path.exists():
        raise FileNotFoundError(f"Prediction JSONL not found: {jsonl_path}")
    print("pred jsonl:", jsonl_path)

    # compute
    metrics = compute_radgraph_f1_from_jsonl(
        jsonl_path=str(jsonl_path),
        id_field=args.id_field,
        pred_field=args.pred_field,
        gt_field=args.gt_field,
        reward_level=args.reward_level,
        model_type=args.model_type,
        show_progress=(not args.no_progress),
        batch_size=args.batch_size,
        strict_jsonl=args.strict_jsonl,
    )

    print("num_samples:", metrics["num_samples"])
    print("RG_E mean:", metrics["rg_e_mean"])
    print("RG_ER mean:", metrics["rg_er_mean"])
    print("RG_bar_ER mean:", metrics["rg_bar_er_mean"])
    print("We use RG_ER as the main metric for evaluation, which considers both edge and relation correctness."
          "RG_E is edge-only F1, and RG_bar_ER is a weighted variant of RG_ER")
    per_sample: pd.DataFrame = metrics["per_sample"]  # type: ignore
    print(per_sample.sort_values("rg_er", ascending=False).head(10))

    # optional save
    if args.out_csv is not None:
        out_csv = Path(args.out_csv).expanduser().resolve()
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        per_sample.reset_index().to_csv(out_csv, index=False)
        print(f"[OK] per-sample RadGraph is saved in: {out_csv}")


if __name__ == "__main__":
    main()
