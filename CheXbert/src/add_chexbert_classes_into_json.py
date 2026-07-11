
from __future__ import annotations

import argparse
import json
import os
import time
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import pandas as pd

import utils
from models.bert_labeler import bert_labeler
from constants import CONDITIONS, PAD_IDX
from transformers import BertTokenizer

_POS_CLASS = 1
_UNC_CLASS = 3
_BLANK_CLASS = 0
_NEG_CLASS = 2

def _normalize_split_name(name: str) -> str:
    name = name.lower()
    if name in ("valid", "validation"):
        return "val"
    return name


def _iter_splits(data: Any) -> List[Tuple[str, List[Dict[str, Any]]]]:
    if isinstance(data, dict):
        hit = []
        for k in ["train", "val", "valid", "validation", "test"]:
            if k in data and isinstance(data[k], list):
                hit.append((k, data[k]))
        if hit:
            return hit

        if "splits" in data and isinstance(data["splits"], dict):
            hit = []
            for k in ["train", "val", "valid", "validation", "test"]:
                if k in data["splits"] and isinstance(data["splits"][k], list):
                    hit.append((k, data["splits"][k]))
            if hit:
                return hit

        hit = []
        for k in ["train", "val", "valid", "validation", "test"]:
            if k in data and isinstance(data[k], dict) and isinstance(data[k].get("samples"), list):
                hit.append((k, data[k]["samples"]))
        if hit:
            return hit

    raise ValueError(
        "Unsupported JSON layout. Expected keys like train/val/test containing lists, "
        "or a 'splits' dict containing those lists."
    )


def _get_by_dotted_path(obj: Any, path: str) -> Any:
    if not path:
        return None
    cur = obj
    for part in path.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return None
    return cur


def _clean_text_like_repo(text: Optional[str]) -> str:
    if not isinstance(text, str):
        return ""
    s = text.strip()
    ser = pd.Series([s])
    ser = ser.replace("\n", " ", regex=True)
    ser = ser.replace(r"\s+", " ", regex=True)
    return str(ser.iloc[0]).strip()


def _encode_one_like_repo(text: str, tokenizer: BertTokenizer, max_len: int = 512) -> List[int]:
    if isinstance(text, str) and text.strip():
        enc = tokenizer(
            text,
            add_special_tokens=True,
            truncation=True,
            max_length=max_len,
            return_attention_mask=False,
            return_token_type_ids=False,
        )
        ids = enc["input_ids"]
        if len(ids) == max_len and ids[-1] != tokenizer.sep_token_id:
            ids[-1] = tokenizer.sep_token_id
        return ids
    return [tokenizer.cls_token_id, tokenizer.sep_token_id]


class JsonFindingsDataset(Dataset):
    def __init__(
        self,
        samples: List[Dict[str, Any]],
        findings_key: str,
        tokenizer: BertTokenizer,
        max_len: int = 512,
    ):
        self.samples = samples
        self.findings_key = findings_key
        self.tokenizer = tokenizer
        self.max_len = int(max_len)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int) -> Dict[str, Any]:
        s = self.samples[i]
        findings = _get_by_dotted_path(s, self.findings_key)
        findings = _clean_text_like_repo(findings if isinstance(findings, str) else "")
        ids = _encode_one_like_repo(findings, self.tokenizer, max_len=self.max_len)
        t = torch.tensor(ids, dtype=torch.long)
        return {"imp": t, "len": int(t.numel()), "idx": i}


def collate_fn_no_labels(sample_list: List[Dict[str, Any]]) -> Dict[str, Any]:
    tensor_list = [s["imp"] for s in sample_list]
    batched_imp = torch.nn.utils.rnn.pad_sequence(
        tensor_list, batch_first=True, padding_value=PAD_IDX
    )
    len_list = [s["len"] for s in sample_list]
    idx_list = [s["idx"] for s in sample_list]
    return {"imp": batched_imp, "len": len_list, "idx": idx_list}


def load_model(checkpoint_path: str, device: Union[str, torch.device]) -> nn.Module:
    device = torch.device(device) if not isinstance(device, torch.device) else device
    model = bert_labeler()

    if device.type == "cuda" and torch.cuda.device_count() > 1:
        print("Using", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model).to(device)
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        new_state_dict = OrderedDict()
        for k, v in checkpoint["model_state_dict"].items():
            if k.startswith("module."):
                new_state_dict[k[len("module."):]] = v
            else:
                new_state_dict[k] = v
        model.load_state_dict(new_state_dict)
        model = model.to(device)

    model.eval()
    return model


def _format_bytes(num_bytes: int) -> str:
    x = float(num_bytes)
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if x < 1024.0 or unit == "TB":
            return f"{x:.2f} {unit}"
        x /= 1024.0
    return f"{x:.2f} TB"


def _get_cpu_rss_bytes() -> Optional[int]:
    """
    Best-effort: process resident set size (RSS) in bytes.
    Works on Linux/macOS. Returns None if not available.
    """
    try:
        import resource  # stdlib

        r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # Linux: KB, macOS: bytes. Heuristic: if too small, assume bytes already.
        if r > 10_000_000:  # likely bytes already
            return int(r)
        return int(r * 1024)  # assume KB
    except Exception:
        return None


def _get_cuda_mem(device: torch.device) -> Optional[Tuple[int, int]]:
    """
    Returns (allocated_bytes, reserved_bytes) for the current process on this device.
    """
    if device.type != "cuda":
        return None
    try:
        torch.cuda.synchronize(device)
    except Exception:
        pass
    try:
        alloc = torch.cuda.memory_allocated(device)
        rsv = torch.cuda.memory_reserved(device)
        return int(alloc), int(rsv)
    except Exception:
        return None


@torch.no_grad()
def label_samples_batched(
    samples: List[Dict[str, Any]],
    model: nn.Module,
    tokenizer: BertTokenizer,
    device: Union[str, torch.device],
    findings_key: str,
    batch_size: int,
    num_workers: int,
    max_len: int,
    positive_key: str = "positive",
    uncertain_key: str = "uncertain",
    negative_key: str = "negative",
    blank_key: str = "blank",
    overwrite: bool = True,
    log_every: int = 50,
) -> Dict[str, Any]:
    """
    Mutates `samples` in place (adds positive/uncertain lists).
    Returns stats including time and memory usage.
    """
    device = torch.device(device) if not isinstance(device, torch.device) else device

    dset = JsonFindingsDataset(
        samples=samples,
        findings_key=findings_key,
        tokenizer=tokenizer,
        max_len=max_len,
    )
    loader = DataLoader(
        dset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn_no_labels,
        pin_memory=(device.type == "cuda"),
    )

    # Memory baselines
    cpu_rss0 = _get_cpu_rss_bytes()
    cuda0 = _get_cuda_mem(device)
    if device.type == "cuda":
        try:
            torch.cuda.reset_peak_memory_stats(device)
        except Exception:
            pass

    t0 = time.perf_counter()

    n_samples = 0
    n_pos_total = 0
    n_unc_total = 0
    n_neg_total = 0
    n_blank_total = 0
    print(f"\nBegin FINDINGS labeling. Batch size = {batch_size}")

    for step, batch in enumerate(tqdm(loader, desc="Batches"), start=1):
        x = batch["imp"].to(device)  # (B, L)
        src_len = batch["len"]
        idxs = batch["idx"]

        attn_mask = utils.generate_attention_masks(x, src_len, device)
        out = model(x, attn_mask)

        pred_classes = [o.argmax(dim=1).tolist() for o in out]

        for b, sample_i in enumerate(idxs):
            n_samples += 1
            pos: List[str] = []
            unc: List[str] = []
            neg: List[str] = []
            blank: List[str] = []

            for j, cond in enumerate(CONDITIONS):
                cls = int(pred_classes[j][b])
                if cls == _POS_CLASS:
                    pos.append(cond)
                elif cls == _NEG_CLASS:
                    neg.append(cond)
                elif cls == _BLANK_CLASS:
                    blank.append(cond)
                elif cls == _UNC_CLASS:
                    unc.append(cond)

            n_pos_total += len(pos)
            n_unc_total += len(unc)
            n_neg_total += len(neg)
            n_blank_total += len(blank)

            s = samples[sample_i]

            if overwrite or positive_key not in s:
                s[positive_key] = pos
            else:
                existing = s.get(positive_key, [])
                if not isinstance(existing, list):
                    existing = []
                s[positive_key] = existing + [c for c in pos if c not in existing]

            if overwrite or uncertain_key not in s:
                s[uncertain_key] = unc
            else:
                existing = s.get(uncertain_key, [])
                if not isinstance(existing, list):
                    existing = []
                s[uncertain_key] = existing + [c for c in unc if c not in existing]

            if overwrite or negative_key not in s:
                s[negative_key] = neg
            else:
                existing = s.get(negative_key, [])
                if not isinstance(existing, list):
                    existing = []
                s[negative_key] = existing + [c for c in neg if c not in existing]

            if overwrite or blank_key not in s:
                s[blank_key] = blank
            else:
                existing = s.get(blank_key, [])
                if not isinstance(existing, list):
                    existing = []
                s[blank_key] = existing + [c for c in blank if c not in existing]

        # Optional periodic memory log (lightweight)
        if log_every > 0 and (step % log_every == 0):
            cpu_rss = _get_cpu_rss_bytes()
            cuda_mem = _get_cuda_mem(device)
            msg = f"[mem] step={step}"
            if cpu_rss is not None:
                msg += f" cpu_rss={_format_bytes(cpu_rss)}"
            if cuda_mem is not None:
                msg += f" cuda_alloc={_format_bytes(cuda_mem[0])} cuda_reserved={_format_bytes(cuda_mem[1])}"
            print(msg)

    if device.type == "cuda":
        try:
            torch.cuda.synchronize(device)
        except Exception:
            pass

    dt = time.perf_counter() - t0

    # Memory end / peaks
    cpu_rss1 = _get_cpu_rss_bytes()
    cuda1 = _get_cuda_mem(device)
    cuda_peak_alloc = None
    cuda_peak_reserved = None
    if device.type == "cuda":
        try:
            cuda_peak_alloc = int(torch.cuda.max_memory_allocated(device))
            cuda_peak_reserved = int(torch.cuda.max_memory_reserved(device))
        except Exception:
            pass

    stats: Dict[str, Any] = {
        "n_samples": n_samples,
        "n_positive_total": n_pos_total,
        "n_uncertain_total": n_unc_total,
        "n_negative_total": n_neg_total,
        "n_blank_total": n_blank_total,
        "seconds": float(dt),
        "samples_per_sec": float(n_samples / dt) if dt > 0 else None,
        "cpu_rss_start_bytes": cpu_rss0,
        "cpu_rss_end_bytes": cpu_rss1,
        "cuda_alloc_end_bytes": cuda1[0] if cuda1 is not None else None,
        "cuda_reserved_end_bytes": cuda1[1] if cuda1 is not None else None,
        "cuda_peak_alloc_bytes": cuda_peak_alloc,
        "cuda_peak_reserved_bytes": cuda_peak_reserved,
    }
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Input JSON file.")
    ap.add_argument("--output", required=True, help="Output JSON file.")
    ap.add_argument("--checkpoint", required=True, help="Path to chexbert.pth.")
    ap.add_argument("--device", default=None, help="cpu, cuda:0, etc. Default: auto.")
    ap.add_argument("--batch_size", type=int, default=8, help="Batch size.")
    ap.add_argument("--num_workers", type=int, default=0, help="DataLoader num_workers.")
    ap.add_argument("--max_len", type=int, default=512, help="Tokenizer max length (default 512).")
    ap.add_argument(
        "--findings_key",
        default="findings",
        help='Findings field path in each sample. Supports dotted path, e.g. "current.findings".',
    )
    ap.add_argument("--positive_key", default="positive", help="Key to store positive condition names.")
    ap.add_argument("--uncertain_key", default="uncertain", help="Key to store uncertain condition names.")
    ap.add_argument("--negative_key", default="negative", help="Key to store negative condition names.")
    ap.add_argument("--blank_key", default="blank", help="Key to store blank condition names.")
    ap.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing positive/uncertain. If not set, merge into existing lists.",
    )
    ap.add_argument(
        "--log_every",
        type=int,
        default=50,
        help="Print memory usage every N batches (0 disables).",
    )
    args = ap.parse_args()

    if args.device is None:
        args.device = "cuda:0" if torch.cuda.is_available() else "cpu"

    t_all0 = time.perf_counter()

    with open(args.input, "r", encoding="utf-8") as f:
        data = json.load(f)

    splits = _iter_splits(data)
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = load_model(args.checkpoint, args.device)

    all_stats: Dict[str, Any] = {"splits": {}}

    for split_key, samples in splits:
        split_norm = _normalize_split_name(split_key)
        print(f"\n=== Split: {split_norm} (n={len(samples)}) ===")
        stats = label_samples_batched(
            samples=samples,
            model=model,
            tokenizer=tokenizer,
            device=args.device,
            findings_key=args.findings_key,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            max_len=args.max_len,
            positive_key=args.positive_key,
            uncertain_key=args.uncertain_key,
            negative_key=args.negative_key,
            blank_key=args.blank_key,
            overwrite=args.overwrite,
            log_every=args.log_every,
        )
        all_stats["splits"][split_norm] = stats

        # Pretty print split stats
        print(
            f"[{split_norm}] time={stats['seconds']:.2f}s "
            f"rate={stats['samples_per_sec']:.2f} samples/s "
            f"pos={stats['n_positive_total']} unc={stats['n_uncertain_total']} "
            f"neg={stats['n_negative_total']} blank={stats['n_blank_total']}"
        )
        if stats["cpu_rss_start_bytes"] is not None and stats["cpu_rss_end_bytes"] is not None:
            print(
                f"[{split_norm}] cpu_rss start={_format_bytes(stats['cpu_rss_start_bytes'])} "
                f"end={_format_bytes(stats['cpu_rss_end_bytes'])}"
            )
        if stats["cuda_peak_alloc_bytes"] is not None:
            print(
                f"[{split_norm}] cuda_peak alloc={_format_bytes(stats['cuda_peak_alloc_bytes'])} "
                f"reserved={_format_bytes(stats['cuda_peak_reserved_bytes'])}"
            )

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    t_all = time.perf_counter() - t_all0
    all_stats["total_seconds"] = float(t_all)

    print("\nDone. Wrote:", args.output)
    print(f"Total time: {t_all:.2f}s")


if __name__ == "__main__":
    main()




# python src/add_chexbert_classes_into_json.py \
#     --input /mnt/ssd4tb/datasets/research/mimic-cxr-jpg/mimic_has_bbox.json \
#     --output /mnt/ssd4tb/datasets/research/mimic-cxr-jpg/mimic_has_bbox_chexbert.json \
#     --checkpoint src/chexbert.pth \
#     --device cuda:0 \
#     --batch_size 128 \
#     --num_workers 0 \
#     --findings_key findings