"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE_Lavis file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""
import logging
import torch
from aprpo.common.registry import registry
from aprpo.tasks.base_task import BaseTask
from aprpo.common.logger import MetricLogger, SmoothedValue
from aprpo.datasets.data_utils import prepare_sample
from aprpo.common.dist_utils import is_dist_avail_and_initialized, main_process, is_main_process
import torch.distributed as dist
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import time 
import numpy as np
import re
import math
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score,precision_score, recall_score
import torch.nn.functional as F
from contextlib import contextmanager
import os
from tqdm import tqdm
from pathlib import Path
import json, gzip

@registry.register_task("chestxray_multilabel_cls")
class ChestXrayMultiLabelClassification(BaseTask):
    def __init__(self):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def train_step(self, model, samples):
        loss = model(samples)["loss"]
        return loss
    
    def _train_inner_loop(
        self,
        epoch,
        iters_per_epoch,
        model,
        data_loader,
        optimizer,
        scaler=None,
        start_iters=None,
        log_freq=50,
        cuda_enabled=False,
        accum_grad_iters=1,
        use_zero_optimizer=False,
        step_log_fn=None,
    ):
        """
        Training inner loop for multi-label classification task
        """
        use_amp = scaler is not None
        if not hasattr(data_loader, "__next__"):
            data_loader = iter(data_loader)

        metric_logger = MetricLogger(delimiter="  ")
        metric_logger.add_meter("lr", SmoothedValue(window_size=1, fmt="{value:.6f}"))
        metric_logger.add_meter("lr_2", SmoothedValue(window_size=1, fmt="{value:.6f}"))
        metric_logger.add_meter("lr_3", SmoothedValue(window_size=1, fmt="{value:.6f}"))
        metric_logger.add_meter("loss", SmoothedValue(window_size=1, fmt="{value:.4f}"))
            
        logging.info(
            f"Start training epoch {epoch}, {iters_per_epoch} iters per inner epoch."
        )
        header = f"Train: data epoch: [{epoch}]"
        if start_iters is None:
            inner_epoch = epoch
        else:
            inner_epoch = start_iters // iters_per_epoch
            header = header + f"; inner epoch [{inner_epoch}]"

        for i in metric_logger.log_every(range(iters_per_epoch), log_freq, header):
            if i >= iters_per_epoch:
                break

            samples = next(data_loader)
            samples = prepare_sample(samples, cuda_enabled=cuda_enabled)
            samples.update({
                "epoch": inner_epoch,
                "num_iters_per_epoch": iters_per_epoch,
                "iters": i,
            })

            # Forward pass
            model_output = model(samples)
            loss = model_output["loss"]

            # Backpropagation and optimization
            if loss.item() != 0.0:
                if use_zero_optimizer:
                    model.backward(loss)
                else:
                    if use_amp:
                        scaler.scale(loss).backward()
                    else:
                        loss.backward()

                if (i + 1) % accum_grad_iters == 0:
                    if use_zero_optimizer:
                        model.step()
                    else:
                        if use_amp:
                            scaler.step(optimizer)
                            scaler.update()
                        else:
                            optimizer.step()
                        optimizer.zero_grad()

            # Record loss and learning rate
            loss_value = loss.item()
            if not math.isnan(loss_value) and not math.isinf(loss_value):
                metric_logger.update(loss=loss_value)
            else:
                print(f"[rank {dist.get_rank() if dist.is_initialized() else 0}] Step {i} loss is nan/inf, skip update.")
            metric_logger.update(lr=optimizer.param_groups[-1]["lr"])
            metric_logger.update(lr_2=optimizer.param_groups[-2]["lr"])
            metric_logger.update(lr_3=optimizer.param_groups[-3]["lr"])
        metric_logger.synchronize_between_processes()
        logging.info("Averaged stats: " + str(metric_logger.global_avg()))

        return {
            k: "{:.3f}".format(meter.global_avg)
            for k, meter in metric_logger.meters.items()
        }
      
    def evaluation(self, model, data_loader, cuda_enabled=True):
        categories = [
            "No Finding", "Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Opacity",
            "Lung Lesion", "Edema", "Consolidation", "Pneumonia", "Atelectasis", "Pneumothorax",
            "Pleural Effusion", "Pleural Other", "Fracture", "Support Devices"
        ]
        metric_logger = MetricLogger(delimiter="  ")
        header = "Validation:"
        print_freq = 10

        model.eval()
        all_probs = []
        all_preds = []
        all_gts = []
        all_ids = []

        for batch_idx, samples in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
            samples = prepare_sample(samples, cuda_enabled=cuda_enabled)
            # Inference, return a dict containing probs, pred_labels, gt_labels
            output = model.generate(samples)
            all_probs.append(output["probs"].cpu().numpy())
            all_preds.append(output["pred_labels"].cpu().numpy())
            all_gts.append(output["gt_labels"].cpu().numpy())
            if "id" in samples:
                all_ids.extend(samples["id"])

        # Collect outputs from all processes
        if is_dist_avail_and_initialized():
            world_size = dist.get_world_size()
            all_probs_list = [None] * world_size
            all_preds_list = [None] * world_size
            all_gts_list = [None] * world_size
            all_ids_list = [None] * world_size
            dist.all_gather_object(all_probs_list, all_probs)
            dist.all_gather_object(all_preds_list, all_preds)
            dist.all_gather_object(all_gts_list, all_gts)
            dist.all_gather_object(all_ids_list, all_ids)
            if is_main_process():
                all_probs = np.concatenate(sum(all_probs_list, []), axis=0)
                all_preds = np.concatenate(sum(all_preds_list, []), axis=0)
                all_gts = np.concatenate(sum(all_gts_list, []), axis=0)
                all_ids = sum(all_ids_list, [])
        else:
            all_probs = np.concatenate(all_probs, axis=0)
            all_preds = np.concatenate(all_preds, axis=0)
            all_gts = np.concatenate(all_gts, axis=0)

        if is_main_process():
            metrics = self.compute_metrics(all_probs, all_preds, all_gts)
            metrics["total_samples"] = all_probs.shape[0]
            self.print_metrics_per_class(metrics, categories)
            print("Eval metrics:", metrics)
        else:
            metrics = None
        
        if is_dist_avail_and_initialized():
            dist.barrier()  # Ensure all processes are synchronized

        return metrics

    def after_evaluation(self, val_result, split_name, epoch):
        """
        Process evaluation results and return logging statistics
        Args:
            val_result (dict): Dictionary containing evaluation metrics 
            split_name (str): Name of dataset split (e.g., 'val', 'test')
            epoch (int/str): Current epoch or 'best'
        Returns:
            log_stats (dict): Processed logging statistics
        """
        # Initialize logging stats with basic info
        log_stats = {
            "epoch": epoch,
            "split_name": split_name,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }

        # Add metrics with split name prefix for all numeric values
        for k, v in val_result.items():
            if isinstance(v, (int, float)):
                log_stats[f"{split_name}/{k}"] = v
        
        # Check and copy agg_metrics (main metric for model selection)
        assert "agg_metrics" in val_result, "agg_metrics not found in evaluation results"
        log_stats["agg_metrics"] = val_result["agg_metrics"]
        
        # Add evaluation time statistics if available
        if hasattr(self, 'eval_start_time'):
            eval_time = time.time() - self.eval_start_time
            log_stats[f"{split_name}/eval_time"] = eval_time
        
        # Print main evaluation metrics
        print(f"\n{split_name} Evaluation Results - Epoch {epoch}:")
        print(f"agg_metrics: {log_stats['agg_metrics']:.4f}")

        return log_stats

    @main_process
    def compute_metrics(self, probs, preds, gts):
        """
        probs, preds, gts: NumPy array, shape = [N, C]
        - Micro in multi-label scenarios: Use sklearn to perform global aggregation on the sample x category dimension
        - Macro: Calculate the mean independently for each category (ignoring NaN)
        """
        results = {}
        num_classes = gts.shape[1]

        auc_list = []
        acc_list = []
        f1_list = []
        prec_list = []
        rec_list = []

        for i in range(num_classes):
            # AUC (by category, using probability)
            try:
                auc = roc_auc_score(gts[:, i], probs[:, i])
            except Exception:
                auc = np.nan
            auc_list.append(auc)

            # Use final binary predictions to compute ACC / P / R / F1 (by category)
            try:
                acc = accuracy_score(gts[:, i], preds[:, i])
            except Exception:
                acc = np.nan
            acc_list.append(acc)

            try:
                precision = precision_score(gts[:, i], preds[:, i], zero_division=0)
            except Exception:
                precision = np.nan
            prec_list.append(precision)

            try:
                recall = recall_score(gts[:, i], preds[:, i], zero_division=0)
            except Exception:
                recall = np.nan
            rec_list.append(recall)

            try:
                f1 = f1_score(gts[:, i], preds[:, i], zero_division=0)
            except Exception:
                f1 = np.nan
            f1_list.append(f1)

        # ---- per-class list ----
        results["auc_each_class"] = auc_list
        results["acc_each_class"] = acc_list
        results["precision_each_class"] = prec_list
        results["recall_each_class"] = rec_list
        results["f1_each_class"] = f1_list

        # ---- macro (class average) ----
        results["auc_avg"] = np.nanmean(auc_list)
        results["acc_avg"] = np.nanmean(acc_list)
        results["precision_macro"] = np.nanmean(prec_list)
        results["recall_macro"] = np.nanmean(rec_list)
        results["f1_macro"] = np.nanmean(f1_list)
        results["f1_avg"] = results["f1_macro"]
        results["acc_avg"] = results["acc_avg"]

        # ---- micro (global aggregation) ----
        try:
            results["precision_micro"] = precision_score(gts, preds, average="micro", zero_division=0)
            results["recall_micro"] = recall_score(gts, preds, average="micro", zero_division=0)
            results["f1_micro"] = f1_score(gts, preds, average="micro", zero_division=0)
        except Exception:
            results["precision_micro"] = np.nan
            results["recall_micro"] = np.nan
            results["f1_micro"] = np.nan

        results["agg_metrics"] = results["auc_avg"]

        return results

    def valid_step(self, model, samples):
        with torch.inference_mode():
            output = model.generate(samples)
        return output
    
    def print_metrics_per_class(self, metrics, categories):
        print("Per-class metrics:")
        print("{:32s}  {:>8s}  {:>8s}  {:>8s}  {:>8s}  {:>8s}".format(
            "Category", "AUC", "ACC", "Prec", "Rec", "F1"
        ))

        auc = metrics["auc_each_class"]
        acc = metrics["acc_each_class"]
        prec = metrics["precision_each_class"]
        rec = metrics["recall_each_class"]
        f1 = metrics["f1_each_class"]

        for i, name in enumerate(categories):
            auc_str  = "{:.3f}".format(auc[i])  if auc[i]  is not None and not np.isnan(auc[i])  else "nan"
            acc_str  = "{:.3f}".format(acc[i])  if acc[i]  is not None and not np.isnan(acc[i])  else "nan"
            prec_str = "{:.3f}".format(prec[i]) if prec[i] is not None and not np.isnan(prec[i]) else "nan"
            rec_str  = "{:.3f}".format(rec[i])  if rec[i]  is not None and not np.isnan(rec[i])  else "nan"
            f1_str   = "{:.3f}".format(f1[i])   if f1[i]   is not None and not np.isnan(f1[i])   else "nan"
            print("{:32s}  {:>8s}  {:>8s}  {:>8s}  {:>8s}  {:>8s}".format(
                name, auc_str, acc_str, prec_str, rec_str, f1_str
            ))

        print("-" * 72)
        # macro (class average)
        print("{:32s}  {:>8s}  {:>8s}  {:>8.3f}  {:>8.3f}  {:>8.3f}".format(
            "Macro Avg",
            "{:.3f}".format(metrics["auc_avg"]),
            "{:.3f}".format(metrics["acc_avg"]),
            metrics["precision_macro"] if isinstance(metrics["precision_macro"], float) else float(metrics["precision_macro"]),
            metrics["recall_macro"] if isinstance(metrics["recall_macro"], float) else float(metrics["recall_macro"]),
            metrics["f1_macro"] if isinstance(metrics["f1_macro"], float) else float(metrics["f1_macro"]),
        ))

        # micro (global aggregation)
        pm = metrics["precision_micro"]; rm = metrics["recall_micro"]; fm = metrics["f1_micro"]
        pm_str = "{:.3f}".format(pm) if pm is not None and not np.isnan(pm) else "nan"
        rm_str = "{:.3f}".format(rm) if rm is not None and not np.isnan(rm) else "nan"
        fm_str = "{:.3f}".format(fm) if fm is not None and not np.isnan(fm) else "nan"
        print("{:32s}  {:>8s}  {:>8s}  {:>8s}  {:>8s}  {:>8s}".format(
            "Micro Avg", "-", "-", pm_str, rm_str, fm_str
        ))

        print("Total samples:", metrics.get("total_samples", "-"))

    def search_best_thresholds_with_metrics(self, model, data_loader, cuda_enabled=True, search_space=None):
        categories = [
            "No Finding", "Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Opacity",
            "Lung Lesion", "Edema", "Consolidation", "Pneumonia", "Atelectasis", "Pneumothorax",
            "Pleural Effusion", "Pleural Other", "Fracture", "Support Devices"
        ]

        model.eval()
        all_probs = []
        all_gts = []

        for samples in tqdm(data_loader, desc="Infer val set"):
            samples = prepare_sample(samples, cuda_enabled=cuda_enabled)
            with torch.inference_mode():
                output = model.generate(samples)
                all_probs.append(output["probs"].cpu().numpy())
                all_gts.append(output["gt_labels"].cpu().numpy())

        all_probs = np.concatenate(all_probs, axis=0)
        all_gts = np.concatenate(all_gts, axis=0)

        print("all_probs shape:", all_probs.shape)
        print("all_gts shape:", all_gts.shape)

        if search_space is None:
            search_space = np.arange(0.0, 1.01, 0.0001)

        num_classes = all_probs.shape[1]
        best_thresholds = []
        best_f1s = []
        aucs = []
        accs = []
        precisions = []
        recalls = []

        print("\n{:32s}  {:>10s}  {:>8s}  {:>8s}  {:>8s}  {:>8s}  {:>8s}".format(
            "Category", "Best_thr", "Best_F1", "AUC", "ACC", "Prec", "Rec"
        ))

        for i in range(num_classes):
            best_thr = 0.5
            best_f1 = 0.0
            for thr in search_space:
                preds = (all_probs[:, i] > thr).astype(int)
                try:
                    f1 = f1_score(all_gts[:, i], preds)
                except Exception:
                    f1 = 0.0
                if f1 > best_f1:
                    best_f1 = f1
                    best_thr = thr
            best_thresholds.append(best_thr)
            best_f1s.append(best_f1)

            final_preds = (all_probs[:, i] > best_thr).astype(int)
            # AUC
            try:
                auc = roc_auc_score(all_gts[:, i], all_probs[:, i])
            except Exception:
                auc = float("nan")
            aucs.append(auc)
            # ACC
            try:
                acc = accuracy_score(all_gts[:, i], final_preds)
            except Exception:
                acc = float("nan")
            accs.append(acc)
            # Precision
            try:
                precision = precision_score(all_gts[:, i], final_preds, zero_division=0)
            except Exception:
                precision = float("nan")
            precisions.append(precision)
            # Recall
            try:
                recall = recall_score(all_gts[:, i], final_preds, zero_division=0)
            except Exception:
                recall = float("nan")
            recalls.append(recall)

            auc_str = "{:.3f}".format(auc) if not np.isnan(auc) else "nan"
            acc_str = "{:.3f}".format(acc) if not np.isnan(acc) else "nan"
            precision_str = "{:.3f}".format(precision) if not np.isnan(precision) else "nan"
            recall_str = "{:.3f}".format(recall) if not np.isnan(recall) else "nan"
            print("{:32s}  {:10.4f}  {:8.4f}  {:8s}  {:8s}  {:8s}  {:8s}".format(
                categories[i], best_thr, best_f1, auc_str, acc_str, precision_str, recall_str
            ))

        mean_f1 = np.mean(best_f1s)
        mean_auc = np.nanmean(aucs)
        mean_acc = np.mean(accs)
        mean_precision = np.mean(precisions)
        mean_recall = np.mean(recalls)
        print("-" * 90)
        print("{:32s}  {:10s}  {:8.4f}  {:8.3f}  {:8.3f}  {:8.3f}  {:8.3f}".format(
            "Average", "", mean_f1, mean_auc, mean_acc, mean_precision, mean_recall
        ))

        return best_thresholds, best_f1s, aucs, accs, precisions, recalls

@registry.register_task("xray_report_generation")
class XrayReportGenerate(BaseTask):
    def __init__(self):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.ref_state_dict = None
        self._last_eval_payload = None
        self._last_eval_metrics = None
    
    def get_last_eval_payload(self):
        """After the evaluation is completed, the main process can read the complete sample-level output of the most recent evaluation."""
        return getattr(self, "_last_eval_payload", None)

    def train_step(self, model, samples):
        loss = model(samples)["loss"]
        return loss

    def get_lora_state_dict(self, model):
        # Extract only LoRA parameters (deepcopy for safety)
        return {k: v.detach().clone().cpu() for k, v in model.named_parameters() if "lora" in k or "merger" in k or v.requires_grad}

    def get_ref_state_dict(self, model):
        # Extract only merger parameters (deepcopy for safety)
        return {k: v.detach().clone().cpu() for k, v in model.named_parameters() if "lora" in k or "merger" in k or v.requires_grad}
        # return {k: v.detach().clone().cpu() for k, v in model.named_parameters() if "merger" in k and v.requires_grad}

    def get_lora_state_dict_cuda(self, model, device=None):
        if device is None:
            device = next(model.parameters()).device
        return {k: v.detach().clone().to(device) 
                for k, v in model.named_parameters() 
                if "lora" in k or "merger" in k or v.requires_grad}

    @contextmanager
    def use_lora_params(self, model, lora_state_dict, ref=False):
        """
        Temporarily swap LoRA adapter parameters for the model.
        peft_model: current model
        lora_state_dict: old model
        """
        current_state = {}
        if ref:
            print("== All parameter names of the current model ==")
            for name, _ in model.named_parameters():
                print(name)
        if ref:
            print("== The following parameter names will be swapped ==")
        # Replace LoRA params with given ones
        for name, param in model.named_parameters():
            if name in lora_state_dict:
                if ref:
                    print("Swapped: ", name)
                current_state[name] = param.data.detach().clone()
                param.data.copy_(lora_state_dict[name].to(param.device))
        try:
            yield
        finally:
            # Restore original params
            for name, param in model.named_parameters():
                if name in current_state:
                    param.data.copy_(current_state[name])
    
    @contextmanager
    def use_lora_params_cuda(self, model, lora_state_dict):
        """
        Temporarily swap LoRA adapter parameters for the model.
        peft_model: current model
        lora_state_dict: old model
        """
        current_state = {}
        # Replace LoRA params with given ones
        for name, param in model.named_parameters():
            if name in lora_state_dict:
                current_state[name] = param.data.detach().clone()
                param.data.copy_(lora_state_dict[name])
        try:
            yield
        finally:
            # Restore original params
            for name, param in model.named_parameters():
                if name in current_state:
                    param.data.copy_(current_state[name])

    def bleu4_reward_fn(self, preds, gts, beta=1.0, n=4):
        """
        """
        if isinstance(gts[0], str):
            gts = [[gt] for gt in gts]
        rewards = []
        bleu_score_list=[]
        smoothie = SmoothingFunction().method1
        for pred, refs in zip(preds, gts):
            if isinstance(refs[0], str):
                refs = [r.split() for r in refs]
            pred_tokens = pred.split()
            bleu_score = sentence_bleu(
                refs,
                pred_tokens,
                weights=(0.25, 0.25, 0.25, 0.25),
                smoothing_function=smoothie
            )
            final_score = bleu_score 
            final_score = max(final_score, 0.0) # The minimum is 0
            rewards.append(final_score)
            bleu_score_list.append(bleu_score)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        return rewards

    def sample_candidates(
        self,
        data_loader,
        model,
        group_size,
        reward_fn,
        old_lora_state_dict,
        cuda_enabled=True,
    ):
        """
        Sample G candidate generations per input example using the "old policy",
        then compute rewards for each candidate.

        Args:
            data_loader:
                An iterator that yields a batch of raw samples (one batch = B inputs).
            model:
                The policy model. We will temporarily load LoRA weights from old_lora_state_dict
                so generation is done by the old policy.
            group_size (int):
                Number of candidates per input example (G). Total generated sequences = N = B * G.
            reward_fn:
                Callable reward function. Signature should be:
                    rewards = reward_fn(pred_texts, gt_texts)
                where pred_texts and gt_texts are lists of length N.
            old_lora_state_dict:
                LoRA snapshot used as the old policy for sampling.
            cuda_enabled (bool):
                Whether to move batch tensors to GPU via prepare_sample.

        Returns:
            samples:
                prepared batch (moved to GPU if cuda_enabled)
            outputs:
                Dict returned by model.generate, expected keys:
                - "predicted_reports": list[str] length N
                - "gt_reports": list[str] length B (before repeat)
                - "output_ids": Tensor [N, T]
            rewards:
                Tensor [N] on the same device as output_ids.
        """

        # Fetch one batch (B examples) from the dataloader iterator.
        # Note: data_loader here is assumed to be an iterator, not the DataLoader itself.
        samples = next(data_loader)
        samples = prepare_sample(samples, cuda_enabled=cuda_enabled)

        # Switch the model to the old policy (pi_old) using LoRA snapshot.
        # We generate rollouts from pi_old for PPO-style importance sampling later.
        with self.use_lora_params(model, old_lora_state_dict):
            model.eval()
            with torch.no_grad():
                # Generate group_size candidates per input example.
                # Total generated sequences N = B * group_size.
                outputs = model.generate(samples, mode="sample", num_return_sequences=group_size)
        
        # Extract predictions and align ground-truth texts.
        # outputs["gt_reports"] is assumed to be length B, so we repeat each GT group_size times to match N predictions.
        pred_texts = outputs["predicted_reports"]
        gt_texts = [gt for gt in outputs["gt_reports"] for _ in range(group_size)]
        
        generated_ids = outputs["output_ids"]
        device = generated_ids.device

        # Compute rewards for each generated candidate.
        rewards  = reward_fn(pred_texts, gt_texts)
        rewards = rewards.to(device)

        return samples, outputs, rewards

    def logprobs_from_logits(self, logits, labels):
        # logits: [batch, seq_len, vocab_size]
        # label: [batch, seq_len]
        if logits.dtype in [torch.float32, torch.float64]:
            logits_labels = torch.gather(logits, dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
            logsumexp_values = torch.logsumexp(logits, dim=-1)
            logprobs_labels = logits_labels - logsumexp_values
        else:
            logprobs_labels = []
            for row_logits, row_labels in zip(logits, labels):
                row_logprobs = F.log_softmax(row_logits, dim=-1) # [batch, seq_len, vocab_size]
                row_logprobs_labels = row_logprobs.gather(dim=-1, index=row_labels.unsqueeze(-1)).squeeze(-1)
                logprobs_labels.append(row_logprobs_labels)
            logprobs_labels = torch.stack(logprobs_labels)
        return logprobs_labels
    
    def _reload_best_model(self, model, output_dir):
        """
        Load the best checkpoint.
        """
        checkpoint_path = os.path.join(output_dir, "checkpoint_best.pth")

        logging.info("Loading checkpoint from {}.".format(checkpoint_path))
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only = False)
        target = model.module if hasattr(model, "module") else model
        msg=target.load_state_dict(checkpoint["model"], strict=False)
        # print("Loading best checkpoint message: ", msg)
        return model
    
    def aprpo_rl_train_loop(
        self,
        epoch,
        iters_per_epoch,
        model,
        data_loader,
        optimizer,
        start_iters=None,
        cuda_enabled=True,
        output_dir=None,
        accum_grad_iters=1,
        epsilon_low=0.2,
        epsilon_high=0.28,
        scaler=None,
        use_amp=False,
        log_freq=50,
        use_zero_optimizer=True,
        group_size=4, # number of candidates sampled per input (G)
        temperature = 0.7, # logit temperature for sampling / logprob computation
        step_log_fn=None,
        update_ref_every: int = 3, # update ref/current from best checkpoint every M epochs
        update_old_every: int = 16, # refresh "old policy" snapshot every N optimizer steps
        ent_coef = 1e-3, # entropy regularization coefficient
        b_min = 0.05, # per-sequence KL weight lower bound
        b_base = 0.10, # per-sequence KL weight base
        b_max = 0.12,  # per-sequence KL weight upper bound
    ):
        """
        Main RL loop for Qwen:
          - DAPO-style token-level clipped policy gradient (PPO-like)
          - group-wise normalized advantages (within each input's sampled candidates)
          - dynamic per-sequence KL coefficient beta based on (advantage sign) and (ref entropy rank)
          - periodic ref/current swap from "best checkpoint"
          - periodic old-policy refresh
        """
        use_amp = scaler is not None
        dtype = next(model.parameters()).dtype
        print(f"use_amp: {use_amp}, dtype: {dtype}")

        # ============================================================
        # 0) Every M epochs: load "best checkpoint" into current model
        #    then rebuild reference policy parameters from that model.
        #    This is used to prevent drift and keep a stable reference.
        # ============================================================
        if update_ref_every and epoch > 0 and (epoch % update_ref_every == 0):
            print(f"== Update ref and current policy from best @ epoch {epoch} ==")
            
            # synchronize all ranks before swapping
            if dist.is_available() and dist.is_initialized():
                dist.barrier()
            
            swapped = False
            try:
                # only rank0 reads checkpoint from disk
                if (not dist.is_available()) or (not dist.is_initialized()) or dist.get_rank() == 0:
                    model = self._reload_best_model(model, output_dir)
            
                # broadcast updated weights to all ranks (so everyone matches rank0)
                if dist.is_available() and dist.is_initialized():
                    dist.barrier()
                    # for p in model.parameters():
                    #     dist.broadcast(p.data, src=0)
                    with torch.no_grad():
                        for t in model.state_dict().values():
                            if isinstance(t, torch.Tensor):
                                dist.broadcast(t, src=0)
                    dist.barrier()
                swapped = True
            except FileNotFoundError:
                logging.warning("checkpoint_best.pth does not exist, skip ref/current swap.")
            except Exception as e:
                logging.exception(f"swap-to-best failed, skip this swap: {e}")

            if swapped:
                # rebuild reference policy snapshot
                self.ref_state_dict = self.get_ref_state_dict(model)

                # also rebuild old policy snapshot to match the swapped weights
                old_lora_state_dict = self.get_lora_state_dict(model)
                
                # clear optimizer states (important when weights changed abruptly)
                try:
                    if hasattr(model, "optimizer") and hasattr(model.optimizer, "zero_grad"):
                        model.optimizer.zero_grad()
                        if hasattr(model.optimizer, "optimizer") and hasattr(model.optimizer.optimizer, "state"):
                            model.optimizer.optimizer.state.clear()
                    else:
                        optimizer.zero_grad(set_to_none=True)
                        if hasattr(optimizer, "state"):
                            optimizer.state.clear()
                except Exception:
                    pass

                # reset AMP scaler if used
                if scaler is not None:
                    try:
                        scaler.reset()
                    except Exception:
                        pass
                torch.cuda.empty_cache()

        # ============================================================
        # 1) Initialize old policy (if not created above) and ref policy
        # ============================================================
        if 'old_lora_state_dict' not in locals():
            old_lora_state_dict = self.get_lora_state_dict(model)
        if epoch == 0 or self.ref_state_dict is None:
            self.ref_state_dict = self.get_ref_state_dict(model)

        # ============================================================
        # 2) Training loop
        # ============================================================
        with torch.amp.autocast(dtype=dtype, device_type="cuda"):
            metric_logger = MetricLogger(delimiter="  ")
            metric_logger.add_meter("lr", SmoothedValue(window_size=1, fmt="{value:.6f}"))
            metric_logger.add_meter("loss", SmoothedValue(window_size=1, fmt="{value:.4f}"))
            metric_logger.add_meter("rl_loss", SmoothedValue(window_size=1, fmt="{value:.4f}"))
            metric_logger.add_meter("kl_loss", SmoothedValue(window_size=1, fmt="{value:.5f}"))
            metric_logger.add_meter("entropy_cur", SmoothedValue(window_size=1, fmt="{value:.4f}"))
            metric_logger.add_meter("reward_mean", SmoothedValue(window_size=1, fmt="{value:.4f}"))
            metric_logger.add_meter("reward_diff", SmoothedValue(window_size=1, fmt="{value:.4f}"))
            metric_logger.add_meter("reward_std", SmoothedValue(window_size=1, fmt="{value:.4f}"))

            logging.info(
                "Start training epoch {}, {} iters per inner epoch.".format(
                    epoch, iters_per_epoch
                )
            )

            header = "Train: data epoch: [{}]".format(epoch)
            if start_iters is None:
                inner_epoch = epoch
            else:
                inner_epoch = start_iters // iters_per_epoch
                header = header + "; inner epoch [{}]".format(inner_epoch)

            update_steps = 0 # counts optimizer steps

            for i in metric_logger.log_every(range(iters_per_epoch), log_freq, header):
                if i >= iters_per_epoch:
                    break

                # ============================================================
                # 2.1) Sample candidate reports with old policy and score reward
                #      group_size = G candidates per input example
                # ============================================================
                samples, outputs, rewards = self.sample_candidates(
                    data_loader,
                    model,
                    group_size,
                    self.bleu4_reward_fn,
                    old_lora_state_dict,
                    cuda_enabled=cuda_enabled,
                )

                # attach loop metadata
                samples.update(
                    {
                        "epoch": inner_epoch,
                        "num_iters_per_epoch": iters_per_epoch,
                        "iters": i,
                    }
                )

                # ============================================================
                # 2.2) Unpack rollout data
                #   - predicted_reports: list length N
                #   - gt_reports: list length B (then repeated to N = B*G)
                #   - generated_ids: [N, T]
                #   - prompt_lens: [N] (prompt length per sampled sequence)
                #   - attention_mask: [N, T]
                #   - pixel_values: [B, ...], then repeated to [N, ...]
                # ============================================================
                pred_texts = outputs["predicted_reports"]
                gt_texts = [gt for gt in outputs["gt_reports"] for _ in range(group_size)]
                generated_ids = outputs["output_ids"]
                prompt_lens = outputs["prompt_lens"]
                attention_mask = outputs["attention_mask"]
                pixel_values = outputs['pixel_values']
                image_grid_thw = outputs['image_grid_thw']

                # repeat image features to match N = B*G
                pixel_values_repeat = pixel_values.repeat_interleave(group_size, dim=0)
                image_grid_thw_repeat = image_grid_thw.repeat_interleave(group_size, dim=0)
                assert len(pred_texts) == len(gt_texts)
                
                # N = B*G
                num_outputs = len(gt_texts)
                batch_size = len(gt_texts) // group_size
                device = generated_ids.device
                assert len(generated_ids) == num_outputs == batch_size * group_size

                # clone tensors so in-place edits do not affect cached outputs
                input_ids = generated_ids.clone()
                attention_mask = attention_mask.clone()
                pixel_values_repeat = pixel_values_repeat.clone()
                image_grid_thw_repeat = image_grid_thw_repeat.clone()

                # ============================================================
                # 2.3) Build token-level loss mask
                #   - remove prompt tokens
                #   - remove padding tokens
                #
                # shift_* aligns with next-token prediction:
                #   shift_input_ids: tokens to be predicted, shape [N, T-1]
                #   mask: attention for those positions, shape [N, T-1]
                # ============================================================
                mask = attention_mask[:, 1:]
                shift_input_ids = input_ids[:, 1:]
                num_outputs = batch_size * group_size
                loss_mask = torch.zeros_like(mask, dtype=torch.bool)

                # mark generated region (excluding prompt)
                for idx in range(num_outputs):
                    start = prompt_lens[idx] - 1 # -1 because we shifted by 1
                    if start < mask.size(1):
                        loss_mask[idx, start:] = True

                # remove padding (only keep positions where attention_mask is 1)      
                loss_mask = loss_mask & mask.bool()

                # ============================================================
                # 2.4) Compute logprobs under old policy (for PPO ratio)
                #   logprobs_action_old: [N, T-1]
                # ============================================================
                with self.use_lora_params(model, old_lora_state_dict):
                    model.eval()
                    with torch.no_grad():
                        outputs_old = model(
                            input_ids=generated_ids,
                            attention_mask=attention_mask,
                            pixel_values=pixel_values_repeat,
                            image_grid_thw=image_grid_thw_repeat
                        )
                    
                logits_old = outputs_old.logits             # [N, T, V]
                shift_logits_old = logits_old[:, :-1, :]    # [N, T-1, V]
                shift_logits_old.div_(temperature)          # temperature scaling
                logprobs_action_old = self.logprobs_from_logits(shift_logits_old, shift_input_ids)  # [N, T-1]
                logprobs_action_old[~loss_mask] = 0.0       # only keep generated tokens

                # ============================================================
                # 2.5) Compute logprobs under reference policy (for KL penalty)
                #   logprobs_action_ref: [N, T-1]
                # ============================================================
                with self.use_lora_params(model, self.ref_state_dict):
                    model.eval()
                    with torch.no_grad():
                        outputs_ref = model(
                            input_ids=generated_ids,
                            attention_mask=attention_mask,
                            pixel_values=pixel_values_repeat,
                            image_grid_thw=image_grid_thw_repeat
                        )

                logits_ref = outputs_ref.logits             # [N, T, V]
                shift_logits_ref = logits_ref[:, :-1, :]    # [N, T-1, V]
                shift_logits_ref.div_(temperature)
                logprobs_action_ref = self.logprobs_from_logits(shift_logits_ref, shift_input_ids)  # [N, T-1]
                logprobs_action_ref[~loss_mask] = 0.0

                # ============================================================
                # 2.6) Compute logprobs under current policy (trainable)
                #   logprobs_action: [N, T-1]
                # ============================================================
                model.train()
                outputs_current = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    pixel_values=pixel_values_repeat,
                    image_grid_thw=image_grid_thw_repeat
                )

                logits = outputs_current.logits            # [N, T, V]
                shift_logits = logits[:, :-1, :]           # [N, T-1, V]
                shift_logits.div_(temperature)
                logprobs_action = self.logprobs_from_logits(shift_logits, shift_input_ids)   # [N, T-1]
                logprobs_action[~loss_mask] = 0.0

                 # ============================================================
                # 2.7) Group-wise normalized advantages
                #   rewards: [N] where N = B*G
                #   advantages: normalize within each group of size G
                # ============================================================
                advantages = torch.zeros_like(rewards)
                for j in range(batch_size):
                    group_rewards = rewards[j*group_size:(j+1)*group_size]
                    group_mean = group_rewards.mean()
                    group_std = group_rewards.std(unbiased=False) + 1e-6
                    advantages[j*group_size:(j+1)*group_size] = (group_rewards - group_mean) / group_std

                # broadcast advantage to token level, only on generated tokens
                adv_tensor = torch.zeros_like(logprobs_action)
                for idx in range(num_outputs):
                    adv_tensor[idx, loss_mask[idx]] = advantages[idx]

                # ============================================================
                # 2.8) DAPO-style clipped policy gradient (PPO-like)
                #
                # ratio = exp(log pi - log pi_old)
                # clipped_ratio in [1-eps_low, 1+eps_high]
                #
                # then extra handling for adv < 0, using pg_losses3 as a cap
                # ============================================================
                delta_logp = logprobs_action - logprobs_action_old # [N, T-1]
                
                # clamp for numeric stability (avoid exp overflow)
                delta_logp = torch.clamp(delta_logp, min=-20.0, max=20.0)  
                
                ratio = torch.exp(delta_logp)
                clipped_ratio = torch.clamp(ratio, 1.0 - epsilon_low, 1.0 + epsilon_high)
                
                pg_losses1 = -adv_tensor * ratio
                pg_losses2 = -adv_tensor * clipped_ratio
                clip_pg_losses1 = torch.maximum(pg_losses1, pg_losses2)

                # when adv < 0 
                pg_losses3 = -adv_tensor * 10
                clip_pg_losses2 = torch.min(pg_losses3, clip_pg_losses1)

                # choose different clipping behavior by advantage sign
                pg_losses = torch.where(adv_tensor < 0, clip_pg_losses2, clip_pg_losses1)

                # final RL loss: masked mean over generated tokens
                rl_loss = (pg_losses * loss_mask).sum() / (loss_mask.sum() + 1e-8)

                # ============================================================
                # 2.9) Entropy regularization term
                #
                #   - per-sequence entropy of reference policy (detached)
                #   - mean entropy of current policy (with gradient)
                #
                # Notation:
                #   N = B*G, L = T-1, V = vocab
                # ============================================================
                loss_mask_f = loss_mask.float()
                
                logits_ref = shift_logits_ref                                        # [N, L, V]
                logsumexp_ref = torch.logsumexp(logits_ref, dim=-1)                  # [N, L]
                probs_ref = torch.softmax(logits_ref, dim=-1)                        # [N, L, V]
                expected_logit_ref = (probs_ref * logits_ref).sum(dim=-1)            # [N, L]
                entropy_token_ref = (logsumexp_ref - expected_logit_ref)             # [N, L]
                
                # masked mean entropy per sequence (then detach)
                entropy_seq_ref = (entropy_token_ref * loss_mask_f).sum(dim=1) / (loss_mask_f.sum(dim=1).clamp_min(1e-8))
                entropy_seq_ref = entropy_seq_ref.detach()  # no gradient through beta construction
                
                # current policy entropy (keep gradient)
                logits_cur = shift_logits                                            # [N, L, V]
                logsumexp_cur   = torch.logsumexp(logits_cur, dim=-1)                # [N, L]
                probs_cur       = torch.softmax(logits_cur, dim=-1)                  # [N, L, V]
                expected_logit  = (probs_cur * logits_cur).sum(dim=-1)               # [N, L]
                entropy_tok_cur = (logsumexp_cur - expected_logit)                   # [N, L]

                # masked mean over all generated tokens (global mean)
                entropy_mean_cur = (entropy_tok_cur * loss_mask_f).sum() / (loss_mask_f.sum().clamp_min(1e-8))

                # ============================================================
                # 2.10) Normalize ref entropy within each group to x in [0,1]
                #   entropy_grouped: [B, G]
                #   x: [N]
                # ============================================================
                entropy_grouped = entropy_seq_ref.view(batch_size, group_size)       # [B, G]
                h_min = entropy_grouped.min(dim=1, keepdim=True)[0]
                h_max = entropy_grouped.max(dim=1, keepdim=True)[0]
                x_group = (entropy_grouped - h_min) / (h_max - h_min + 1e-6)         # [B, G], in [0,1]
                x = x_group.reshape(-1)                                              # [N]
                
                # ============================================================
                # 2.11) Build per-sequence KL coefficient beta
                #   If adv > 0:
                #     beta = b_min + (b_base - b_min) * (1 - x)
                #   If adv < 0:
                #     beta = b_base + (b_max - b_base) * x
                #   If adv == 0:
                #     beta = b_base
                #
                # Intuition (rough):
                #   - positive samples: prefer smaller beta for higher entropy
                #   - negative samples: allow larger beta for higher entropy
                # ============================================================
                beta = torch.empty_like(x)
                pos = advantages > 0
                neg = advantages < 0
                eq  = ~(pos | neg)

                beta[pos] = b_min + (b_base - b_min) * (1.0 - x[pos])
                beta[neg] = b_base + (b_max - b_base) * x[neg]
                beta[eq]  = b_base
                beta = beta.clamp(min=0.0)  # safety
                
                # expand to token dimension, shape [N, 1] to broadcast to [N, L]
                beta_tokens = beta.view(-1, 1)                                   
                
                # ============================================================
                # 2.12) Tokenwise KL penalty (reverse KL estimator)
                #
                # kl = log q - log p, where:
                #   q = ref policy, p = current policy
                #
                # estimator:
                #   kld = (q/p) - log(q/p) - 1
                #
                # J. Schulman. Approximating kl divergence, 2020.
                # URL http://joschu.net/blog/kl-approx.html
                # ============================================================
                kl = logprobs_action_ref - logprobs_action                          # [N, L]
                kl = torch.clamp(kl, min=-20, max=20)
                ratio = torch.exp(kl)                                               # q/p
                kld = ratio - kl - 1.0                                              
                kld = torch.clamp(kld, min=-10, max=10)
                
                # apply per-sequence coefficient and mask
                kl_weighted = kld * beta_tokens                                     # [N, L]
                kl_loss = (kl_weighted * loss_mask_f).sum() / (loss_mask_f.sum() + 1e-8)

                # total loss:
                #   minimize RL loss + KL loss, maximize entropy
                loss = rl_loss + kl_loss - ent_coef * entropy_mean_cur

                # ============================================================
                # 2.13) Backprop and optimizer step
                #   - support ZeRO optimizer (model.backward/model.step)
                #   - support AMP scaler when not using ZeRO
                # ============================================================
                if use_zero_optimizer:
                    model.backward(loss)
                else:
                    if use_amp:
                        scaler.scale(loss).backward()
                    else:
                        loss.backward()
                
                # gradient accumulation: step every accum_grad_iters iterations
                if (i + 1) % accum_grad_iters == 0:
                    if use_zero_optimizer:
                        model.step()
                    else:
                        if use_amp:
                            scaler.step(optimizer)
                            scaler.update()
                        else:
                            optimizer.step()
                        optimizer.zero_grad()
                    update_steps += 1
                
                # ============================================================
                # 2.14) Refresh old policy snapshot every N update steps
                #   old_lora_state_dict is used as pi_old in the ratio
                # ============================================================
                if update_steps % update_old_every == 0:
                    old_lora_state_dict = self.get_lora_state_dict(model)

                # free intermediate tensors to reduce peak memory
                del outputs, outputs_old, outputs_ref, outputs_current
                torch.cuda.empty_cache()

                # ============================================================
                # 2.15) Metric logging
                #   rewards are grouped by input (B groups), each size G
                #   we log mean, within-group std, and within-group diff
                # ============================================================
                assert rewards.numel() % group_size == 0
                B = rewards.numel() // group_size
                rewards_grouped = rewards.view(B, group_size)                  # [B, G]
                group_means = rewards_grouped.mean(dim=1)                      # [B]
                group_diffs = rewards_grouped.max(dim=1).values - rewards_grouped.min(dim=1).values  # [B]
                
                device = rewards.device
                local_sum_mean   = torch.tensor(float(group_means.sum().item()), device=device)
                group_stds = rewards_grouped.std(dim=1, unbiased=False)  # [B]
                local_sum_std = torch.tensor(float(group_stds.sum().item()), device=device)
                local_cnt_groups = torch.tensor(float(B), device=device)
                local_sum_diff   = torch.tensor(float(group_diffs.sum().item()), device=device)
                if dist.is_available() and dist.is_initialized():
                    dist.all_reduce(local_sum_mean,   op=dist.ReduceOp.SUM)
                    dist.all_reduce(local_sum_std,   op=dist.ReduceOp.SUM)
                    dist.all_reduce(local_cnt_groups, op=dist.ReduceOp.SUM)
                    dist.all_reduce(local_sum_diff,   op=dist.ReduceOp.SUM)

                # global averages over all groups across all ranks
                global_group_mean = (local_sum_mean / (local_cnt_groups + 1e-8)).item()
                global_within_std_mean = (local_sum_std / (local_cnt_groups + 1e-8)).item()
                global_group_diff_mean = (local_sum_diff / (local_cnt_groups + 1e-8)).item()

                metric_logger.update(
                    loss=loss.item(),
                    rl_loss=rl_loss.item(),
                    kl_loss=kl_loss.item(),
                    entropy_cur=entropy_mean_cur.item(),
                    lr=optimizer.param_groups[0]["lr"],
                    reward_mean=global_group_mean,
                    reward_std=global_within_std_mean,
                    reward_diff=global_group_diff_mean,
                )
            
                # ============================================================
                # 2.16) Optional: write per-step stats to log.txt (rank0 only)
                # ============================================================
                if step_log_fn is not None and is_main_process():
                    step_stats = {
                        "epoch": int(epoch),
                        "iter": int(i),
                        "global_step": int(epoch * iters_per_epoch + i),
                        "ts": time.time(),
                    }
                    for name, meter in metric_logger.meters.items():
                        v = getattr(meter, "value", None)
                        if v is not None:
                            try:
                                step_stats[name] = float(v)
                            except Exception:
                                pass
                    step_log_fn(step_stats)

        # ============================================================
        # 3) End of epoch: synchronize and return global averages
        # ============================================================
        metric_logger.synchronize_between_processes()
        print("Averaged RL stats: " + str(metric_logger.global_avg()))
        return {
            k: "{:.5f}".format(meter.global_avg)
            for k, meter in metric_logger.meters.items()
        }
    
    def _train_inner_loop(
        self,
        epoch,
        iters_per_epoch,
        model,
        data_loader,
        optimizer,
        scaler=None,
        start_iters=None,
        log_freq=50,
        cuda_enabled=False,
        accum_grad_iters=1,
        use_zero_optimizer=False,
        step_log_fn=None
    ):
        """
        An inner training loop compatible with both epoch-based and iter-based training.

        When using epoch-based, training stops after one epoch; when using iter-based,
        training stops after #iters_per_epoch iterations.
        """
    
        use_amp = scaler is not None
        dtype = next(model.parameters()).dtype
        print(f"use_amp: {use_amp}, dtype: {dtype}")
        if not hasattr(data_loader, "__next__"):
            # convert to iterator if not already
            data_loader = iter(data_loader)

        metric_logger = MetricLogger(delimiter="  ")
        metric_logger.add_meter("lr", SmoothedValue(window_size=1, fmt="{value:.6f}"))
        metric_logger.add_meter("loss", SmoothedValue(window_size=1, fmt="{value:.4f}"))

        logging.info(
            "Start training epoch {}, {} iters per inner epoch.".format(
                epoch, iters_per_epoch
            )
        )
        header = "Train: data epoch: [{}]".format(epoch)
        if start_iters is None:
            # epoch-based runner
            inner_epoch = epoch
        else:
            # In iter-based runner, we schedule the learning rate based on iterations.
            inner_epoch = start_iters // iters_per_epoch
            header = header + "; inner epoch [{}]".format(inner_epoch)

        for i in metric_logger.log_every(range(iters_per_epoch), log_freq, header):
            # if using iter-based runner, we stop after iters_per_epoch iterations.
            if i >= iters_per_epoch:
                break

            samples = next(data_loader)
            samples = prepare_sample(samples, cuda_enabled=cuda_enabled)
            samples.update(
                {
                    "epoch": inner_epoch,
                    "num_iters_per_epoch": iters_per_epoch,
                    "iters": i,
                }
            )
            model_output = model(samples)
            loss = model_output["loss"]
            
            for key in model_output:
                if key.endswith('_loss') and key != "loss" and not hasattr(metric_logger, key):
                    metric_logger.add_meter(key, SmoothedValue(window_size=1, fmt="{value:.4f}"))
                   
            # after_train_step()
            if use_zero_optimizer:
                model.backward(loss)
            else:
                if use_amp:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
                    
            # update gradients every accum_grad_iters iterations
            if (i + 1) % accum_grad_iters == 0:
                if use_zero_optimizer:
                    # optimizer.clip_grad_norm(1.0)
                    model.step()
                else:
                    if use_amp:
                        scaler.step(optimizer)
                        scaler.update()                     
                    else:    
                        optimizer.step()
                    optimizer.zero_grad()

            loss_value = loss.item()
            if not math.isnan(loss_value) and not math.isinf(loss_value):
                metric_logger.update(loss=loss_value)
            else:
                print(f"[rank {dist.get_rank() if dist.is_initialized() else 0}] Step {i} loss is nan/inf, skip update.")


            for key in model_output:
                if key.endswith('_loss') and key != "loss" and hasattr(metric_logger, key):
                    value = model_output[key]
                    metric_logger.update(**{key: value.item() if torch.is_tensor(value) else value})
        
            metric_logger.update(lr=optimizer.param_groups[0]["lr"])

            del model_output, loss
            torch.cuda.empty_cache()

            if step_log_fn is not None and is_main_process():
                step_stats = {
                    "epoch": int(epoch),
                    "iter": int(i),
                    "global_step": int(epoch * iters_per_epoch + i),
                    "ts": time.time(),
                }
                for name, meter in metric_logger.meters.items():
                    v = getattr(meter, "value", None)
                    if v is not None:
                        try:
                            step_stats[name] = float(v)
                        except Exception:
                            pass
                step_log_fn(step_stats)
            
        # after train_epoch()
        # gather the stats from all processes
        metric_logger.synchronize_between_processes()
        logging.info("Averaged stats: " + str(metric_logger.global_avg()))
        
        return {
            k: "{:.3f}".format(meter.global_avg)
            for k, meter in metric_logger.meters.items()
        }

    def evaluation(self, model, data_loader, cuda_enabled=True):
        """
        Run evaluation focusing on NLP metrics with main_process decorator
        """

        metric_logger = MetricLogger(delimiter="  ")
        header = "Validation:"
        print_freq = 100
        model.eval()

        all_predictions = []
        all_ground_truths = []
        all_study_id = []
        
        all_predicted_categories=[]
        all_categories = []
        all_positive_categories = []
        
        for batch_idx, samples in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
            samples = prepare_sample(samples, cuda_enabled=cuda_enabled)
            
            outputs = self.valid_step(model=model, samples=samples)
            
            all_predictions.extend(outputs['predicted_reports'])
            all_ground_truths.extend(outputs['gt_reports'])
            all_study_id.extend(outputs['id'])
            
            all_predicted_categories.extend(outputs['predicted_categories'])
            all_categories.extend(outputs['categories'])
            all_positive_categories.extend(outputs['positive_categories'])
        
        if is_dist_avail_and_initialized():
            rank = dist.get_rank()
            world_size = dist.get_world_size()
        else:
            rank = 0
            world_size = 1

        result_dir = Path(registry.get_path("result_dir"))
        cache_dir = result_dir / "val_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)

        part_path = cache_dir / f"pred_rank{rank}.jsonl.gz"
        with gzip.open(part_path, "wt", encoding="utf-8") as f:
            for pred, gt, sid, pred_cat, cat, pos_cat in zip(
                all_predictions,
                all_ground_truths,
                all_study_id,
                all_predicted_categories,
                all_categories,
                all_positive_categories,
            ):
                rec = {
                    "id": sid,
                    "pred": pred,
                    "gt": gt,
                    "predicted_categories": pred_cat,
                    "categories": cat,
                    "positive_categories": pos_cat,
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        if is_dist_avail_and_initialized():
            dist.barrier()  

        # Merge, remove duplicates
        if is_main_process():
            merged_predictions = []
            merged_ground_truths = []
            merged_study_id = []
            merged_predicted_categories = []
            merged_categories = []
            merged_positive_categories = []

            for r in range(world_size):
                p = cache_dir / f"pred_rank{r}.jsonl.gz"
                if not p.exists():
                    print(f"[warn] missing cache file from rank {r}: {p}")
                    continue
                with gzip.open(p, "rt", encoding="utf-8") as f:
                    for line in f:
                        obj = json.loads(line)
                        merged_predictions.append(obj["pred"])
                        merged_ground_truths.append(obj["gt"])
                        merged_study_id.append(obj["id"])
                        merged_predicted_categories.append(obj["predicted_categories"])
                        merged_categories.append(obj["categories"])
                        merged_positive_categories.append(obj["positive_categories"])

            # Deduplication: Keep the last occurrence of the ID 
            # when reading from multiple GPUs, the same image may be read by multiple GPUs.
            last_idx = {}
            for idx, sid in enumerate(merged_study_id):
                last_idx[sid] = idx
            unique_indices = list(last_idx.values())

            all_predictions = [merged_predictions[i] for i in unique_indices]
            all_ground_truths = [merged_ground_truths[i] for i in unique_indices]
            all_study_id = [merged_study_id[i] for i in unique_indices]
            all_predicted_categories = [merged_predicted_categories[i] for i in unique_indices]
            all_categories = [merged_categories[i] for i in unique_indices]
            all_positive_categories = [merged_positive_categories[i] for i in unique_indices]

            print(f"total number of unique_indices: {len(unique_indices)}")
            print(f"Total samples processed: {len(all_predictions)}")

            print("\n" + "=" * 50 + " Examples " + "=" * 50)
            for pred, true, sid, pred_cat, gt_cat in zip(
                all_predictions[-15:-4],
                all_ground_truths[-15:-4],
                all_study_id[-15:-4],
                all_predicted_categories[-15:-4],
                all_categories[-15:-4],
            ):
                print(f"id: {sid}")
                print(f"Pred: {str(pred)}")
                print(f"True: {str(true)}")
                print("-" * 100)
                print(f"id: {sid}")
                print(f"pred cate: {pred_cat}")
                print(f"gt cate: {gt_cat}")
                print("*" * 100)
            print("=" * 100 + "\n")

            print("len(all_predictions): ", len(all_predictions))
            print("len(all_ground_truths): ", len(all_ground_truths))
            val_metrics = self.compute_metrics(all_predictions, all_ground_truths)
        else:
            val_metrics = None
            all_predictions = []
            all_ground_truths = []
            all_study_id = []
            all_predicted_categories = []
            all_categories = []
            all_positive_categories = []

        self._last_eval_payload = {
            "pred": all_predictions,
            "gt": all_ground_truths,
            "id": all_study_id,
            "predicted_categories": all_predicted_categories,
            "categories": all_categories,
            "positive_categories": all_positive_categories,
        }
        self._last_eval_metrics = val_metrics

        if is_dist_avail_and_initialized():
            dist.barrier()  # Ensure all processes are synchronized
            
        return val_metrics

    def clean_for_meteor_io(self, text: str) -> str:
        """
        Minimal cleaning: only to avoid issues with the METEOR evaluation protocol when the generated results are too bad.
        - Replace '|||' with a space
        - Replace \n \r \t with a space
        - Collapse multiple spaces
        - Strip leading and trailing spaces
        """
        if not isinstance(text, str):
            return ""
        s = (text.replace("|||", " ")
                .replace("\n", " ")
                .replace("\r", " ")
                .replace("\t", " "))
        _PROTO_PAT = re.compile(r"\s+")
        s = _PROTO_PAT.sub(" ", s).strip()
        return s

    @main_process
    def compute_metrics(self, predictions, ground_truths):
    # def compute_metrics(self, predictions, ground_truths, probs, gt_labels):
        val_metrics = {}
        # Prepare data for NLP metrics calculation
        gts = {}  # ground truth
        res = {}  # predictions
        valid_samples = 0
        empty_samples = 0
        
        for i, (pred, gt) in enumerate(zip(predictions, ground_truths)):
            if not pred or not gt:
                empty_samples += 1
                continue
                
            try:
                # This cleaning step prevents the evaluation from crashing when the generated output is invalid, 
                # which would otherwise cause METEOR scoring to fail.
                pred = self.clean_for_meteor_io(str(pred)) 
                gt = self.clean_for_meteor_io(str(gt))
                gts[valid_samples] = [gt]
                res[valid_samples] = [pred]
                valid_samples += 1
            except Exception as e:
                print(f"Error processing sample {i}: {e}")
                empty_samples += 1
                continue

        if valid_samples > 0:
            try:
                # Calculate BLEU scores
                eval_results = self.compute_scores(gts, res)
                
                # update metric
                val_metrics.update(eval_results)

                agg_metrics = 0.0
                count = 0
                for metric, score in eval_results.items():
                    if metric.startswith('BLEU') or metric.startswith('METEOR') or metric.startswith('ROUGE'):  
                        agg_metrics += score
                        count += 1
                if count > 0:
                    agg_metrics /= count
                
                val_metrics['agg_metrics'] = agg_metrics

                metrics_str = "NLP Metrics |"
                for metric, score in eval_results.items():
                    try:
                        if isinstance(score, bytes):
                            score = float(score.decode('utf-8').split()[0])
                        elif isinstance(score, str) and ' ' in score:
                            score = float(score.split()[0])
                        score = float(score)
                        metrics_str += f" {metric}: {score:.4f} |"
                    except (ValueError, AttributeError) as e:
                        metrics_str += f" {metric}: {score} |"
                        print(metrics_str)
                        print('\n')

            except Exception as e:
                print(f"Error computing NLP metrics: {e}")


        # Add sample statistics to metrics
        val_metrics.update({
            'valid_samples': valid_samples,
            'empty_samples': empty_samples,
            'total_samples': len(predictions)
        })
        return val_metrics

    def get_scorer(self):
        scorers = [
            (Bleu(4), ["BLEU_1", "BLEU_2", "BLEU_3", "BLEU_4"]),
            (Meteor(), "METEOR"),
            (Rouge(), "ROUGE_L"),
        ]
        return scorers

    def compute_scores(self, gts, res):
        """
        Performs the MS COCO evaluation using the Python 3 implementation (https://github.com/salaniz/pycocoevalcap)

        :param gts: Dictionary with the image ids and their gold captions,
        :param res: Dictionary with the image ids ant their generated captions
        :print: Evaluation score (the mean of the scores of all the instances) for each measure
        """

        # Set up scorers
        scorers = self.get_scorer()
        eval_res = {}
        # Compute score for each metric
        for scorer, method in scorers:
            try:
                score, scores = scorer.compute_score(gts, res, verbose=0)
            except TypeError:
                score, scores = scorer.compute_score(gts, res)

            if type(method) == list:
                for sc, m in zip(score, method):
                    eval_res[m] = sc
            else:
                eval_res[method] = score
        return eval_res

    def valid_step(self, model, samples):
        """
        Validation step that computes ITC and ITM losses without gradients
        """
        with torch.inference_mode():
            output = model.generate(samples)
        
        preds = output["predicted_reports"]
        pred_cats = output.get("predicted_categories_list", None)

        # Handle missing predicted categories for stage 1
        if pred_cats is None:
            pred_cats = [None] * len(preds)

        return {
            "predicted_reports": output["predicted_reports"],
            "gt_reports": output["gt_reports"],
            "predicted_categories": pred_cats,
            "id": samples["id"],
            "categories": samples['categories'],
            "positive_categories": samples['positive_categories'],
        }

    def after_evaluation(self, val_result, split_name, epoch):
        """
        Process evaluation results and return logging statistics
        Args:
            val_result (dict): Dictionary containing evaluation metrics 
            split_name (str): Name of dataset split (e.g., 'val', 'test')
            epoch (int/str): Current epoch or 'best'
        Returns:
            log_stats (dict): Processed logging statistics
        """
        # Initialize logging stats with basic info
        log_stats = {
            "epoch": epoch,
            "split_name": split_name,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }

        # Add metrics with split name prefix for all numeric values
        for k, v in val_result.items():
            if isinstance(v, (int, float)):
                log_stats[f"{split_name}/{k}"] = v
        
        # Check and copy agg_metrics (main metric for model selection)
        assert "agg_metrics" in val_result, "agg_metrics not found in evaluation results"
        log_stats["agg_metrics"] = val_result["agg_metrics"]
        
        # Add sample statistics if available
        stats_keys = ["valid_samples", "empty_samples", "total_samples"]
        for key in stats_keys:
            if key in val_result:
                log_stats[f"{split_name}/{key}"] = val_result[key]
        
        # Add evaluation time statistics if available
        if hasattr(self, 'eval_start_time'):
            eval_time = time.time() - self.eval_start_time
            log_stats[f"{split_name}/eval_time"] = eval_time
        
        # Print main evaluation metrics
        print(f"\n{split_name} Evaluation Results - Epoch {epoch}:")
        print(f"agg_metrics: {log_stats['agg_metrics']:.4f}")
        
        # Print all BLEU, ROUGE and CIDEr scores
        for k, v in log_stats.items():
            # Only print standard NLP metrics
            if k.startswith(f"{split_name}/BLEU") or \
                k.startswith(f"{split_name}/ROUGE") or \
                k.startswith(f"{split_name}/CIDEr") or \
                k.startswith(f"{split_name}/METEOR"):
                print(f"{k}: {v:.4f}")
        print("\n")

        return log_stats
