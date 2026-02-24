"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE_Lavis file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

from dataclasses import dataclass
from typing import Optional, List

import torch
from transformers.modeling_outputs import (
    ModelOutput,
)

@dataclass
class Output(ModelOutput):
    """
    The output class for APRPO model.
    - predicted_reports (List[str]): The generated reports.
    - gt_reports (List[str]): The ground truth reports.
    - ids (List[str]): The ids of the samples.
    - predicted_categories_list (List[str]): The predicted categories.
    - probs (torch.Tensor): The probabilities of the predictions.
    - gt_labels (torch.Tensor): The ground truth labels.
    """
    predicted_reports: Optional[List[str]] = None
    gt_reports: Optional[List[str]] = None
    ids: Optional[List[str]] = None
    predicted_categories_list: Optional[List[str]] = None
    probs: Optional[torch.Tensor] = None
    gt_labels: Optional[torch.Tensor] = None