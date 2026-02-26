import cv2
import os
import datetime
import torch.distributed.distributed_c10d as c10d
import argparse
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import reva_po.tasks as tasks
from reva_po.common.config import Config
from reva_po.common.dist_utils import get_rank, init_distributed_mode
from reva_po.common.logger import setup_logger
from reva_po.common.registry import registry
from reva_po.common.utils import now
from reva_po.datasets.builders import *
from reva_po.models import *
from reva_po.processors import *
from reva_po.runners import *
from reva_po.tasks import *
import torchvision.transforms as transforms
import torch
from torchvision import transforms
import torch.distributed as dist
import socket
import torch.multiprocessing as mp 

def parse_args():
    parser = argparse.ArgumentParser(description="Training")
    parser.add_argument("--cfg-path", required=True, help="path to configuration file.")
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file (deprecate), "
        "change to --cfg-options instead.",
    )
    
    # deepspeed configurations
    parser.add_argument('--use_zero_optimizer', action='store_true', help='use ZeRO optimizer to save GPU memory')
    parser.add_argument('--local_rank', default=0, type=int, help='local rank')
    parser.add_argument('--deepspeed_config', type=str, default='train_configs/zero_configs/stage1.json', help='path to deepspeed configuration file')
    parser.add_argument('--train_batch_size', type=int, default=1, help='training batch size')
    parser.add_argument('--train_micro_batch_size_per_gpu', type=int, default=1, help='batch size per GPU')
    
    args = parser.parse_args()

    return args


def setup_seeds(config):
    seed = config.run_cfg.seed + get_rank()
    print("seed: ", seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    cudnn.benchmark = False
    cudnn.deterministic = True


def get_runner_class(cfg):
    """
    Get runner class from config. Default to epoch-based runner.
    """
    runner_cls = registry.get_runner_class(cfg.run_cfg.get("runner", "runner_base"))
    return runner_cls


def main():
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    hostname = socket.gethostname()
    local_rank = int(os.getenv('LOCAL_RANK', '0'))
    world_size = int(os.getenv('WORLD_SIZE', '1'))
    rank = int(os.getenv('RANK', '0'))
    print(f"HOSTNAME={hostname}, LOCAL_RANK={local_rank}, WORLD_SIZE={world_size}, RANK={rank}")

    # set before init_distributed_mode() to ensure the same job_id shared across all ranks.
    job_id = now()
    cfg = Config(parse_args())

    c10d._DEFAULT_PG_TIMEOUT = datetime.timedelta(days=365)
    if hasattr(c10d, "_DEFAULT_PG_NCCL_TIMEOUT"):
        c10d._DEFAULT_PG_NCCL_TIMEOUT = datetime.timedelta(days=365)

    init_distributed_mode(cfg.run_cfg)
    setup_seeds(cfg)

    # set after init_distributed_mode() to only log on master.
    setup_logger()
    # logging.getLogger("torch.distributed").setLevel(logging.ERROR)
    cfg.pretty_print()
    task = tasks.setup_task(cfg)
    print("cfg: ", cfg)
    datasets = task.build_datasets(cfg)
    model = task.build_model(cfg)
    # define arguments, required by deepspeed
    args = parse_args()
    args.train_batch_size = cfg.run_cfg.batch_size_train * cfg.run_cfg.world_size
    args.train_micro_batch_size_per_gpu = cfg.run_cfg.batch_size_train
    print("args.train_batch_size ", args.train_batch_size)
    print("args.train_micro_batch_size_per_gpu", args.train_micro_batch_size_per_gpu)

    runner = get_runner_class(cfg)(
        cfg=cfg, job_id=job_id, task=task, model=model, datasets=datasets, cmd_args=args,
    )

    runner.train()
    
if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
