"""Utility functions to measure system status."""

import logging
import os
import random
import time
from collections import OrderedDict

import numpy as np
import torch
from colorama import Back, Fore, Style
from prettytable import PrettyTable
from torchinfo import summary

# -------------------------------------------------------------------------------------------------


def set_seed(seed: int = 42) -> None:
    """Take care of all the random seeds to run deterministically."""
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

    # When running on the CuDNN backend, two further options must be set
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        gpu_name = torch.cuda.get_device_name()
        if "amd" not in gpu_name.lower():
            print(f"{Fore.YELLOW}Set the cudnn.deterministic to be True.{Style.RESET_ALL}")
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        else:
            print(f"{Fore.YELLOW}Do not set the cudnn.deterministic.{Style.RESET_ALL}")
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")


# -------------------------------------------------------------------------------------------------


def start_timer(enable=False):
    if enable:
        if torch.cuda.is_available():
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            return (start, end)
        else:
            start = time.perf_counter()
            return (start,)
    else:
        return None


def end_timer(enable=False, t=None, msg="", verbose=True):
    if enable:
        if torch.cuda.is_available():
            t[1].record()
            torch.cuda.synchronize()
            duration = t[0].elapsed_time(t[1])
            if verbose:
                print(f"{Fore.LIGHTBLUE_EX}{msg} {duration} ms ...{Style.RESET_ALL}", flush=True)
            return duration
        else:
            duration = time.perf_counter() - t[0]
            if verbose:
                print(
                    f"{Fore.LIGHTBLUE_EX}{msg} {duration * 1e3} ms ...{Style.RESET_ALL}",
                    flush=True,
                )
            return duration
    else:
        return 0.0


# -------------------------------------------------------------------------------------------------


def get_cuda_info(device):
    if torch.cuda.is_available():
        return {
            "PyTorch_version": torch.__version__,
            "CUDA_version": torch.version.cuda,
            "cuDNN_version": torch.backends.cudnn.version(),
            "Arch_version": torch._C._cuda_getArchFlags(),
            "device_count": torch.cuda.device_count(),
            "device_name": torch.cuda.get_device_name(device=device),
            "device_id": torch.cuda.current_device(),
            "cuda_capability": torch.cuda.get_device_capability(device=device),
            "device_properties": torch.cuda.get_device_properties(device=device),
            "reserved_memory": torch.cuda.memory_reserved(device=device) / 1024**3,
            "allocated_memory": torch.cuda.memory_allocated(device=device) / 1024**3,
            "max_allocated_memory": torch.cuda.max_memory_allocated(device=device) / 1024**3,
            "gpu_name": torch.cuda.get_device_name(),
        }
    else:
        return {
            "PyTorch_version": torch.__version__,
            "CUDA_version": -1,
            "cuDNN_version": -1,
            "Arch_version": -1,
            "device_count": 0,
            "device_name": "cpu",
            "device_id": 0,
            "cuda_capability": 0,
            "device_properties": 0,
            "reserved_memory": 0,
            "allocated_memory": 0,
            "max_allocated_memory": 0,
            "gpu_name": "none",
        }


def support_bfloat16(device):
    if torch.cuda.is_available():
        DISABLE_FLOAT16_INFERENCE = os.environ.get("DISABLE_FLOAT16_INFERENCE", "False")
        if DISABLE_FLOAT16_INFERENCE == "True":
            return False

        info = get_cuda_info(device)
        if (
            info["gpu_name"].find("A100") >= 0
            or info["gpu_name"].find("H100")
            or info["gpu_name"].find("B100")
            or info["gpu_name"].find("B200") >= 0
        ):
            return True
        else:
            return False
    else:
        return False


# -------------------------------------------------------------------------------------------------
def get_gpu_ram_usage(device="cuda:0"):
    """
    Get info regarding memory usage of a device
    @args:
        - device (torch.device): the device to get info about
    @rets:
        - result_string (str): a string containing the info.
    """
    result_string = (
        f"torch.cuda.memory_allocated: {torch.cuda.memory_allocated(device=device) / 1024 / 1024 / 1024:.3}GB\n"
        + f"torch.cuda.memory_reserved: {torch.cuda.memory_reserved(device=device) / 1024 / 1024 / 1024:.3f}GB\n"
        + f"torch.cuda.max_memory_reserved: {torch.cuda.max_memory_reserved(device=device) / 1024 / 1024 / 1024:.3f}GB"
    )

    return result_string


# -------------------------------------------------------------------------------------------------
# Model info


def model_info(model, config, H, W, D):
    """
    Prints model info and sets total and trainable parameters in the config
    @args:
        - model (torch model): the model to check parameters of
        - config (Namespace): runtime namespace for setup
    @rets:
        - model_summary (ModelStatistics object): the model summary
            see torchinfo/model_statistics.py for more information.
    """
    c = config
    col_names = (
        "num_params",
        "params_percent",
        "mult_adds",
        "input_size",
        "output_size",
        "trainable",
    )
    row_settings = ["var_names", "depth"]
    dtypes = [torch.float32]
    model = model.module if config.trainer.ddp else model
    model.train()

    for task_ind, task in enumerate(model.tasks.values()):
        C_out = task.C_out

        a_sample = task.train_set[0][0]  # Get a sample from the training set
        _C, D, H, W = a_sample.shape[:4]

        mod_batch_size = config.trainer.batch_size[task_ind]

        example_pre_input = a_sample.unsqueeze(0).repeat(mod_batch_size, 1, 1, 1, 1).to(c.device)

        example_pre_output = task.pre_component(example_pre_input)
        if not isinstance(example_pre_output, list):
            example_pre_output = [example_pre_output]

        pre_model_summary = summary(
            task.pre_component,
            verbose=0,
            mode="train",
            depth=c.trainer.summary_depth,
            input_data=[example_pre_input],
            col_names=col_names,
            row_settings=row_settings,
            dtypes=dtypes,
            device=config.device,
        )
        logging.info(
            f"{Fore.MAGENTA}{'-' * 40}Summary of pre component for task {task.task_name}{'-' * 40}{Style.RESET_ALL}"
        )
        logging.info(f"\n{pre_model_summary!s}")

        # -----------------------------------------------------------------------------------------
        example_backbone_output = model.backbone_component(example_pre_output)
        if not isinstance(example_backbone_output, list):
            example_backbone_output = [example_backbone_output]

        torch.cuda.empty_cache()

        # -----------------------------------------------------------------------------------------
        example_backbone_output.insert(0, example_pre_input)
        post_model_summary = summary(
            task.post_component,
            verbose=0,
            mode="train",
            depth=c.trainer.summary_depth,
            input_data=[example_backbone_output],
            col_names=col_names,
            row_settings=row_settings,
            dtypes=dtypes,
            device=config.device,
        )
        logging.info(
            f"{Fore.MAGENTA}{'-' * 40}Summary of post component for task {task.task_name}{'-' * 40}{Style.RESET_ALL}"
        )
        logging.info(f"\n{post_model_summary!s}")

        torch.cuda.empty_cache()

    # ------------------------------------------------------------------------------------------------------------------
    task_ind = 0
    torch.ones((mod_batch_size, C_out, D, H, W)).to(c.device)
    backbone_model_summary = summary(
        model.backbone_component,
        verbose=0,
        mode="train",
        depth=c.trainer.summary_depth,
        input_data=[example_pre_output],
        col_names=col_names,
        row_settings=row_settings,
        dtypes=dtypes,
        device=config.device,
    )

    logging.info(
        f"{Fore.MAGENTA}{'-' * 40}Summary of backbone component{'-' * 40}{Style.RESET_ALL}"
    )
    logging.info(f"\n{backbone_model_summary!s}")

    torch.cuda.empty_cache()

    return backbone_model_summary


# -------------------------------------------------------------------------------------------------


def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        params = parameter.numel()
        table.add_row([name, params])
        total_params += params
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params


# -------------------------------------------------------------------------------------------------
def get_device(device=None):
    """
    Wrapper around getting device
    @args:
        - device (torch.device): if not None this device will be returned
            otherwise check if cuda is available
    @rets:
        - device (torch.device): the device to be used.
    """
    return device if device is not None else "cuda" if torch.cuda.is_available() else "cpu"


# -------------------------------------------------------------------------------------------------


def none_or_str(value):
    """Convert arg from a string to None."""
    if value in ["None", "none", None]:
        return None
    return value


def str_to_bool(value):
    """Convert arg from a string to bool."""
    if str(value) in ["1", "True", "true", "T", "t"]:
        return True
    else:
        return False


def create_generic_class_str(obj: object, exclusion_list=None) -> str:
    """
    Create a generic name of a class
    @args:
        - obj (object): the class to make string of
        - exclusion_list (object list): the objects to exclude from the class string
    @rets:
        - class_str (str): the generic class string.
    """
    if exclusion_list is None:
        exclusion_list = [torch.nn.Module, OrderedDict]
    name = type(obj).__name__

    vars_list = []
    for key, value in vars(obj).items():
        valid = True
        for type_e in exclusion_list:
            if isinstance(value, type_e) or key.startswith("_"):
                valid = False
                break

        if valid:
            vars_list.append(f"{key}={value!r}")

    vars_str = ",\n".join(vars_list)
    return f"{name}({vars_str})"


# -------------------------------------------------------------------------------------------------


def get_rank_str(rank, global_rank=-1):
    if rank == 0:
        if global_rank >= 0:
            return (
                f"{Fore.BLUE}{Back.WHITE}rank {rank}, global_rank {global_rank} {Style.RESET_ALL}"
            )
        else:
            return f"{Fore.BLUE}{Back.WHITE}rank {rank} {Style.RESET_ALL}"
    if rank == 1:
        if global_rank >= 0:
            return f"{Fore.GREEN}{Back.WHITE}rank {rank}, global_rank {global_rank}  {Style.RESET_ALL}"
        else:
            return f"{Fore.GREEN}{Back.WHITE}rank {rank} {Style.RESET_ALL}"
    if rank == 2:
        if global_rank >= 0:
            return f"{Fore.YELLOW}{Back.WHITE}rank {rank}, global_rank {global_rank} {Style.RESET_ALL}"
        else:
            return f"{Fore.YELLOW}{Back.WHITE}rank {rank} {Style.RESET_ALL}"
    if rank == 3:
        if global_rank >= 0:
            return f"{Fore.MAGENTA}{Back.WHITE}rank {rank}, global_rank {global_rank} {Style.RESET_ALL}"
        else:
            return f"{Fore.MAGENTA}{Back.WHITE}rank {rank} {Style.RESET_ALL}"
    if rank == 4:
        if global_rank >= 0:
            return f"{Fore.LIGHTYELLOW_EX}{Back.WHITE}rank {rank}, global_rank {global_rank} {Style.RESET_ALL}"
        else:
            return f"{Fore.LIGHTYELLOW_EX}{Back.WHITE}rank {rank} {Style.RESET_ALL}"
    if rank == 5:
        if global_rank >= 0:
            return f"{Fore.LIGHTBLUE_EX}{Back.WHITE}rank {rank}, global_rank {global_rank} {Style.RESET_ALL}"
        else:
            return f"{Fore.LIGHTBLUE_EX}{Back.WHITE}rank {rank} {Style.RESET_ALL}"
    if rank == 6:
        if global_rank >= 0:
            return f"{Fore.LIGHTRED_EX}{Back.WHITE}rank {rank}, global_rank {global_rank} {Style.RESET_ALL}"
        else:
            return f"{Fore.LIGHTRED_EX}{Back.WHITE}rank {rank} {Style.RESET_ALL}"
    if rank == 7:
        if global_rank >= 0:
            return f"{Fore.LIGHTCYAN_EX}{Back.WHITE}rank {rank}, global_rank {global_rank} {Style.RESET_ALL}"
        else:
            return f"{Fore.LIGHTCYAN_EX}{Back.WHITE}rank {rank} {Style.RESET_ALL}"

    if global_rank >= 0:
        return (
            f"{Fore.WHITE}{Style.BRIGHT}rank {rank}, global_rank {global_rank} {Style.RESET_ALL}"
        )
    else:
        return f"{Fore.WHITE}{Style.BRIGHT}rank {rank} {Style.RESET_ALL}"


# -------------------------------------------------------------------------------------------------


class AverageMeter:
    """Computes and stores the average and current value."""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.vals = []
        self.counts = []

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.vals = []
        self.counts = []

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

        self.vals.append(val)
        self.counts.append(n)

    def update_array(self, vals):
        self.val = np.mean(vals)
        self.sum += np.sum(vals)
        self.count += vals.size
        self.avg = self.sum / self.count

        self.vals.append(list(vals))
        self.counts.append(list(np.ones_like(vals)))

    def status(self):
        return np.array(self.vals), np.array(self.counts)


# -------------------------------------------------------------------------------------------------
