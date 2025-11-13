"""Configuration for the optimizer and scheduler."""

import torch.nn as nn
import torch.optim as optim

from snraware.components.sophia import SophiaG

__all__ = ["OptimScheduler"]

# -------------------------------------------------------------------------------------------------
# the transformer type models will be better trained if the weight decay is not performed
# on some parameters, such as bias, normalization etc.
# This class will split the parameters into those that will and won't use weight decay, as two parameter groups


class OptimScheduler:
    """
    Create optimizer and scheduler.
    self.optim holds the optimizer.
    self.sched holds the learning rate scheduler.
    """

    def __init__(self, config, model, total_num_steps):
        """
        @args:
            - model (torch.nn.module): model containing pre/backbone/post module we aim to optimize
            - total_num_steps (int): total number of steps in training, used for OneCycleLR.
        """
        super().__init__()

        # Save vars
        self.config = config
        self.model = model
        self.total_num_steps = total_num_steps

        self.optim = None
        self.sched = None
        self.curr_epoch = 0

        self.set_up_optim_and_scheduling(total_updates=self.total_num_steps)

    def split_decay_optim_groups(self, module, lr=-1, wd=0.0):
        """
        This function splits up parameter groups into those that will and those that won"t experience weight decay
        Adapted from mingpt: https://github.com/karpathy/minGPT
        @args:
            - module (torch module): module to create weight decay parameter groups for
            - lr (float): learning rate for the module m
            - weight_decay (float, from config): weight decay coefficient for regularization.
        """
        # separate out all parameters to those that will and won"t experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (nn.Linear, nn.Conv2d, nn.Conv3d)
        blacklist_weight_modules = (
            nn.LayerNorm,
            nn.BatchNorm2d,
            nn.BatchNorm3d,
            nn.InstanceNorm2d,
            nn.InstanceNorm3d,
            nn.parameter.Parameter,
        )
        for mn, m in module.named_modules():
            for pn, _p in m.named_parameters():
                fpn = f"{mn}.{pn}" if mn else pn  # full param name

                # random note: because named_modules and named_parameters are recursive
                # we will see the same tensors p many many times. but doing it this way
                # allows us to know which parent module any tensor p belongs to...
                if len([sub_m for sub_mn, sub_m in m.named_modules()]) == 1 or pn.endswith(
                    "relative_position_bias_table"
                ):
                    if pn.endswith("bias"):
                        # all biases will not be decayed
                        no_decay.add(fpn)
                    elif pn.endswith("weight") and isinstance(m, whitelist_weight_modules):
                        # weights of whitelist modules will be weight decayed
                        decay.add(fpn)
                    elif pn.endswith("weight") and isinstance(m, blacklist_weight_modules):
                        # weights of blacklist modules will NOT be weight decayed
                        no_decay.add(fpn)
                    elif pn.endswith("relative_position_bias_table"):
                        no_decay.add(fpn)
                    else:
                        no_decay.add(fpn)

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in module.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, (
            f"parameters {inter_params!s} made it into both decay/no_decay sets!"
        )
        assert len(param_dict.keys() - union_params) == 0, (
            f"parameters {param_dict.keys() - union_params!s} were not separated into either decay/no_decay set!"
        )

        # create the pytorch optimizer object
        optim_groups = [
            {
                "params": [param_dict[pn] for pn in sorted(list(decay))],
                "weight_decay": wd,
            },  # With weight decay group
            {
                "params": [param_dict[pn] for pn in sorted(list(no_decay))],
                "weight_decay": 0.0,
            },  # Without weight decay group
        ]

        if lr >= 0:
            optim_groups[0]["lr"] = lr
            optim_groups[1]["lr"] = lr

        return optim_groups

    def configure_optim_groups(self):
        """This function splits up pre, backbone, and post parameters into different parameter groups."""
        optim_groups = self.split_decay_optim_groups(
            self.model, lr=self.config.optim.lr, wd=self.config.optim.weight_decay
        )

        return optim_groups

    def set_up_optim_and_scheduling(self, total_updates=1):
        """
        Sets up the optimizer and the learning rate scheduler using the config
        @args:
            - total_updates (int): total number of updates in training (used for OneCycleLR)
        @outputs:
            - self.optim: optimizer
            - self.sched: scheduler.
        """
        c = self.config

        optim_groups = self.configure_optim_groups()

        self.optim = SophiaG(
            optim_groups,
            betas=(c.optim.beta1, c.optim.beta2),
            rho=c.optim.rho,
            weight_decay=c.optim.weight_decay,
        )

        # get the list of max_lr
        max_lrs = [t["lr"] for t in optim_groups]
        self.sched = optim.lr_scheduler.OneCycleLR(
            self.optim,
            max_lr=max_lrs,
            total_steps=total_updates,
            pct_start=c.scheduler.pct_start,
            anneal_strategy=c.scheduler.anneal_strategy,
            div_factor=c.scheduler.div_factor,
        )

    def report_lr(self):
        lr_record = []
        for ind, g in enumerate(self.optim.param_groups):
            lr_record.append({ind: g["lr"]})
        return lr_record


# -------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    pass
