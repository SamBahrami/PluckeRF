# Copyright (c) 2023-2024, Zexin He
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import os
import time
import math
import argparse
import shutil
import torch
import safetensors
from omegaconf import OmegaConf
from abc import abstractmethod
from contextlib import contextmanager
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DistributedDataParallelKwargs, ProjectConfiguration, set_seed

from openlrm.utils.logging import configure_logger
from openlrm.utils.compile import configure_dynamo
from openlrm.runners.abstract import Runner


logger = get_logger(__name__)


def parse_configs():
    # Define argparse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./assets/config.yaml')
    args, unknown = parser.parse_known_args()

    # Load configuration file
    cfg = OmegaConf.load(args.config)

    # Override with command-line arguments
    cli_cfg = OmegaConf.from_cli(unknown)
    cfg = OmegaConf.merge(cfg, cli_cfg)

    return cfg


class Trainer(Runner):

    def __init__(self):
        super().__init__()

        self.cfg = parse_configs()
        self.timestamp = time.strftime("%Y%m%d-%H%M%S")

        self.accelerator = Accelerator(
            mixed_precision=self.cfg.train.mixed_precision,
            gradient_accumulation_steps=self.cfg.train.accum_steps,
            log_with=tuple(self.cfg.logger.trackers),
            project_config=ProjectConfiguration(
                logging_dir=self.cfg.logger.tracker_root,
            ),
            use_seedable_sampler=True,
            kwargs_handlers=[
                DistributedDataParallelKwargs(
                    find_unused_parameters=self.cfg.train.find_unused_parameters,
                ),
            ],
        )
        set_seed(self.cfg.experiment.seed, device_specific=True)
        with self.accelerator.main_process_first():
            configure_logger(
                stream_level=self.cfg.logger.stream_level,
                log_level=self.cfg.logger.log_level,
                file_path=os.path.join(
                    self.cfg.logger.log_root,
                    self.cfg.experiment.parent, self.cfg.experiment.child,
                    f"{self.timestamp}.log",
                ) if self.accelerator.is_main_process else None,
            )
        logger.info(self.accelerator.state, main_process_only=False, in_order=True)
        configure_dynamo(dict(self.cfg.compile))

        # attributes with defaults
        self.model : torch.nn.Module = None
        self.optimizer: torch.optim.Optimizer = None
        self.scheduler: torch.optim.lr_scheduler.LRScheduler = None
        self.train_loader: torch.utils.data.DataLoader = None
        self.val_loader: torch.utils.data.DataLoader = None
        self.N_max_global_steps: int = None
        self.N_global_steps_per_epoch: int = None
        self.global_step: int = 0
        self.current_epoch: int = 0

    def __enter__(self):
        self.accelerator.init_trackers(
            project_name=f"{self.cfg.experiment.parent}",
            # init_kwargs={"wandb": {"name": self.cfg.experiment.child}},
        )
        self.prepare_everything()
        self.log_inital_info()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.accelerator.end_training()

    @staticmethod
    def control(option: str = None, synchronized: bool = False):
        def decorator(func):
            def wrapper(self, *args, **kwargs):
                if option is None or hasattr(self.accelerator, option):
                    accelerated_func = getattr(self.accelerator, option)(func) if option is not None else func
                    result = accelerated_func(self, *args, **kwargs)
                    if synchronized:
                        self.accelerator.wait_for_everyone()
                    return result
                else:
                    raise AttributeError(f"Accelerator has no attribute {option}")
            return wrapper
        return decorator

    @contextmanager
    def exec_in_order(self):
        for rank in range(self.accelerator.num_processes):
            try:
                if self.accelerator.process_index == rank:
                    yield
            finally:
                self.accelerator.wait_for_everyone()

    @property
    def device(self):
        return self.accelerator.device

    @property
    def is_distributed(self) -> bool:
        return self.accelerator.num_processes > 1

    def prepare_everything(self, is_dist_validation: bool = True):
        # prepare with accelerator
        if is_dist_validation:
            self.model, self.optimizer, self.train_loader, self.val_loader = \
                self.accelerator.prepare(
                    self.model, self.optimizer, self.train_loader, self.val_loader,
                )
        else:
            self.model, self.optimizer, self.train_loader = \
                self.accelerator.prepare(
                    self.model, self.optimizer, self.train_loader,
                )
        self.accelerator.register_for_checkpointing(self.scheduler)
        
        # Create a simple state object to hold global_step for checkpointing
        class TrainingState:
            def __init__(self):
                self.global_step = 0
                
            def state_dict(self):
                return {'global_step': self.global_step}
                
            def load_state_dict(self, state_dict):
                self.global_step = state_dict.get('global_step', 0)
                
        self.training_state = TrainingState()
        self.training_state.global_step = self.global_step  # Sync initial state
        self.accelerator.register_for_checkpointing(self.training_state)
        
        # prepare stats
        N_total_batch_size = self.cfg.train.batch_size * self.accelerator.num_processes * self.cfg.train.accum_steps
        self.N_global_steps_per_epoch = math.ceil(len(self.train_loader) / self.cfg.train.accum_steps)
        if 'training_iterations' in self.cfg.train:
            if self.cfg.train.training_iterations is not None:
                self.N_max_global_steps = self.cfg.train.training_iterations
        else:
            self.N_max_global_steps = self.N_global_steps_per_epoch * self.cfg.train.epochs
        if self.cfg.train.debug_global_steps is not None:
            logger.warning(f"Overriding max global steps from {self.N_max_global_steps} to {self.cfg.train.debug_global_steps}")
            self.N_max_global_steps = self.cfg.train.debug_global_steps
        logger.info(f"======== Statistics ========")
        logger.info(f"** N_max_global_steps: {self.N_max_global_steps}")
        logger.info(f"** N_total_batch_size: {N_total_batch_size}")
        logger.info(f"** N_epochs: {self.cfg.train.epochs}")
        logger.info(f"** N_global_steps_per_epoch: {self.N_global_steps_per_epoch}")
        logger.debug(f"** Prepared loader length: {len(self.train_loader)}")
        logger.info(f"** Distributed validation: {is_dist_validation}")
        logger.info(f"============================")
        logger.info(f"======== Trainable parameters ========")
        logger.info(f"** Total: {sum(p.numel() for p in self.model.parameters() if p.requires_grad)}")
        for sub_name, sub_module in self.accelerator.unwrap_model(self.model).named_children():
            logger.info(f"** {sub_name}: {sum(p.numel() for p in sub_module.parameters() if p.requires_grad)}")
        logger.info(f"=====================================")
        self.accelerator.wait_for_everyone()
        # load checkpoint or model
        self.load_ckpt_or_auto_resume_(self.cfg)
        # register hooks
        self.register_hooks()

    @abstractmethod
    def register_hooks(self):
        pass

    def auto_resume_(self, cfg) -> bool:
        ckpt_root = os.path.join(
            cfg.saver.checkpoint_root,
            cfg.experiment.parent, cfg.experiment.child,
        )
        if not os.path.exists(ckpt_root):
            return False
        ckpt_dirs = os.listdir(ckpt_root)
        if len(ckpt_dirs) == 0:
            return False
        ckpt_dirs.sort()
        latest_ckpt = ckpt_dirs[-1]
        latest_ckpt_dir = os.path.join(ckpt_root, latest_ckpt)
        logger.info(f"======== Auto-resume from {latest_ckpt_dir} ========")
        
        # Extract step from directory name for backward compatibility
        step = int(latest_ckpt)
        
        # Set accelerator step before loading to avoid KeyError
        self.accelerator.step = step
        
        try:
            self.accelerator.load_state(latest_ckpt_dir)
        except KeyError as e:
            if 'step' in str(e):
                logger.warning(f"Old checkpoint format detected, manually setting step to {step}")
                # For old checkpoints, we'll just continue with the manually set step
                pass
            else:
                raise e
        
        # Handle backward compatibility: old checkpoints don't have training_state
        if self.training_state.global_step > 0:
            # New checkpoint with training_state - use it if it's valid
            self.global_step = self.training_state.global_step
            logger.info(f"======== Loaded global_step from training_state: {self.global_step} ========")
        else:
            # Old checkpoint without training_state - use the step we extracted from directory
            self.global_step = step
            self.training_state.global_step = step  # Sync the training state
            logger.info(f"======== Inferred global_step from checkpoint directory: {self.global_step} ========")
        
        self.current_epoch = self.global_step // self.N_global_steps_per_epoch
        logger.info(f"======== Resumed at global step {self.global_step}, epoch {self.current_epoch} ========")
        return True

    def load_model_(self, cfg):
        logger.info(f"======== Loading model from {cfg.saver.load_model} ========")
        safetensors.torch.load_model(
            self.accelerator.unwrap_model(self.model),
            cfg.saver.load_model,
            strict=True,
        )
        logger.info(f"======== Model loaded ========")

    @control(synchronized=True)
    def load_ckpt_or_auto_resume_(self, cfg):
        # auto resume has higher priority, load model from path if auto resume is not available
        # cfg.saver.auto_resume and cfg.saver.load_model
        if cfg.saver.auto_resume:
            successful_resume = self.auto_resume_(cfg)
            if successful_resume:
                return
        if cfg.saver.load_model:
            successful_load = self.load_model_(cfg)
            if successful_load:
                return
        logger.debug(f"======== No checkpoint or model is loaded ========")

    @control('on_main_process', synchronized=True)
    def save_checkpoint(self):
        save_points = [1000, 2000, 4000, 8000, 16000, 32000, 64000, 128000, 256000, 384000, 512000]
        if self.global_step % 10000 == 0 or self.global_step in save_points or self.global_step == self.cfg.train.training_iterations:
            # Sync training state before saving
            self.training_state.global_step = self.global_step
            ckpt_dir = os.path.join(
                self.cfg.saver.checkpoint_root,
                self.cfg.experiment.parent, self.cfg.experiment.child,
                f"{self.global_step:06d}",
            )
            self.accelerator.save_state(output_dir=ckpt_dir, safe_serialization=True)
            logger.info(f"======== Saved checkpoint at global step {self.global_step} ========")

    @property
    def global_step_in_epoch(self):
        return self.global_step % self.N_global_steps_per_epoch

    @abstractmethod
    def _build_model(self):
        pass

    @abstractmethod
    def _build_optimizer(self):
        pass

    @abstractmethod
    def _build_scheduler(self):
        pass

    @abstractmethod
    def _build_dataloader(self):
        pass

    @abstractmethod
    def _build_loss_fn(self):
        pass

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def evaluate(self):
        pass

    def _get_str_progress(self, epoch: int = None, step: int = None):
        if epoch is not None:
            log_type = 'epoch'
            log_progress = epoch * self.N_global_steps_per_epoch
        elif step is not None:
            log_type = 'step'
            log_progress = step
        else:
            raise ValueError('Either epoch or step must be provided')
        return log_type, log_progress

    @control('on_main_process')
    def log_scalar_kwargs(self, epoch: int = None, step: int = None, split: str = None, **scalar_kwargs):
        log_type, log_progress = self._get_str_progress(epoch, step)
        split = f'/{split}' if split else ''
        for key, value in scalar_kwargs.items():
            self.accelerator.log({f'{key}{split}/{log_type}': value}, log_progress)

    @control('on_main_process')
    def log_gamma_scalars(self, epoch: int = None, step: int = None, split: str = None, **gamma_kwargs):
        log_type, log_progress = self._get_str_progress(epoch, step)
        split = f'/{split}' if split else ''
        for key, value in gamma_kwargs.items():
            # get last part of key, reverse split from _
            group_id = key.split('_', 1)[-1]
            self.accelerator.log({f'gamma/{group_id}': value}, log_progress)

    @control('on_main_process')
    def log_images(self, values: dict, step: int | None = None, log_kwargs: dict | None = {}):
        for tracker in self.accelerator.trackers:
            if hasattr(tracker, 'log_images'):
                tracker.log_images(values, step=step, **log_kwargs.get(tracker.name, {}))

    @control('on_main_process')
    def log_optimizer(self, epoch: int = None, step: int = None, attrs: list[str] = [], group_ids: list[int] = []):
        log_type, log_progress = self._get_str_progress(epoch, step)
        assert self.optimizer is not None, 'Optimizer is not initialized'
        if not attrs:
            logger.warning('No optimizer attributes are provided, nothing will be logged')
        if not group_ids:
            logger.warning('No optimizer group ids are provided, nothing will be logged')
        for attr in attrs:
            assert attr in ['lr', 'momentum', 'weight_decay'], f'Invalid optimizer attribute {attr}'
            for group_id in group_ids:
                self.accelerator.log({f'opt/{attr}/{group_id}': self.optimizer.param_groups[group_id][attr]}, log_progress)

    @control('on_main_process')
    def log_inital_info(self):
        assert self.model is not None, 'Model is not initialized'
        assert self.optimizer is not None, 'Optimizer is not initialized'
        assert self.scheduler is not None, 'Scheduler is not initialized'
        self.accelerator.log({'Config': "```\n" + OmegaConf.to_yaml(self.cfg) + "\n```"})
        self.accelerator.log({'Model': "```\n" + str(self.model) + "\n```"})
        self.accelerator.log({'Optimizer': "```\n" + str(self.optimizer) + "\n```"})
        self.accelerator.log({'Scheduler': "```\n" + str(self.scheduler) + "\n```"})

    def run(self):
        self.train()
