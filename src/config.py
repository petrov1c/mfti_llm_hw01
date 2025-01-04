from typing import List

from omegaconf import OmegaConf
from pydantic import BaseModel


class LossConfig(BaseModel):
    name: str
    weight: float
    loss_fn: str
    loss_kwargs: dict


class DataConfig(BaseModel):
    batch_size: int
    n_workers: int
    labels: List[str]


class Config(BaseModel):
    task_name: str
    output_dir: str
    data_config: DataConfig
    n_epochs: int
    monitor_metric: str
    model_kwargs: dict
    accelerator: str
    optimizer: str
    optimizer_kwargs: dict
    scheduler: str
    scheduler_kwargs: dict
    deepspeed_config: str
    losses: List[LossConfig]

    @classmethod
    def from_yaml(cls, path: str) -> 'Config':
        cfg = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
        return cls(**cfg)
