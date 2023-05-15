import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig
from typing import Dict, Optional


@hydra.main(config_path='conf', config_name='config')
def main(cfg: DictConfig) -> None:
    # instantiate data module
    data_module = instantiate(cfg.data)
    data_module.prepare_data()
    data_module.setup()

    # instantiate model
    model = instantiate(cfg.model)

    # instantiate trainer
    trainer = instantiate(cfg.trainer)

    # train
    trainer.fit(model, data_module)

if __name__ == '__main__':
    main()