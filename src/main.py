import hydra, torch
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate, to_absolute_path

from src.data import build_loaders
from src.utils import set_all_seed# same helpers as before

# src/main.py
import logging
# root = logging.getLogger()
# root.handlers.clear()

@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig) -> None:
    
    logging.basicConfig(level=logging.ERROR, format="%(asctime)s - %(levelname)s - %(message)s", force=True)
    logger = logging.getLogger(__name__)
    logger.info(f"Configurations: {OmegaConf.to_yaml(cfg, resolve=True)}")   # nice dump of active cfg

    set_all_seed(cfg.seed)

    # DATA -----------------------------------------------------------------
    train_dataloader, val_dataloader, vocab = build_loaders(cfg.data)

    # MODEL + OPTIM --------------------------------------------------------
    # cfg.model.vocab_size = len(vocab)
    logger.info(f"Loading model {cfg.model._target_} with vocab size{len(vocab)}...")
    model = instantiate(cfg.model, vocab_size=len(vocab)).to(cfg.device)
    logger.info(f"Instantiating optimizer: {cfg.optim._target_}...")
    optim = instantiate(cfg.optim, params=model.parameters())
    loss_fn  = torch.nn.BCEWithLogitsLoss()
    logger.info(f"Instantiating trainer: {cfg.trainer._target_}...")
    trainer = instantiate(cfg.trainer, 
                        model=model, 
                        train_loader=train_dataloader,
                        val_loader=val_dataloader,
                        criterion=loss_fn, 
                        optimizer=optim)
    
    logger.info(f"Training model {cfg.model._target_}...")
    trainer.fit()
    
    # TODO: add a test phase
    
if __name__ == "__main__":
    main()
