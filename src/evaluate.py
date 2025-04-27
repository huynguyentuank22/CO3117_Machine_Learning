import hydra, torch
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate, to_absolute_path
from tqdm import tqdm

from src.data import build_test_loader
from src.utils import set_all_seed# same helpers as before

from src.models.lstm import SentimentLSTM
from src.models.gru import SentimentGRU
from src.models.rnn import SentimentSimpleRNN

# src/main.py
import logging

# TODO: configure hydra for evaluation

@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", force=True)
    logger = logging.getLogger(__name__)
    set_all_seed(cfg.seed)

    # DATA -----------------------------------------------------------------
    test_dataloader, vocab = build_test_loader(cfg.data)

    # MODEL + OPTIM --------------------------------------------------------
    # cfg.model.vocab_size = len(vocab)
    logger.info(f"Loading model {cfg.model._target_} with vocab size{len(vocab)}...")
    model = SentimentLSTM(
        vocab_size=len(vocab),
        embed_dim=cfg.model.embed_dim,
        hidden_dim=cfg.model.hidden_dim,
        num_layers=cfg.model.num_layers,
        bidir=cfg.model.bidir,
        dropout=cfg.model.dropout
    )
    
    # model = SentimentGRU(
    #     vocab_size=len(vocab),
    #     embed_dim=cfg.model.embed_dim,
    #     hidden_dim=cfg.model.hidden_dim,
    #     num_layers=cfg.model.num_layers,
    #     bidir=cfg.model.bidir,
    #     dropout=cfg.model.dropout
    # )
    
    # model = SentimentSimpleRNN(
    #     vocab_size=len(vocab),
    #     embed_dim=cfg.model.embed_dim,
    #     hidden_dim=cfg.model.hidden_dim,
    #     num_layers=cfg.model.num_layers,
    #     bidir=cfg.model.bidir,
    #     dropout=cfg.model.dropout
    # )
    
    model.eval()
    
    # Load the model state
    checkpoint_path = "checkpoints/best_model_epoch_1.pt"
    # checkpoint_path = "checkpoints/best_SentimentGRU_epoch_2.pt"
    # checkpoint_path = "checkpoints/best_SentimentSimpleRNN_epoch_2.pt"
    # checkpoint_path = "checkpoints/best_SentimentSimpleRNN_epoch_8.pt"
    state_dict = torch.load(checkpoint_path)
    model.load_state_dict(state_dict['model_state_dict'])
    model.to(cfg.device)
    
    # Evaluate the model
    logger.info(f"Evaluating model {cfg.model._target_}...")
    total_acc = 0
    total_sentences = 0
    for (inputs, labels) in tqdm(test_dataloader):
        inputs, labels = inputs.to(cfg.device), labels.to(cfg.device)
        
        with torch.no_grad():
            outputs = model(inputs)
            print(f"outputs: {outputs}, labels: {labels}")
            total_acc += ((torch.sigmoid(outputs) >= 0.5) == labels).sum().item()
            print(f"total_acc: {total_acc}")
            total_sentences += labels.size(0)
    accuracy = total_acc / total_sentences
    
    logger.info(f"Accuracy: {accuracy}...")
    
if __name__ == "__main__":
    main()
