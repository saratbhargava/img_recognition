from build import logging
from tqdm import tqdm
from yacs.config import CfgNode

from img_recognition.config import cfg
from img_recognition.engine.simple_trainer import train_fn
from img_recognition.engine.simple_inference import test_fn
from img_recognition.utils.logger import setup_logger

logger = logging.getLogger(__name__)


def train(config: CfgNode):
    
    device = cfg.MODEL.DEVICE

    model = build_model(config)
    
    optimizer = make_optimizer(config, model)
    
    train_loader = make_data_loader(config, is_train=True)
    val_loader = make_data_loader(config, is_train=False)

    for epoch in tqdm(range(config['num_epochs'])):
        print("epoch: " + str(epoch), end=' ')
        logger.info("epoch: {}".format(epoch))
        train_fn(model, train_dataloader, loss_fn, optimizer, epoch, config['device'])
        test_fn(model, valid_dataloader, epoch, config['device'], dataset="valid")    
        print()

if __name__ == "__main__":
    logger = setup_logger()
    train(cfg)
    print("Done!")