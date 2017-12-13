from models.BaseModel import BaseModel
from models.FC3_NN import FC3_NN
from models.CNN import CNN
from models.NonPaddedCNN import NonPaddedCNN
from models.WouterCNN import WouterCNN

available_models = [
    "BaseModel",
    "WouterCNN",
    "FC3_NN",
    "CNN",
    "NonPaddedCNN"
]


def make_model(config):
    if config.model_name in available_models:
        return globals()[config.model_name](config)
    else:
        raise Exception('The model name {} does not exist'.format(config.model_name))


def get_model_class(config):
    if config['model_name'] in available_models:
        return globals()[config['model_name']]
    else:
        raise Exception('The model name {} does not exist'.format(config.model_name))
