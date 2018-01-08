# -*- coding: utf-8 -*-
# Copyright (c) 2018 João Martins
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

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
