from functools import partial

import transformers
from timm import create_model
from torchvision import models

NAME2MODEL_LANGUAGE = {
    "bert": [transformers.FlatxBertModel, transformers.BertTokenizer, "bert-base-uncased"],
    "roberta": [
        transformers.FlaxRobertaModel,
        transformers.RobertaTokenizer,
        "roberta-base",
    ],
    "distilbert": [
        transformers.DistilBertModel,
        transformers.DistilBertTokenizer,
        "distilbert-base-uncased",
    ],
    "albert": [
        transformers.AlbertModel,
        transformers.AlbertTokenizer,
        "albert-base-v2",
    ],
}

NAME2MODEL_VISION = {
    "vit32": partial(create_model, "vit_base_patch32_224"),
    "efficientnet": partial(create_model, "efficientnetv2_m"),
    "efficientnet_lite": partial(create_model, "efficientnet_lite1"),
    "gernet": partial(create_model, "gernet_m"),
    "resnet18": models.resnet18,
    "alexnet": models.alexnet,
    "squeezenet": models.squeezenet1_0,
    "vgg16": models.vgg16,
    "densenet": models.densenet161,
    "inception": models.inception_v3,
    "googlenet": models.googlenet,
    "shufflenet": models.shufflenet_v2_x1_0,
    "mobilenet_v2": models.quantization.mobilenet_v2,
    "resnext50_32x4d": models.resnext50_32x4d,
    "wide_resnet50_2": models.wide_resnet50_2,
    "mnasnet": models.mnasnet1_0,
}
