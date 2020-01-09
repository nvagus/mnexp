from .combine import *
from .cook import *
from .event import *
from .lz import *
from .news import *
from .paper import *
from .personal import *
from .seq import *
from .seq2vec import *
from .test import *
from .test_pipeline import *
from .vert import *


def get(config: settings.Config):
    return eval(config.task)(config)
