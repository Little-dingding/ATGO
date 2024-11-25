import logging.config
from config import *

LOGGING = {
    'version': 1,
    'formatters': {
        'default': {
            'format': '%(asctime)s [%(levelname)-8s]: %(message)s',
            'datefmt': '%Y-%m-%d %H:%M:%S'
        } 
    },
    'filters': {

    },
    'handlers': {
        'file': {
            'level': 'INFO', 
            'class': 'logging.FileHandler',
            'filename': LOG_FILE_NAME
        },
        'console': {
            'level': 'INFO',
            'formatter': 'default',
            'class': 'logging.StreamHandler'
        }
    },
    'root': {
        'level': 'INFO',
        'handlers': ['console', 'file']
    }
}

logging.config.dictConfig(LOGGING)
