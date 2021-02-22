"""Top-level package for entity-embed."""
import logging

# libgomp issue, must import n2 before torch. See: https://github.com/kakao/n2/issues/42
import n2

from .data_utils.numericalizer import default_tokenizer
from .entity_embed import *

__author__ = """Flávio Juvenal"""
__email__ = "flavio@vinta.com.br"
__version__ = "0.0.1"

# Good practice: https://docs.python-guide.org/writing/logging/#logging-in-a-library
logging.getLogger(__name__).addHandler(logging.NullHandler())
