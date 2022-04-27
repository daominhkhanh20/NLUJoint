import logging
import sys

logger = logging.getLogger(__name__)
stream_handler = logging.StreamHandler(sys.stdout)

formatter = logging.Formatter(
    "{asctime} {levelname} {name}:{lineno} - {message}", style="{"
)
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)
logger.setLevel(logging.INFO)
logger.propagate = False