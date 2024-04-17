import logging

from asgi_correlation_id import CorrelationIdFilter

from src.config.core import settings


class CustomFormatter(logging.Formatter):

    green = "\x1b[0;32m"
    grey = "\x1b[38;5;248m"
    yellow = "\x1b[38;5;229m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    blue = "\x1b[38;5;31m"
    white = "\x1b[38;5;255m"
    reset = "\x1b[38;5;15m"
    
    base_format = f"{grey}%(asctime)s | %(name)s | %(correlation_id)s | {{level_color}}%(levelname)-8s{grey} | {blue}%(module)s:%(lineno)d{grey} - {white}%(message)s"
    
    FORMATS = {
        logging.INFO: base_format.format(level_color=green),
        logging.WARNING: base_format.format(level_color=yellow),
        logging.ERROR: base_format.format(level_color=red),
        logging.CRITICAL: base_format.format(level_color=bold_red),
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


def custom_logger(app_name="APP"):
    logger = logging.getLogger(app_name)
    logger.setLevel(logging.DEBUG)

    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    
    ch.setFormatter(CustomFormatter())
    
    cid_filter = CorrelationIdFilter()
    ch.addFilter(cid_filter)

    logger.addHandler(ch)
    return logger

logger = custom_logger(app_name=settings.PROJECT_NAME)