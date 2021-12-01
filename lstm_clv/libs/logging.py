import logging


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


def _set_logging() -> None:

    log_format = logging.Formatter("%(asctime)s %(name)s [%(levelname)s] %(message)s")
    logger = logging.getLogger("")
    logger.setLevel(logging.INFO)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(log_format)
    logger.addHandler(console_handler)


_set_logging()  # set once per session
