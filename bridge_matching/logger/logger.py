import logging
import logging.config
import json
import os
from pathlib import Path
from collections import OrderedDict


def read_json(fname):
    fname = Path(fname)
    with fname.open("rt") as handle:
        return json.load(handle, object_hook=OrderedDict)


def setup_logging(save_dir, default_level=logging.INFO):
    """
    Setup logging configuration
    """
    save_dir = Path(save_dir)
    os.makedirs(save_dir, exist_ok=True)
    log_config = Path(__file__).parent.resolve() / "logger_config.json"
    if log_config.is_file():
        config = read_json(log_config)
        # modify logging paths based on run config
        for _, handler in config["handlers"].items():
            if "filename" in handler:
                handler["filename"] = str(save_dir / handler["filename"])
        logging.config.dictConfig(config)
    else:
        print(
            "Warning: logging configuration file is not found in {}.".format(log_config)
        )
        logging.basicConfig(level=default_level)
