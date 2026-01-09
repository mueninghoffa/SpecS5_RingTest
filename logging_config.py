"""
Logging configuration, including filters and getter.
"""

import logging
from pathlib import Path
from typing import cast

SAFE_MODULES = {"__main__", "ueye_camera", "ueye_commands", "logging_config"}


def get_logger(name: str) -> logging.Logger:
    """
    Sets up logger with given name.

    Centralizes logic for logger getting, if it ever is changed or
    updates.

    Parameters
    ----------
    name : str
        Name of the logger to be returned.

    Returns
    -------
    logging.Logger
        Logger instance.
    """
    return logging.getLogger(name)


logger = get_logger(__name__)


class OnlyApprovedDebugs(logging.Filter):
    """
    A filter that blocks DEBUG and below messages from any logger not in the safe list.

    Parameters
    ----------
    name : str, default=''
        Name of filter instance.
    safe_modules : sequence[str], optional
        Sequence containing names of loggers whose messages of level DEBUG
        or lower are allowed through the filter.

    Attributes
    ----------
    safe_modules : set[str]
        Set containing names of loggers whose messages of level DEBUG or
        lower are allowed through the filter.

    Methods
    -------
    filter(record)
        Determines whether a log record should be allowed through.
    """

    def __init__(self, name="", safe_modules=None):
        super().__init__(name)
        # Convert to a set for fast lookup
        self.safe_modules = set(safe_modules) if safe_modules else set()

    def filter(self, record: logging.LogRecord):
        """
        Determine whether a log record should be allowed through.

        Log records with log levels 10 (DEBUG) and below (e.g. TRACE) are allowed through only if they come from a module on the `safe_modules` list.

        Parameters
        ----------
        record : `logging.LogRecord`
            Log record to be evaluated by the filter.

        Returns
        -------
        bool
            Whether the log record should be allowed to proceed (``True``)
            or not (``False``).
        """
        return (record.levelno > logging.DEBUG) or (record.name in self.safe_modules)


class FuncNameFilter(logging.Filter):
    """
    Injects a context-aware function name attribute into log records.

    If the log record was generated inside of a function, this filter adds
    the function name to the metadata prefixe of the message.

    Methods
    -------
    filter(record)
        Reassigns function name to an empty string if appropriate.
    """

    def filter(self, record):
        """
        Changes `record.funcName` to an empty string if it is ``"<module>"``.

        Parameters
        ----------
        record : logging.LogRecord
            The log record to be inspected and potentially modified.

        Return
        ------
        bool
            Always True. This filter modifies the record rather than block it.
        """

        if record.funcName == "<module>":
            record.funcName = ""
        else:
            record.funcName = "-" + record.funcName
        return True


def set_up_logging(
    log_to_console: bool = False,
    log_to_file: bool = True,
    console_log_level: str | int = logging.INFO,
    file_log_level: str | int = logging.DEBUG,
    filename: str = "./log_output.txt",
    filter: bool = True,
):
    """
    Set up console and/or file logging, along with warnings logging.

    Choose whether to log to a file and/or the console, each with their own
    log level. If both `log_to_console` and `log_to_file` are ``False``,
    the root logger is restricted to messages at or above the ``CRITICAL``
    level instead of being disabled entirely. Warnings are always logged to
    the console, and are logged to the log file if enabled.

    Paramters
    ---------
    log_to_console : bool, default=False
        Whether to log to the console.
    log_to_file : bool, default True
        Whether to log to a file.
    console_log_level : str, int, default=logging.INFO
        Log level for the console logger.
    file_log_level : str, int, default=logging.DEBUG
        Log level for the file logger.
    filename : str, default="./log_output.txt"
        Path to the log file. File will be created if it does not already
        exist.
    filter : bool, default=True
        Whether to filter out debug and below log messages from modules
        outside this package.
    """

    log_file = Path(filename)

    if not log_file.is_file():
        with open(log_file, "x") as file:
            logging.info(f"Created log file {file}")
    else:
        with open(log_file, "r") as file:
            logging.info(f"Found log file {file}")

    if isinstance(file_log_level, str):
        file_log_level = cast(
            int, getattr(logging, file_log_level.upper(), logging.DEBUG)
        )
    if isinstance(console_log_level, str):
        console_log_level = cast(
            int, getattr(logging, console_log_level.upper(), logging.DEBUG)
        )

    standard_format = logging.Formatter(
        "%(asctime)s-%(levelname)s-%(module)s-line%(lineno)d"
        + "%(funcName)s :: %(message)s"
        # a dash "-" is added to funcName in FuncNameFilter
    )
    caught_warning_format = logging.Formatter("%(asctime)s-WARNING(CAUGHT)-%(message)s")
    debug_filter = OnlyApprovedDebugs(safe_modules=SAFE_MODULES)

    root_logger = logging.getLogger()
    root_logger.setLevel(0)  # log everything
    warnings_logger = logging.getLogger("py.warnings")
    warnings_logger.setLevel(logging.WARNING)
    warnings_logger.propagate = (
        False  # so root_logger does not get already caught warning logs
    )

    # clears existing handlers to avoid duplicates
    if root_logger.hasHandlers():
        for handler in root_logger.handlers:
            handler.close()
        root_logger.handlers.clear()
    if warnings_logger.hasHandlers():
        for handler in warnings_logger.handlers:
            handler.close()
        warnings_logger.handlers.clear()

    if log_to_console:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(console_log_level)
        console_handler.setFormatter(standard_format)
        if filter:
            console_handler.addFilter(debug_filter)
        console_handler.addFilter(FuncNameFilter())
        root_logger.addHandler(console_handler)

    # always log warnings to console
    console_handler_warnings = logging.StreamHandler()
    console_handler_warnings.setLevel(logging.WARNING)
    console_handler_warnings.setFormatter(caught_warning_format)
    warnings_logger.addHandler(console_handler_warnings)

    # Check and add file handling
    if log_to_file and log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(standard_format)
        if filter:
            file_handler.addFilter(debug_filter)
        file_handler.addFilter(FuncNameFilter())
        root_logger.addHandler(file_handler)

        file_handler_warnings = logging.FileHandler(log_file)
        file_handler_warnings.setFormatter(caught_warning_format)
        warnings_logger.addHandler(file_handler_warnings)

    # Still log message that are ``CRITICAL`` and above
    if not (log_to_console or log_to_file):
        logging.disable(logging.CRITICAL)

    log_locations = []
    log_levels = []

    if log_to_console:
        log_locations.append("console")
        log_levels.append(logging.getLevelName(console_log_level))
    if log_to_file:
        log_locations.append(filename)
        log_levels.append(logging.getLevelName(file_log_level))
    if len(log_locations) == 0:
        logger.info("Logging to nowhere")
    else:
        for loc, level in zip(log_locations, log_levels):
            logger.info(f"Logging to {loc} with level {level}")
