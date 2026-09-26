"""Tests for structured logging setup."""

import logging

from src.utils.logger import setup_logger


def test_setup_logger_returns_logger() -> None:
    logger = setup_logger("test_module")
    assert isinstance(logger, logging.Logger)
    assert logger.name == "test_module"


def test_setup_logger_default_level() -> None:
    logger = setup_logger("test_level")
    assert logger.level == logging.INFO


def test_setup_logger_custom_level() -> None:
    logger = setup_logger("test_debug", level=logging.DEBUG)
    assert logger.level == logging.DEBUG


def test_setup_logger_has_handler() -> None:
    logger = setup_logger("test_handler")
    assert len(logger.handlers) >= 1
    assert isinstance(logger.handlers[0], logging.StreamHandler)


def test_setup_logger_no_duplicate_handlers() -> None:
    logger = setup_logger("test_no_dup")
    initial_count = len(logger.handlers)
    setup_logger("test_no_dup")
    assert len(logger.handlers) == initial_count


def test_logger_format(capfd) -> None:
    logger = setup_logger("test_format")
    logger.info("hello world")
    captured = capfd.readouterr()
    assert "test_format" in captured.out
    assert "INFO" in captured.out
    assert "hello world" in captured.out
