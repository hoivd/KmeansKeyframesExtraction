import logging
import os
import sys
from datetime import datetime

def get_current_timestamp():
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

def _setup_logger(name: str, level=logging.INFO, log_file: str = None) -> logging.Logger:
    """
    Khởi tạo và trả về logger:
    - Ghi ra console (stdout).
    - Ghi ra file ./log/app_YYYY-MM-DD_HH-MM-SS.log (UTF-8).
    - Tự động tạo thư mục log nếu chưa tồn tại.
    """
    # Nếu không truyền log_file thì dùng mặc định theo timestamp
    if log_file is None:
        log_file = f'./log/app_{get_current_timestamp()}.txt'

    formatter = logging.Formatter(
        fmt="%(asctime)s | <%(levelname)s-%(name)s> - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False  # tránh log trùng với root logger

    # Clear handler cũ (trường hợp chạy lại cell trong Kaggle/Colab)
    if logger.hasHandlers():
        logger.handlers.clear()

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler
    log_dir = os.path.dirname(log_file)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


# Demo chạy trên Kaggle/Colab
if __name__ == "__main__":
    logger = _setup_logger("main", level=logging.DEBUG)
    logger.debug("⚙ Debug message")
    logger.info("ℹ Info message")
    logger.warning("⚠ Warning message")
    logger.error("❌ Error message")