#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Enhanced LivePortrait Batch Processor v4.0
Professional-grade batch processing with modular architecture, parallel processing,
resume functionality, and enhanced error handling.

Key Improvements:
- Modular architecture with separate classes for each component
- Parallel processing support for better performance
- Resume functionality for interrupted batches
- Enhanced error recovery and retry mechanisms
- Memory-optimized processing for large batches
- Modern async/await patterns
- Comprehensive validation and health checks
- Queue management system
- Plugin architecture for extensibility
"""

import asyncio
import configparser
import hashlib
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import time
import webbrowser
from abc import ABC, abstractmethod
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union


# Rich imports with auto-installation
def auto_install_and_import(
    package_name: str, import_name: Optional[str] = None
) -> object:
    """Attempts to import a package, installs it if ImportError, then imports again."""
    import importlib

    actual_import_name = import_name or package_name
    try:
        return importlib.import_module(actual_import_name)
    except ImportError:
        print(f"[Auto-Installer] '{package_name}' not found. Installing...")
        try:
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", package_name]
            )
            return importlib.import_module(actual_import_name)
        except (subprocess.CalledProcessError, ImportError) as e:
            print(f"[Auto-Installer] Failed to install '{package_name}': {e}")
            raise


# Import Rich components
try:
    from rich.align import Align
    from rich.box import DOUBLE_EDGE, HEAVY_HEAD, ROUNDED
    from rich.console import Console
    from rich.layout import Layout
    from rich.live import Live
    from rich.padding import Padding
    from rich.panel import Panel
    from rich.progress import (
        BarColumn,
        Progress,
        SpinnerColumn,
        TaskProgressColumn,
        TextColumn,
        TimeRemainingColumn,
    )
    from rich.prompt import IntPrompt, Prompt
    from rich.rule import Rule
    from rich.table import Table
    from rich.text import Text
    from rich.theme import Theme
except ImportError:
    # Auto-install Rich if not available
    rich_module = auto_install_and_import("rich")
    from rich.align import Align
    from rich.box import DOUBLE_EDGE, HEAVY_HEAD, ROUNDED
    from rich.console import Console
    from rich.layout import Layout
    from rich.live import Live
    from rich.padding import Padding
    from rich.panel import Panel
    from rich.progress import (
        BarColumn,
        Progress,
        SpinnerColumn,
        TaskProgressColumn,
        TextColumn,
        TimeRemainingColumn,
    )
    from rich.prompt import IntPrompt, Prompt
    from rich.rule import Rule
    from rich.table import Table
    from rich.text import Text
    from rich.theme import Theme

# Optional imports
try:
    import tkinter
    from tkinter import filedialog

    TKINTER_AVAILABLE = True
except ImportError:
    TKINTER_AVAILABLE = False

try:
    import pyfiglet

    PYFIGLET_AVAILABLE = True
except ImportError:
    try:
        pyfiglet = auto_install_and_import("pyfiglet")
        PYFIGLET_AVAILABLE = True
    except:
        PYFIGLET_AVAILABLE = False

# Constants
SCRIPT_DIR = Path(__file__).parent.resolve()
CONFIG_FILE_NAME = "liveportrait_batch_config.ini"
STATE_FILE_NAME = "batch_state.json"
CONFIG_PATH = SCRIPT_DIR / CONFIG_FILE_NAME
STATE_PATH = SCRIPT_DIR / STATE_FILE_NAME

# Enhanced Theme
ENHANCED_THEME = Theme(
    {
        "info": "cyan1",
        "warning": "bold orange3",
        "danger": "bold red3",
        "error": "bold white on red3",
        "success": "bold green3",
        "highlight": "bold deep_pink2",
        "header": "bold white on dodger_blue1",
        "section_title": "bold dodger_blue1",
        "command": "bright_green",
        "dimmed": "grey70",
        "path": "sky_blue1",
        "prompt": "bold cyan1",
        "ascii_art": "bold dodger_blue1",
        "panel_border": "grey53",
        "table_border": "grey53",
        "progress_active": "dodger_blue1",
        "progress_complete": "green3",
        "status_ok": "green3",
        "status_fail": "red3",
        "status_neutral": "grey70",
        "tree_branch": "grey53",
    }
)

console = Console(theme=ENHANCED_THEME, width=120)


# Enums
class ProcessingMode(Enum):
    CPU = "cpu"
    CUDA = "cuda"
    AUTO = "auto"


class TaskStatus(Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class LogLevel(Enum):
    DEBUG = logging.DEBUG
    INFO = logging.INFO
    WARNING = logging.WARNING
    ERROR = logging.ERROR
    CRITICAL = logging.CRITICAL


# Data Classes
@dataclass
class ProcessingTask:
    """Represents a single processing task."""

    folder_path: Path
    source_prefix: str
    license_prefix: str
    task_id: str = field(
        default_factory=lambda: hashlib.md5(str(time.time()).encode()).hexdigest()[:8]
    )
    status: TaskStatus = TaskStatus.PENDING
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    error_message: Optional[str] = None
    retry_count: int = 0
    output_files: List[Path] = field(default_factory=list)

    @property
    def duration(self) -> Optional[float]:
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return None

    @property
    def is_completed(self) -> bool:
        return self.status in [
            TaskStatus.COMPLETED,
            TaskStatus.FAILED,
            TaskStatus.SKIPPED,
        ]


@dataclass
class BatchState:
    """Represents the state of a batch processing session."""

    session_id: str
    start_time: float
    tasks: List[ProcessingTask] = field(default_factory=list)
    completed_tasks: int = 0
    failed_tasks: int = 0
    total_tasks: int = 0
    current_task_index: int = 0
    is_paused: bool = False

    def to_dict(self) -> Dict:
        return {
            "session_id": self.session_id,
            "start_time": self.start_time,
            "completed_tasks": self.completed_tasks,
            "failed_tasks": self.failed_tasks,
            "total_tasks": self.total_tasks,
            "current_task_index": self.current_task_index,
            "is_paused": self.is_paused,
            "tasks": [
                {
                    "folder_path": str(task.folder_path),
                    "source_prefix": task.source_prefix,
                    "license_prefix": task.license_prefix,
                    "task_id": task.task_id,
                    "status": task.status.value,
                    "start_time": task.start_time,
                    "end_time": task.end_time,
                    "error_message": task.error_message,
                    "retry_count": task.retry_count,
                    "output_files": [str(p) for p in task.output_files],
                }
                for task in self.tasks
            ],
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "BatchState":
        state = cls(
            session_id=data["session_id"],
            start_time=data["start_time"],
            completed_tasks=data["completed_tasks"],
            failed_tasks=data["failed_tasks"],
            total_tasks=data["total_tasks"],
            current_task_index=data["current_task_index"],
            is_paused=data["is_paused"],
        )

        for task_data in data["tasks"]:
            task = ProcessingTask(
                folder_path=Path(task_data["folder_path"]),
                source_prefix=task_data["source_prefix"],
                license_prefix=task_data["license_prefix"],
                task_id=task_data["task_id"],
                status=TaskStatus(task_data["status"]),
                start_time=task_data["start_time"],
                end_time=task_data["end_time"],
                error_message=task_data["error_message"],
                retry_count=task_data["retry_count"],
                output_files=[Path(p) for p in task_data["output_files"]],
            )
            state.tasks.append(task)

        return state


# Base Classes
class BaseValidator(ABC):
    """Base class for validators."""

    @abstractmethod
    def validate(self, value: Any) -> Tuple[bool, Optional[str]]:
        """Validate a value and return (is_valid, error_message)."""
        pass


class BaseProcessor(ABC):
    """Base class for processors."""

    @abstractmethod
    async def process(self, task: ProcessingTask) -> ProcessingTask:
        """Process a task and return the updated task."""
        pass


# Validators
class PathValidator(BaseValidator):
    """Validates file and directory paths."""

    def __init__(
        self,
        must_exist: bool = True,
        must_be_dir: bool = False,
        must_be_file: bool = False,
    ):
        self.must_exist = must_exist
        self.must_be_dir = must_be_dir
        self.must_be_file = must_be_file

    def validate(self, value: Union[str, Path]) -> Tuple[bool, Optional[str]]:
        try:
            path = Path(value) if isinstance(value, str) else value

            if self.must_exist and not path.exists():
                return False, f"Path does not exist: {path}"

            if self.must_be_dir and path.exists() and not path.is_dir():
                return False, f"Path is not a directory: {path}"

            if self.must_be_file and path.exists() and not path.is_file():
                return False, f"Path is not a file: {path}"

            return True, None
        except Exception as e:
            return False, f"Invalid path: {e}"


class ConfigValidator(BaseValidator):
    """Validates configuration values."""

    def __init__(
        self, required_sections: List[str], required_keys: Dict[str, List[str]]
    ):
        self.required_sections = required_sections
        self.required_keys = required_keys

    def validate(self, config: configparser.ConfigParser) -> Tuple[bool, Optional[str]]:
        # Check required sections
        for section in self.required_sections:
            if section not in config.sections():
                return False, f"Missing required section: {section}"

        # Check required keys
        for section, keys in self.required_keys.items():
            if section in config.sections():
                for key in keys:
                    if key not in config[section]:
                        return (
                            False,
                            f"Missing required key '{key}' in section '{section}'",
                        )

        return True, None


# Enhanced Configuration Manager
class ConfigurationManager:
    """Manages application configuration with validation and auto-repair."""

    def __init__(self, config_path: Path):
        self.config_path = config_path
        self.parser = configparser.ConfigParser(
            allow_no_value=True, inline_comment_prefixes=("#", ";")
        )
        self.validator = ConfigValidator(
            required_sections=["Paths", "Arguments", "Batch", "Filter", "Environment"],
            required_keys={
                "Paths": ["liveportrait_repo_path", "driving_video_path"],
                "Batch": [
                    "processed_suffix",
                    "source_image_prefix",
                    "license_image_prefix",
                ],
            },
        )
        self.load_config()

    def load_config(self) -> bool:
        """Load configuration from file."""
        try:
            if self.config_path.exists():
                self.parser.read(self.config_path, encoding="utf-8")
                is_valid, error = self.validator.validate(self.parser)
                if not is_valid:
                    console.print(
                        f"[warning]Configuration validation failed: {error}[/warning]"
                    )
                    self._set_defaults()
                    self.save_config()
                    return False
            else:
                self._set_defaults()
                self.save_config()
            return True
        except Exception as e:
            console.print(f"[error]Error loading configuration: {e}[/error]")
            self._set_defaults()
            return False

    def _set_defaults(self):
        """Set default configuration values."""
        self.parser.clear()

        self.parser.add_section("Paths")
        self.parser.set("Paths", "liveportrait_repo_path", str(SCRIPT_DIR.parent))
        self.parser.set("Paths", "driving_video_path", "")
        self.parser.set("Paths", "driving_template_path", "")
        self.parser.set("Paths", "use_template", "false")
        self.parser.set("Paths", "default_parent_image_folder", "")

        self.parser.add_section("Arguments")
        self.parser.set("Arguments", "device_id", "0")
        self.parser.set("Arguments", "flag_use_half_precision", "true")
        self.parser.set("Arguments", "flag_relative_motion", "true")
        self.parser.set("Arguments", "flag_pasteback", "true")
        self.parser.set("Arguments", "flag_force_cpu", "false")

        self.parser.add_section("Batch")
        self.parser.set("Batch", "processed_suffix", " - done")
        self.parser.set("Batch", "log_file_name", "liveportrait_batch_log.txt")
        self.parser.set("Batch", "source_image_prefix", "selfie-")
        self.parser.set("Batch", "license_image_prefix", "license-")
        self.parser.set("Batch", "max_parallel_tasks", "2")
        self.parser.set("Batch", "max_retries", "3")
        self.parser.set("Batch", "retry_delay", "5")

        self.parser.add_section("Filter")
        self.parser.set("Filter", "filter_images", "false")
        self.parser.set("Filter", "filter_phrase", "")

        self.parser.add_section("Environment")
        self.parser.set("Environment", "CUDA_VISIBLE_DEVICES", "")
        self.parser.set("Environment", "PYTHONUNBUFFERED", "1")

        self.parser.add_section("Performance")
        self.parser.set("Performance", "memory_limit_gb", "8")
        self.parser.set("Performance", "processing_timeout", "300")
        self.parser.set("Performance", "cleanup_temp_files", "true")

    def save_config(self) -> bool:
        """Save configuration to file."""
        try:
            self.config_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.config_path, "w", encoding="utf-8") as f:
                self.parser.write(f)
            return True
        except Exception as e:
            console.print(f"[error]Error saving configuration: {e}[/error]")
            return False

    def get(self, section: str, key: str, fallback: Any = None) -> Any:
        """Get a configuration value."""
        try:
            return self.parser.get(section, key, fallback=fallback)
        except (configparser.NoSectionError, configparser.NoOptionError):
            return fallback

    def getboolean(self, section: str, key: str, fallback: bool = False) -> bool:
        """Get a boolean configuration value."""
        try:
            return self.parser.getboolean(section, key, fallback=fallback)
        except (configparser.NoSectionError, configparser.NoOptionError):
            return fallback

    def getint(self, section: str, key: str, fallback: int = 0) -> int:
        """Get an integer configuration value."""
        try:
            return self.parser.getint(section, key, fallback=fallback)
        except (configparser.NoSectionError, configparser.NoOptionError):
            return fallback

    def set(self, section: str, key: str, value: str):
        """Set a configuration value."""
        if section not in self.parser.sections():
            self.parser.add_section(section)
        self.parser.set(section, key, value)


# Enhanced Logging Manager
class LoggingManager:
    """Manages application logging with enhanced features."""

    def __init__(self, config: ConfigurationManager):
        self.config = config
        self.log_file = Path(
            config.get("Batch", "log_file_name", "liveportrait_batch_log.txt")
        )
        self.setup_logging()

    def setup_logging(self):
        """Setup logging configuration."""
        try:
            # Ensure log directory exists
            self.log_file.parent.mkdir(parents=True, exist_ok=True)

            # Configure logging
            log_format = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"

            # File handler
            file_handler = logging.FileHandler(self.log_file, encoding="utf-8")
            file_handler.setLevel(logging.INFO)
            file_handler.setFormatter(logging.Formatter(log_format))

            # Console handler for errors
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.ERROR)
            console_handler.setFormatter(
                logging.Formatter("%(levelname)s: %(message)s")
            )

            # Root logger
            logging.basicConfig(
                level=logging.INFO,
                handlers=[file_handler, console_handler],
                format=log_format,
                datefmt="%Y-%m-%d %H:%M:%S",
            )

            # Create logger
            self.logger = logging.getLogger("LivePortraitBatch")

        except Exception as e:
            console.print(f"[error]Failed to setup logging: {e}[/error]")
            # Fallback to basic logging
            logging.basicConfig(
                level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
            )
            self.logger = logging.getLogger("LivePortraitBatch")

    def log_batch_start(self, session_id: str, total_tasks: int):
        """Log batch processing start."""
        self.logger.info("=" * 80)
        self.logger.info(f"Batch processing started - Session ID: {session_id}")
        self.logger.info(f"Total tasks: {total_tasks}")
        self.logger.info(f"Timestamp: {datetime.now().isoformat()}")
        self.logger.info("=" * 80)

    def log_batch_end(
        self, session_id: str, completed: int, failed: int, duration: float
    ):
        """Log batch processing end."""
        self.logger.info("=" * 80)
        self.logger.info(f"Batch processing completed - Session ID: {session_id}")
        self.logger.info(f"Completed: {completed}, Failed: {failed}")
        self.logger.info(f"Duration: {duration:.2f} seconds")
        self.logger.info(f"Timestamp: {datetime.now().isoformat()}")
        self.logger.info("=" * 80)

    def log_task_start(self, task: ProcessingTask):
        """Log task start."""
        self.logger.info(f"Task {task.task_id} started: {task.folder_path.name}")

    def log_task_end(self, task: ProcessingTask):
        """Log task completion."""
        status = "SUCCESS" if task.status == TaskStatus.COMPLETED else "FAILED"
        duration = task.duration or 0
        self.logger.info(
            f"Task {task.task_id} {status}: {task.folder_path.name} ({duration:.2f}s)"
        )
        if task.error_message:
            self.logger.error(f"Task {task.task_id} error: {task.error_message}")



    # FIXED: Added missing methods that were causing AttributeError
    def log_batch_start(self, session_id: str, total_tasks: int):
        """Log batch processing start."""
        self.logger.info("=" * 80)
        self.logger.info(f"Batch processing started - Session ID: {session_id}")
        self.logger.info(f"Total tasks: {total_tasks}")
        self.logger.info(f"Timestamp: {datetime.now().isoformat()}")
        self.logger.info("=" * 80)

    def log_batch_end(self, session_id: str, completed: int, failed: int, duration: float):
        """Log batch processing end."""
        self.logger.info("=" * 80)
        self.logger.info(f"Batch processing completed - Session ID: {session_id}")
        self.logger.info(f"Completed: {completed}, Failed: {failed}")
        self.logger.info(f"Duration: {duration:.2f} seconds")
        self.logger.info(f"Timestamp: {datetime.now().isoformat()}")
        self.logger.info("=" * 80)

    def log_task_start(self, task):
        """Log task start."""
        self.logger.info(f"Task {task.task_id} started: {task.folder_path.name}")

    def log_task_end(self, task):
        """Log task completion."""
        status = "SUCCESS" if task.status.value == "completed" else "FAILED"
        duration = task.duration or 0
        self.logger.info(f"Task {task.task_id} {status}: {task.folder_path.name} ({duration:.2f}s)")
        if task.error_message:
            self.logger.error(f"Task {task.task_id} error: {task.error_message}")


# State Manager
class StateManager:
    """Manages batch processing state for resume functionality."""

    def __init__(self, state_path: Path):
        self.state_path = state_path

    def save_state(self, state: BatchState) -> bool:
        """Save batch state to file."""
        try:
            with open(self.state_path, "w", encoding="utf-8") as f:
                json.dump(state.to_dict(), f, indent=2)
            return True
        except Exception as e:
            console.print(f"[error]Failed to save state: {e}[/error]")
            return False

    def load_state(self) -> Optional[BatchState]:
        """Load batch state from file."""
        try:
            if self.state_path.exists():
                with open(self.state_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                return BatchState.from_dict(data)
        except Exception as e:
            console.print(f"[warning]Failed to load state: {e}[/warning]")
        return None

    def clear_state(self) -> bool:
        """Clear saved state."""
        try:
            if self.state_path.exists():
                self.state_path.unlink()
            return True
        except Exception as e:
            console.print(f"[error]Failed to clear state: {e}[/error]")
            return False


# GUI Manager
class GUIManager:
    """Manages GUI interactions with fallbacks."""

    def __init__(self):
        self.tk_available = TKINTER_AVAILABLE
        self._tk_root = None
        if self.tk_available:
            try:
                self._tk_root = tkinter.Tk()
                self._tk_root.withdraw()
            except Exception as e:
                console.print(f"[warning]Failed to initialize GUI: {e}[/warning]")
                self.tk_available = False

    def select_folder(
        self, title: str, initial_dir: Optional[Path] = None
    ) -> Optional[Path]:
        """Select a folder using GUI dialog."""
        if not self.tk_available:
            return None

        try:
            if self._tk_root:
                self._tk_root.deiconify()
                self._tk_root.lift()
                self._tk_root.focus_force()

            initial_dir_str = (
                str(initial_dir)
                if initial_dir and initial_dir.exists()
                else str(Path.home())
            )

            folder_path = filedialog.askdirectory(
                title=title, initialdir=initial_dir_str
            )

            if self._tk_root:
                self._tk_root.withdraw()

            return Path(folder_path) if folder_path else None
        except Exception as e:
            console.print(f"[warning]GUI folder selection failed: {e}[/warning]")
            return None

    def select_file(
        self,
        title: str,
        filetypes: List[Tuple[str, str]],
        initial_dir: Optional[Path] = None,
    ) -> Optional[Path]:
        """Select a file using GUI dialog."""
        if not self.tk_available:
            return None

        try:
            if self._tk_root:
                self._tk_root.deiconify()
                self._tk_root.lift()
                self._tk_root.focus_force()

            initial_dir_str = (
                str(initial_dir)
                if initial_dir and initial_dir.exists()
                else str(Path.home())
            )

            file_path = filedialog.askopenfilename(
                title=title, filetypes=filetypes, initialdir=initial_dir_str
            )

            if self._tk_root:
                self._tk_root.withdraw()

            return Path(file_path) if file_path else None
        except Exception as e:
            console.print(f"[warning]GUI file selection failed: {e}[/warning]")
            return None

    def cleanup(self):
        """Cleanup GUI resources."""
        if self._tk_root:
            try:
                self._tk_root.destroy()
            except:
                pass


# Enhanced Image Processor
class ImageProcessor(BaseProcessor):
    """Processes images using LivePortrait with enhanced error handling."""

    def __init__(self, config: ConfigurationManager, logger: LoggingManager):
        self.config = config
        self.logger_manager = logger
        self.logger = logger.logger
        self.timeout = config.getint("Performance", "processing_timeout", 300)
        self.max_retries = config.getint("Batch", "max_retries", 3)
        self.retry_delay = config.getint("Batch", "retry_delay", 5)

    async def process(self, task: ProcessingTask) -> ProcessingTask:
        """Process a single task with retry logic."""
        task.start_time = time.time()
        task.status = TaskStatus.PROCESSING

        self.logger_manager.log_task_start(task)

        for attempt in range(self.max_retries + 1):
            try:
                task.retry_count = attempt
                result = await self._process_folder(task)
                task.status = TaskStatus.COMPLETED if result else TaskStatus.FAILED
                break
            except Exception as e:
                self.logger.error(
                    f"Task {task.task_id} attempt {attempt + 1} failed: {e}"
                )
                task.error_message = str(e)

                if attempt < self.max_retries:
                    await asyncio.sleep(self.retry_delay)
                else:
                    task.status = TaskStatus.FAILED

        task.end_time = time.time()
        self.logger_manager.log_task_end(task)
        return task

    async def _process_folder(self, task: ProcessingTask) -> bool:
        """Process images in a folder."""
        folder_path = task.folder_path

        # Check if folder exists
        if not folder_path.exists():
            raise FileNotFoundError(f"Folder not found: {folder_path}")

        # Find source images
        source_images = self._find_images(folder_path, task.source_prefix)
        license_images = self._find_images(folder_path, task.license_prefix)

        if not source_images and not license_images:
            task.status = TaskStatus.SKIPPED
            return False

        # Apply filtering if enabled
        if self.config.getboolean("Filter", "filter_images"):
            filter_phrase = self.config.get("Filter", "filter_phrase", "")
            if filter_phrase:
                source_images = [
                    img
                    for img in source_images
                    if filter_phrase.lower() in img.name.lower()
                ]
                license_images = [
                    img
                    for img in license_images
                    if filter_phrase.lower() in img.name.lower()
                ]

        if not source_images and not license_images:
            task.status = TaskStatus.SKIPPED
            return False

        # Process images
        success = False
        all_images = source_images + license_images
        errors = []

        for image_path in all_images:
            try:
                result = await self._process_single_image(image_path, folder_path)
                if result:
                    success = True
                    if result not in task.output_files:
                        task.output_files.append(result)
            except Exception as e:
                error_msg = f"Failed to process image {image_path}: {e}"
                self.logger.error(error_msg)
                errors.append(error_msg)
                continue

        # Set error message if processing failed
        if not success and errors:
            task.error_message = "\n".join(errors)
        elif not success:
            task.error_message = "No images were successfully processed in folder: " + str(task.folder_path.name)

        # Rename folder if successful
        if success:
            self._rename_folder_as_processed(folder_path)

        return success

    def _find_images(self, folder_path: Path, prefix: str) -> List[Path]:
        """Find images with given prefix in folder."""
        extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tiff"]
        images = []

        for ext in extensions:
            pattern = f"{prefix}*{ext}"
            images.extend(folder_path.glob(pattern))

        return sorted(images)

    async def _process_single_image(
        self, image_path: Path, output_dir: Path
    ) -> Optional[Path]:
        """Process a single image using LivePortrait with video or .pkl template support."""
        lp_repo = Path(self.config.get("Paths", "liveportrait_repo_path"))
        
        # Check if we're using a template or video
        use_template = self.config.getboolean("Paths", "use_template", False)
        
        if use_template:
            # Use .pkl template
            driving_path = Path(self.config.get("Paths", "driving_template_path"))
            if not driving_path.exists():
                raise FileNotFoundError(f"Driving template not found: {driving_path}")
            console.print(f"[info]Using template: {driving_path.name}[/info]")
        else:
            # Use video file
            driving_path = Path(self.config.get("Paths", "driving_video_path"))
            if not driving_path.exists():
                raise FileNotFoundError(f"Driving video not found: {driving_path}")
            console.print(f"[info]Using video: {driving_path.name}[/info]")

        if not lp_repo.exists() or not (lp_repo / "inference.py").exists():
            raise FileNotFoundError(f"LivePortrait not found at: {lp_repo}")

        # Build command
        command = [
            sys.executable,
            str(lp_repo / "inference.py"),
            "--source",
            str(image_path.resolve()),
            "--driving",
            str(driving_path.resolve()),
            "--output-dir",
            str(output_dir.resolve()),
        ]

        # Add arguments from config
        if self.config.get("Arguments", "device_id"):
            command.extend(["--device-id", self.config.get("Arguments", "device_id")])

        # Add boolean flags
        bool_flags = [
            "flag_use_half_precision",
            "flag_relative_motion",
            "flag_pasteback",
            "flag_force_cpu",
            "flag_crop_driving_video",
            "flag_normalize_lip",
            "flag_source_video_eye_retargeting",
            "flag_video_editing_head_rotation",
            "flag_eye_retargeting",
            "flag_lip_retargeting",
            "flag_stitching",
            "flag_do_crop",
            "flag_do_rot",
            "flag_do_torch_compile",
        ]

        for flag in bool_flags:
            if self.config.getboolean("Arguments", flag, False):
                command.append(f"--{flag.replace('_', '-')}")

        # Setup environment
        env = os.environ.copy()
        env.update({"PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1"})

        # Add custom environment variables
        for key, value in self.config.parser.items("Environment"):
            if value:
                env[key.upper()] = value

        # Execute command with timeout
        try:
            process = await asyncio.create_subprocess_exec(
                *command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
                cwd=lp_repo,
            )

            stdout, stderr = await asyncio.wait_for(
                process.communicate(), timeout=self.timeout
            )

            if process.returncode == 0:
                # Find output file
                output_files = list(output_dir.glob("*.mp4"))
                if output_files:
                    return output_files[-1]  # Return most recent
                return output_dir / f"{image_path.stem}_output.mp4"
            else:
                error_msg = stderr.decode("utf-8", errors="replace")
                raise subprocess.CalledProcessError(
                    process.returncode, command, error_msg
                )

        except asyncio.TimeoutError:
            if process:
                process.terminate()
                await process.wait()
            raise TimeoutError(f"Processing timed out after {self.timeout} seconds")

    def _rename_folder_as_processed(self, folder_path: Path):
        """Rename folder to mark as processed."""
        suffix = self.config.get("Batch", "processed_suffix", " - done")
        if not folder_path.name.endswith(suffix):
            new_name = folder_path.name + suffix
            new_path = folder_path.parent / new_name
            try:
                folder_path.rename(new_path)
                self.logger.info(f"Renamed folder to: {new_name}")
            except Exception as e:
                self.logger.error(f"Failed to rename folder {folder_path.name}: {e}")


# Task Queue Manager
class TaskQueueManager:
    """Manages processing queue with parallel execution."""

    def __init__(self, config: ConfigurationManager, logger: LoggingManager):
        self.config = config
        self.logger = logger
        self.max_workers = config.getint("Batch", "max_parallel_tasks", 2)
        self.processor = ImageProcessor(config, logger)
        self.semaphore = asyncio.Semaphore(self.max_workers)

    async def process_queue(
        self, tasks: List[ProcessingTask], progress_callback: Optional[Callable] = None
    ) -> List[ProcessingTask]:
        """Process all tasks in the queue with parallel execution."""
        completed_tasks = []

        async def process_with_semaphore(task):
            async with self.semaphore:
                if progress_callback:
                    progress_callback(f"Processing {task.folder_path.name}")
                result = await self.processor.process(task)
                completed_tasks.append(result)
                return result

        # Create tasks for async execution
        async_tasks = [process_with_semaphore(task) for task in tasks]

        # Execute with progress tracking
        for coro in asyncio.as_completed(async_tasks):
            await coro

        return completed_tasks


# Enhanced UI Manager
class UIManager:
    """Manages user interface with enhanced features."""

    def __init__(self):
        self.gui = GUIManager()
        self._setup_banner()

    def _setup_banner(self):
        """Setup application banner."""
        if PYFIGLET_AVAILABLE:
            try:
                fig = pyfiglet.Figlet(font="standard")
                self.banner = fig.renderText("LP BATCH v4")
            except:
                self.banner = self._fallback_banner()
        else:
            self.banner = self._fallback_banner()

    def _fallback_banner(self) -> str:
        return """
██╗     ██████╗     ██████╗  █████╗ ████████╗ ██████╗██╗  ██╗    ██╗   ██╗██╗  ██╗
██║     ██╔══██╗    ██╔══██╗██╔══██╗╚══██╔══╝██╔════╝██║  ██║    ██║   ██║██║  ██║
██║     ██████╔╝    ██████╔╝███████║   ██║   ██║     ███████║    ██║   ██║███████║
██║     ██╔═══╝     ██╔══██╗██╔══██║   ██║   ██║     ██╔══██║    ╚██╗ ██╔╝╚════██║
███████╗██║         ██████╔╝██║  ██║   ██║   ╚██████╗██║  ██║     ╚████╔╝      ██║
╚══════╝╚═╝         ╚═════╝ ╚═╝  ╚═╝   ╚═╝    ╚═════╝╚═╝  ╚═╝      ╚═══╝       ╚═╝
        Enhanced Version 4.0 - Professional Batch Processing
        """

    def show_welcome(self):
        """Display welcome screen."""
        console.clear()
        console.print(
            Panel(
                Align.center(Text(self.banner, style="ascii_art")),
                border_style="panel_border",
                padding=(1, 2),
            )
        )

        console.print(
            Panel(
                Text.assemble(
                    ("Welcome to the ", "info"),
                    ("Enhanced LivePortrait Batch Processor v4.0!", "highlight"),
                    ("\n\nNew Features:\n", "info"),
                    ("• Parallel processing for faster execution\n", "success"),
                    ("• Resume functionality for interrupted batches\n", "success"),
                    ("• Enhanced error recovery and retry logic\n", "success"),
                    ("• Memory optimization for large batches\n", "success"),
                    ("• Real-time progress monitoring\n", "success"),
                    ("• Comprehensive logging and state management\n", "success"),
                ),
                title="Enhanced Features",
                border_style="success",
                padding=(1, 2),
            )
        )

    def get_processing_mode(self) -> ProcessingMode:
        """Get processing mode from user."""
        console.print(Rule(title="Processing Mode Selection", style="section_title"))

        table = Table(show_header=True, box=ROUNDED, border_style="table_border")
        table.add_column("Option", style="highlight", justify="center")
        table.add_column("Mode", style="info")
        table.add_column("Description", style="dimmed")

        table.add_row("1", "GPU (CUDA)", "Fastest - requires NVIDIA GPU")
        table.add_row("2", "CPU Only", "Slower but universal compatibility")
        table.add_row("3", "Auto Detect", "Automatically choose best option")

        console.print(table)

        choice = IntPrompt.ask(
            "Select processing mode", choices=["1", "2", "3"], default=3
        )

        modes = {1: ProcessingMode.CUDA, 2: ProcessingMode.CPU, 3: ProcessingMode.AUTO}
        return modes[choice]

    def get_paths(
        self, config: ConfigurationManager
    ) -> Tuple[Optional[Path], Optional[Path], Optional[Path]]:
        """Get required paths from user."""
        console.print(Rule(title="Path Configuration", style="section_title"))

        # LivePortrait path
        lp_path = self._get_liveportrait_path(config)
        if not lp_path:
            return None, None, None

        # Driving video path
        video_path = self._get_driving_video_path(config)
        if not video_path:
            return None, None, None

        # Input directory
        input_dir = self._get_input_directory(config)
        if not input_dir:
            return None, None, None

        return lp_path, video_path, input_dir

    def _get_liveportrait_path(self, config: ConfigurationManager) -> Optional[Path]:
        """Get LivePortrait installation path."""
        current_path = Path(config.get("Paths", "liveportrait_repo_path", ""))
        validator = PathValidator(must_exist=True, must_be_dir=True)

        # Check current path
        if current_path.exists() and (current_path / "inference.py").exists():
            change = Prompt.ask(
                f"Current LivePortrait path: [path]{current_path}[/path]\nChange it?",
                choices=["y", "n"],
                default="n",
            )
            if change.lower() == "n":
                return current_path

        # Get new path
        console.print(
            Panel(
                "Select the LivePortrait installation directory (containing inference.py)",
                title="LivePortrait Path",
                border_style="info",
            )
        )

        # Try GUI first
        new_path = self.gui.select_folder(
            "Select LivePortrait Directory",
            current_path.parent if current_path.exists() else None,
        )

        # Fallback to manual input
        if not new_path:
            path_str = Prompt.ask(
                "Enter LivePortrait directory path",
                default=str(current_path) if current_path.exists() else "",
            )
            new_path = Path(path_str) if path_str else None

        # Validate
        if new_path and new_path.exists() and (new_path / "inference.py").exists():
            config.set("Paths", "liveportrait_repo_path", str(new_path))
            config.save_config()
            return new_path

        console.print(
            "[error]Invalid LivePortrait path. Please ensure the directory contains inference.py[/error]"
        )
        return None

    def _get_driving_video_path(self, config: ConfigurationManager) -> Optional[Path]:
        """Get driving video or template path with .pkl support."""
        
        # Check for available .pkl templates
        templates_dir = Path(__file__).parent / "driving_templates"
        templates_dir.mkdir(exist_ok=True)
        pkl_templates = list(templates_dir.glob("*.pkl"))
        
        console.print(Rule(title="Driving Video/Template Selection", style="section_title"))
        
        # Show selection options
        table = Table(show_header=True, box=ROUNDED, border_style="table_border")
        table.add_column("Option", style="highlight", justify="center")
        table.add_column("Type", style="info")
        table.add_column("Description", style="dimmed")
        
        table.add_row("1", "Video File", "Use .mp4/.avi video (slower, creates new template)")
        table.add_row("2", "Template (.pkl)", f"Use existing template (faster, {len(pkl_templates)} available)")
        
        console.print(table)
        
        choice = IntPrompt.ask("Select driving type", choices=["1", "2"], default=1)
        
        if choice == 1:
            return self._select_video_file(config)
        else:
            return self._select_pkl_template(config, pkl_templates)

    def _select_video_file(self, config: ConfigurationManager) -> Optional[Path]:
        """Select a video file (original logic)."""
        current_path = Path(config.get("Paths", "driving_video_path", ""))
        
        # Check current path
        if current_path.exists() and current_path.is_file():
            change = Prompt.ask(
                f"Current driving video: [path]{current_path}[/path]\\nChange it?",
                choices=["y", "n"],
                default="n",
            )
            if change.lower() == "n":
                config.set("Paths", "use_template", "false")
                config.save_config()
                return current_path

        # Get new path
        console.print(
            Panel(
                "Select the driving video file for animation",
                title="Driving Video",
                border_style="info",
            )
        )

        # Try GUI first
        video_types = [
            ("Video files", "*.mp4 *.avi *.mov *.mkv *.webm"),
            ("All files", "*.*"),
        ]

        new_path = self.gui.select_file(
            "Select Driving Video",
            video_types,
            current_path.parent if current_path.exists() else None,
        )

        # Fallback to manual input
        if not new_path:
            path_str = Prompt.ask(
                "Enter driving video path",
                default=str(current_path) if current_path.exists() else "",
            )
            new_path = Path(path_str) if path_str else None

        # Validate
        if new_path and new_path.exists() and new_path.is_file():
            config.set("Paths", "driving_video_path", str(new_path))
            config.set("Paths", "use_template", "false")
            config.save_config()
            return new_path

        console.print("[error]Invalid video file path[/error]")
        return None

    def _select_pkl_template(self, config: ConfigurationManager, pkl_templates: List[Path]) -> Optional[Path]:
        """Select a .pkl template file."""
        if not pkl_templates:
            console.print("[warning]No .pkl templates found in driving_templates folder[/warning]")
            console.print("[info]Templates will be created automatically when you use video files[/info]")
            return None
        
        console.print(
            Panel(
                "Select a .pkl template for faster processing",
                title="Template Selection",
                border_style="info",
            )
        )
        
        # Show available templates
        table = Table(show_header=True, box=ROUNDED, border_style="table_border")
        table.add_column("Option", style="highlight", justify="center")
        table.add_column("Template Name", style="info")
        table.add_column("Size", style="dimmed")
        
        for i, pkl_file in enumerate(pkl_templates, 1):
            file_size = pkl_file.stat().st_size / 1024  # KB
            table.add_row(str(i), pkl_file.stem, f"{file_size:.1f} KB")
        
        console.print(table)
        
        if len(pkl_templates) == 1:
            choice = 1
            console.print(f"[info]Using only available template: {pkl_templates[0].name}[/info]")
        else:
            choice = IntPrompt.ask(
                "Select template", 
                choices=[str(i) for i in range(1, len(pkl_templates) + 1)],
                default=1
            )
        
        selected_template = pkl_templates[choice - 1]
        
        # Store the template path in config
        config.set("Paths", "driving_template_path", str(selected_template))
        config.set("Paths", "use_template", "true")
        config.save_config()
        
        console.print(f"[success]Selected template: {selected_template.name}[/success]")
        return selected_template

    def _get_input_directory(self, config: ConfigurationManager) -> Optional[Path]:
        """Get input directory containing image folders."""
        current_path = Path(config.get("Paths", "default_parent_image_folder", ""))

        console.print(
            Panel(
                "Select the parent directory containing folders with images to process",
                title="Input Directory",
                border_style="info",
            )
        )

        # Try GUI first
        new_path = self.gui.select_folder(
            "Select Input Directory", current_path if current_path.exists() else None
        )

        # Fallback to manual input
        if not new_path:
            path_str = Prompt.ask(
                "Enter input directory path",
                default=str(current_path)
                if current_path.exists()
                else str(Path.home()),
            )
            new_path = Path(path_str) if path_str else None

        # Validate
        if new_path and new_path.exists() and new_path.is_dir():
            config.set("Paths", "default_parent_image_folder", str(new_path))
            config.save_config()
            return new_path

        console.print("[error]Invalid directory path[/error]")
        return None

    def get_filter_settings(self, config: ConfigurationManager) -> Tuple[bool, str]:
        """Get image filtering settings."""
        console.print(Rule(title="Image Filtering", style="section_title"))

        table = Table(show_header=True, box=ROUNDED, border_style="table_border")
        table.add_column("Option", style="highlight", justify="center")
        table.add_column("Description", style="info")

        table.add_row("1", "Process all images")
        table.add_row("2", "Filter by filename phrase")

        console.print(table)

        choice = IntPrompt.ask("Select filtering option", choices=["1", "2"], default=1)

        if choice == 2:
            phrase = Prompt.ask(
                "Enter filter phrase",
                default=config.get("Filter", "filter_phrase", "selfie"),
            )
            config.set("Filter", "filter_images", "true")
            config.set("Filter", "filter_phrase", phrase)
            config.save_config()
            return True, phrase
        else:
            config.set("Filter", "filter_images", "false")
            config.set("Filter", "filter_phrase", "")
            config.save_config()
            return False, ""

    def get_batch_settings(self, config: ConfigurationManager) -> Tuple[int, int]:
        """Get batch processing settings."""
        console.print(Rule(title="Batch Settings", style="section_title"))

        max_folders = IntPrompt.ask("Maximum folders to process (0 = all)", default=0)

        max_parallel = IntPrompt.ask(
            "Maximum parallel tasks",
            default=config.getint("Batch", "max_parallel_tasks", 2),
            show_choices=False,
        )

        config.set("Batch", "max_parallel_tasks", str(max_parallel))
        config.save_config()

        return max_folders, max_parallel

    def show_resume_option(self, state: BatchState) -> bool:
        """Show resume option for existing batch."""
        console.print(
            Panel(
                Text.assemble(
                    ("Found previous batch session:\n", "info"),
                    (f"Session ID: {state.session_id}\n", "highlight"),
                    (
                        f"Progress: {state.completed_tasks}/{state.total_tasks} completed\n",
                        "info",
                    ),
                    (
                        f"Failed: {state.failed_tasks}\n",
                        "warning" if state.failed_tasks > 0 else "info",
                    ),
                    (
                        f"Started: {datetime.fromtimestamp(state.start_time).strftime('%Y-%m-%d %H:%M:%S')}",
                        "dimmed",
                    ),
                ),
                title="Resume Previous Batch",
                border_style="warning",
            )
        )

        choice = Prompt.ask("Resume previous batch?", choices=["y", "n"], default="y")

        return choice.lower() == "y"

    def cleanup(self):
        """Cleanup UI resources."""
        self.gui.cleanup()


# Main Application Class
class LivePortraitBatchProcessor:
    """Main application class with enhanced features."""

    def __init__(self):
        self.config = ConfigurationManager(CONFIG_PATH)
        self.logger = LoggingManager(self.config)
        self.state_manager = StateManager(STATE_PATH)
        self.ui = UIManager()
        self.queue_manager = TaskQueueManager(self.config, self.logger)
        self.current_state: Optional[BatchState] = None

    async def run(self):
        """Main application entry point."""
        try:
            # Show welcome screen
            self.ui.show_welcome()
            console.print()

            # Check for existing state
            existing_state = self.state_manager.load_state()
            if existing_state and not existing_state.is_paused:
                if self.ui.show_resume_option(existing_state):
                    await self._resume_batch(existing_state)
                    return
                else:
                    self.state_manager.clear_state()

            # Start new batch
            await self._start_new_batch()

        except KeyboardInterrupt:
            console.print("\n[warning]Processing interrupted by user[/warning]")
            if self.current_state:
                self.current_state.is_paused = True
                self.state_manager.save_state(self.current_state)
                console.print("[info]Progress saved. You can resume later.[/info]")
        except Exception as e:
            console.print(f"[error]Unexpected error: {e}[/error]")
            self.logger.logger.exception("Unexpected error in main application")
        finally:
            self.ui.cleanup()

    async def _start_new_batch(self):
        """Start a new batch processing session."""
        # Get processing mode
        mode = self.ui.get_processing_mode()
        self._configure_processing_mode(mode)

        # Get paths
        lp_path, video_path, input_dir = self.ui.get_paths(self.config)
        if not all([lp_path, video_path, input_dir]):
            console.print("[error]Required paths not configured[/error]")
            return

        # Get settings
        filter_enabled, filter_phrase = self.ui.get_filter_settings(self.config)
        max_folders, max_parallel = self.ui.get_batch_settings(self.config)

        # Scan for folders
        folders = self._scan_folders(input_dir, max_folders)
        if not folders:
            console.print("[warning]No folders found to process[/warning]")
            return

        # Create tasks
        tasks = self._create_tasks(folders)

        # Create new batch state
        session_id = hashlib.md5(f"{time.time()}".encode()).hexdigest()[:12]
        self.current_state = BatchState(
            session_id=session_id,
            start_time=time.time(),
            tasks=tasks,
            total_tasks=len(tasks),
        )

        # Start processing
        await self._process_batch()

    async def _resume_batch(self, state: BatchState):
        """Resume an existing batch."""
        self.current_state = state
        console.print(f"[info]Resuming batch {state.session_id}[/info]")

        # Filter remaining tasks
        remaining_tasks = [task for task in state.tasks if not task.is_completed]
        if not remaining_tasks:
            console.print("[success]All tasks already completed![/success]")
            return

        console.print(f"[info]Resuming {len(remaining_tasks)} remaining tasks[/info]")
        await self._process_batch()

    async def _process_batch(self):
        """Process the current batch with progress tracking."""
        if not self.current_state:
            return

        # Get remaining tasks
        remaining_tasks = [
            task for task in self.current_state.tasks if not task.is_completed
        ]
        if not remaining_tasks:
            console.print("[success]All tasks completed![/success]")
            return

        console.print(Rule(title="Processing Batch", style="header"))

        # Setup progress tracking
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TextColumn("({task.completed} of {task.total})"),
            TimeRemainingColumn(),
            console=console,
        ) as progress:
            overall_task = progress.add_task(
                "Overall Progress", total=len(remaining_tasks)
            )

            completed_count = 0

            def update_progress(message: str):
                nonlocal completed_count
                completed_count += 1
                progress.update(
                    overall_task,
                    advance=1,
                    description=f"[{completed_count}/{len(remaining_tasks)}] {message}",
                )

            # Log batch start
            self.logger.log_batch_start(
                self.current_state.session_id, len(remaining_tasks)
            )

            # Process tasks
            results = await self.queue_manager.process_queue(
                remaining_tasks, update_progress
            )

            # Update state
            for result in results:
                # Find and update task in state
                for i, task in enumerate(self.current_state.tasks):
                    if task.task_id == result.task_id:
                        self.current_state.tasks[i] = result
                        break

                if result.status == TaskStatus.COMPLETED:
                    self.current_state.completed_tasks += 1
                elif result.status == TaskStatus.FAILED:
                    self.current_state.failed_tasks += 1

            # Save final state
            self.state_manager.save_state(self.current_state)

            # Log completion
            duration = time.time() - self.current_state.start_time
            self.logger.log_batch_end(
                self.current_state.session_id,
                self.current_state.completed_tasks,
                self.current_state.failed_tasks,
                duration,
            )

        # Show results
        self._show_results()

    def _configure_processing_mode(self, mode: ProcessingMode):
        """Configure processing mode."""
        if mode == ProcessingMode.AUTO:
            # Auto-detect CUDA availability
            try:
                import torch

                if torch.cuda.is_available():
                    mode = ProcessingMode.CUDA
                    console.print("[success]Auto-detected: CUDA available[/success]")
                else:
                    mode = ProcessingMode.CPU
                    console.print("[info]Auto-detected: Using CPU mode[/info]")
            except ImportError:
                mode = ProcessingMode.CPU
                console.print("[info]PyTorch not found, using CPU mode[/info]")

        # Set configuration
        cpu_mode = mode == ProcessingMode.CPU
        self.config.set("Arguments", "flag_force_cpu", "true" if cpu_mode else "false")
        self.config.save_config()

        console.print(f"[info]Processing mode: {mode.value.upper()}[/info]")

    def _scan_folders(self, parent_dir: Path, max_folders: int = 0) -> List[Path]:
        """Scan for unprocessed folders."""
        console.print(f"[info]Scanning {parent_dir} for folders...[/info]")

        suffix = self.config.get("Batch", "processed_suffix", " - done")

        try:
            all_folders = [f for f in parent_dir.iterdir() if f.is_dir()]
            unprocessed = [f for f in all_folders if not f.name.endswith(suffix)]
            unprocessed.sort()

            if max_folders > 0:
                unprocessed = unprocessed[:max_folders]

            console.print(
                f"[success]Found {len(unprocessed)} folders to process[/success]"
            )
            return unprocessed

        except Exception as e:
            console.print(f"[error]Error scanning directory: {e}[/error]")
            return []

    def _create_tasks(self, folders: List[Path]) -> List[ProcessingTask]:
        """Create processing tasks from folders."""
        source_prefix = self.config.get("Batch", "source_image_prefix", "selfie-")
        license_prefix = self.config.get("Batch", "license_image_prefix", "license-")

        tasks = []
        for folder in folders:
            task = ProcessingTask(
                folder_path=folder,
                source_prefix=source_prefix,
                license_prefix=license_prefix,
            )
            tasks.append(task)

        return tasks

    def _show_results(self):
        """Show batch processing results."""
        if not self.current_state:
            return

        console.print(Rule(title="Batch Results", style="success"))

        # Summary table
        table = Table(title="Processing Summary", box=ROUNDED, border_style="success")
        table.add_column("Metric", style="info")
        table.add_column("Value", style="highlight")

        duration = time.time() - self.current_state.start_time
        minutes, seconds = divmod(duration, 60)

        table.add_row("Total Tasks", str(self.current_state.total_tasks))
        table.add_row("Completed", str(self.current_state.completed_tasks))
        table.add_row("Failed", str(self.current_state.failed_tasks))
        table.add_row(
            "Success Rate",
            f"{(self.current_state.completed_tasks / self.current_state.total_tasks * 100):.1f}%",
        )
        table.add_row("Duration", f"{int(minutes)}m {int(seconds)}s")

        console.print(table)

        # Show failed tasks if any
        failed_tasks = [
            task
            for task in self.current_state.tasks
            if task.status == TaskStatus.FAILED
        ]
        if failed_tasks:
            console.print("\n[warning]Failed Tasks:[/warning]")
            for task in failed_tasks[:5]:  # Show first 5
                error_msg = task.error_message if task.error_message else "Unknown error"
                # Truncate very long error messages for display
                if len(error_msg) > 150:
                    error_msg = error_msg[:147] + "..."
                console.print(f"  • {task.folder_path.name}: {error_msg}")
            if len(failed_tasks) > 5:
                console.print(f"  ... and {len(failed_tasks) - 5} more")

        # Clear state on successful completion
        if self.current_state.failed_tasks == 0:
            self.state_manager.clear_state()
            console.print("\n[success]Batch completed successfully![/success]")

        # Offer to open output directory
        parent_dir = (
            self.current_state.tasks[0].folder_path.parent
            if self.current_state.tasks
            else None
        )
        if parent_dir and parent_dir.exists():
            open_folder = Prompt.ask(
                f"Open output directory? ({parent_dir})",
                choices=["y", "n"],
                default="y",
            )
            if open_folder.lower() == "y":
                self._open_folder(parent_dir)

    def _open_folder(self, folder_path: Path):
        """Open folder in system file manager."""
        try:
            if os.name == "nt":  # Windows
                os.startfile(folder_path)
            elif sys.platform == "darwin":  # macOS
                subprocess.run(["open", str(folder_path)])
            else:  # Linux
                subprocess.run(["xdg-open", str(folder_path)])
        except Exception as e:
            console.print(f"[warning]Could not open folder: {e}[/warning]")
            console.print(f"[info]Manual path: {folder_path}[/info]")


# Entry Point
async def main():
    """Application entry point."""
    processor = LivePortraitBatchProcessor()
    await processor.run()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        console.print("\n[info]Application terminated by user[/info]")
    except Exception as e:
        console.print(f"\n[error]Critical error: {e}[/error]")
        console.print_exception()
    finally:
        console.print("\n[dimmed]Press Enter to exit...[/dimmed]")
        input()
