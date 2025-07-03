#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# pylint: disable=too-many-lines,invalid-name,line-too-long
"""LivePortrait Batch Processor with Rich UI and File Pickers
Automates batch processing of 'selfie-' and 'license-' images using LivePortrait.
Features GUI file/folder pickers and a rich terminal interface.
Reads/writes configuration from liveportrait_batch_config.ini.

UI Enhancement Summary (beyond original):
- ASCII Art Banner: pyfiglet-generated "LP BATCH" (font: 'standard'), styled and centered.
- Custom Theme: Cohesive and professional aesthetic.
- Rich Components: Consistent styling for Panels, Tables, Progress Bar.
- Pyfiglet: Dependency for banner.
- Styling Fixes: Corrected 'dimmed' text rendering.
- Critical Fix: Removed erroneous `return` from `finally` block in `process_single_image`.
- Return Logic Refinement: `process_single_image` now returns a more informative tuple:
  `(image_found_for_prefix, image_matched_filter_and_processed, processing_successful_for_this_image)`
  to allow more precise logic in `process_one_folder` for renaming.
- UI Enhancement - Tree Output: Processing steps within each folder are now displayed
  in a tree-like structure for improved clarity using Rich's Text and Padding.
- Enhanced Status Messages: More specific messages for image processing outcomes.
- Open Folder Option: Prompts to open the output folder after successful processing.
"""

import configparser
import logging
import os
import subprocess
import sys
import time
import webbrowser
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

# --- Rich, Tkinter, and Pyfiglet Imports ---
TKINTER_AVAILABLE = False
tk_filedialog = None  # Will be assigned if tkinter and filedialog are available
PYFIGLET_AVAILABLE = False
Figlet = None

try:
    import tkinter
    from tkinter import filedialog as tk_filedialog_module

    tk_filedialog = tk_filedialog_module
    TKINTER_AVAILABLE = True
except ImportError:
    pass  # Tkinter is optional


def auto_install_and_import(
    package_name: str, import_name: Optional[str] = None
) -> object:
    """Attempts to import a package, installs it if ImportError, then imports again."""
    import importlib

    actual_import_name = import_name or package_name
    try:
        return importlib.import_module(actual_import_name)
    except ImportError:
        # Use print for this feedback as Rich console might not be set up yet
        print(
            f"[Auto-Installer] '{package_name}' not found. Attempting to install with pip..."
        )
        try:
            # Add retry logic with timeout to handle network issues
            for attempt in range(3):  # Try up to 3 times
                try:
                    # Use --timeout parameter to avoid waiting too long
                    subprocess.check_call(
                        [
                            sys.executable,
                            "-m",
                            "pip",
                            "install",
                            "--timeout",
                            "30",
                            package_name,
                        ],
                        timeout=120,  # 2 minute timeout
                    )
                    return importlib.import_module(actual_import_name)
                except (subprocess.TimeoutExpired, subprocess.CalledProcessError) as e:
                    if attempt < 2:  # If not the last attempt
                        print(
                            f"[Auto-Installer] Retry {attempt + 1}/3 for '{package_name}' after error: {e}"
                        )
                        time.sleep(2)  # Wait before retry
                    else:
                        print(
                            f"[Auto-Installer] Failed to install '{package_name}' after 3 attempts: {e}"
                        )
                        raise
        except ImportError as e_after_install:
            print(
                f"[Auto-Installer] Still could not import '{actual_import_name}' after attempting install: {e_after_install}"
            )
            raise
        except Exception as e:
            print(f"[Auto-Installer] Failed to install '{package_name}': {e}")
            raise


# Attempt to import/install Rich components
try:
    Align = auto_install_and_import("rich.align", "rich.align").Align
    DOUBLE_EDGE = auto_install_and_import("rich.box", "rich.box").DOUBLE_EDGE
    HEAVY_HEAD = auto_install_and_import("rich.box", "rich.box").HEAVY_HEAD
    ROUNDED = auto_install_and_import("rich.box", "rich.box").ROUNDED
    Console = auto_install_and_import("rich.console", "rich.console").Console
    Padding = auto_install_and_import("rich.padding", "rich.padding").Padding
    Panel = auto_install_and_import("rich.panel", "rich.panel").Panel
    BarColumn = auto_install_and_import("rich.progress", "rich.progress").BarColumn
    Progress = auto_install_and_import("rich.progress", "rich.progress").Progress
    SpinnerColumn = auto_install_and_import(
        "rich.progress", "rich.progress"
    ).SpinnerColumn
    TaskProgressColumn = auto_install_and_import(
        "rich.progress", "rich.progress"
    ).TaskProgressColumn
    TextColumn = auto_install_and_import("rich.progress", "rich.progress").TextColumn
    TimeRemainingColumn = auto_install_and_import(
        "rich.progress", "rich.progress"
    ).TimeRemainingColumn
    IntPrompt = auto_install_and_import("rich.prompt", "rich.prompt").IntPrompt
    Prompt = auto_install_and_import("rich.prompt", "rich.prompt").Prompt
    Rule = auto_install_and_import("rich.rule", "rich.rule").Rule
    Table = auto_install_and_import("rich.table", "rich.table").Table
    Text = auto_install_and_import("rich.text", "rich.text").Text
    Theme = auto_install_and_import("rich.theme", "rich.theme").Theme
except Exception as e:  # Catch any exception during auto-install/import of Rich
    print(
        f"Error: The 'rich' library (or one of its components) is required but could not be "
        f"installed or imported automatically. Please install it manually:\n"
        f"pip install rich\n"
        f"Details: {e}"
    )
    sys.exit(1)


# Fallback ASCII art for when pyfiglet is not available
LP_BATCH_ASCII_ART_FALLBACK = """
██╗     ██████╗  ██████╗  █████╗ ████████╗ ██████╗ ██╗  ██╗
██║     ██╔══██╗██╔════╝ ██╔══██╗╚══██╔══╝██╔════╝ ██║  ██║
██║     ██████╔╝███████╗ ███████║   ██║   ███████╗ ███████║
██║     ██╔═══╝ ██╔═══██╗██╔══██║   ██║   ██╔═══██╗██╔══██║
███████╗██║     ╚██████╔╝██║  ██║   ██║   ╚██████╔╝██║  ██║
╚══════╝╚═╝      ╚═════╝ ╚═╝  ╚═╝   ╚═╝    ╚═════╝ ╚═╝  ╚═╝
    (Install pyfiglet for an enhanced banner)
"""

try:
    FigletModule = auto_install_and_import("pyfiglet")
    Figlet = FigletModule.Figlet
    PYFIGLET_AVAILABLE = True
except Exception:  # Catch any exception during auto-install/import of PyFiglet
    PYFIGLET_AVAILABLE = False
    print(
        "Warning: The 'pyfiglet' library is not installed or could not be imported. Displaying fallback banner.\n"
        "For an enhanced banner, you can install it using: pip install pyfiglet"
    )


# --- Global Variables & Constants ---
SCRIPT_DIR = Path(__file__).parent.resolve()
CONFIG_FILE_NAME = "liveportrait_batch_config.ini"
CONFIG_PATH = SCRIPT_DIR / CONFIG_FILE_NAME

# Generate ASCII art using pyfiglet if available
if PYFIGLET_AVAILABLE and Figlet:
    fig = Figlet(font="standard")
    LP_BATCH_ASCII_ART = fig.renderText("LP BATCH")
else:
    LP_BATCH_ASCII_ART = LP_BATCH_ASCII_ART_FALLBACK


# Rich Console Setup
custom_theme = Theme(
    {
        "info": "cyan1",
        "warning": "bold orange3",
        "danger": "bold red3",
        "error": "bold white on red3",
        "success": "bold green3",
        "highlight": "bold deep_pink2",
        "header": "bold white on dodger_blue1",
        "section_title_text": "bold dodger_blue1",
        "command": "bright_green",
        "dimmed": "grey70",
        "path": "sky_blue1",
        "prompt_style": "bold cyan1",
        "prompt_default": "dim grey70",
        "ascii_art": "bold dodger_blue1",
        "panel_border": "grey53",
        "rule_style": "grey53",
        "table_header": "bold dodger_blue1",
        "table_border": "grey53",
        "progress_desc": "cyan1",
        "progress_name": "dodger_blue1",
        "status_ok": "green3",
        "status_fail": "red3",
        "status_neutral": "grey70",
        "progress_bar_active": "dodger_blue1",
        "progress_bar_complete": "green3",
        "progress_bar_finished": "grey53",
        "tree_branch": "grey53",  # For tree connectors
    }
)
console = Console(
    theme=custom_theme, width=120, markup=True
)  # Default width, can be overridden by terminal

# Initialize the Tkinter root window (hidden) - if available
_tk_root = None
if TKINTER_AVAILABLE:
    try:
        _tk_root = tkinter.Tk()
        _tk_root.withdraw()  # Hide the root window
    except tkinter.TclError as e:
        console.print(
            f"[warning]Could not initialize Tkinter for GUI dialogs: {e}. GUI pickers will be unavailable.[/warning]"
        )
        TKINTER_AVAILABLE = False  # Disable if Tk init fails (e.g. no display)


class AppConfig:
    """Contains all the application configuration from the INI file."""

    def __init__(self, config_path: Path):
        self.config_path = config_path
        self.parser = configparser.ConfigParser(
            allow_no_value=True, inline_comment_prefixes=("#", ";")
        )
        self.load_config()

        # Set default paths
        self.liveportrait_repo_path = Path(
            self.parser.get(
                "Paths", "liveportrait_repo_path", fallback=str(SCRIPT_DIR.parent)
            )
        )
        self.driving_video_path = Path(
            self.parser.get("Paths", "driving_video_path", fallback="")
        )
        self.default_parent_image_folder = Path(
            self.parser.get("Paths", "default_parent_image_folder", fallback="")
        )

        # Get batch settings
        self.processed_suffix = self.parser.get(
            "Batch", "processed_suffix", fallback=" - done"
        )
        self.log_file_name = Path(
            self.parser.get(
                "Batch", "log_file_name", fallback="liveportrait_batch_log.txt"
            )
        )
        self.source_image_prefix = self.parser.get(
            "Batch", "source_image_prefix", fallback="selfie-"
        )
        self.license_image_prefix = self.parser.get(
            "Batch", "license_image_prefix", fallback="license-"
        )

        # Filter settings
        self.filter_images = self.parser.getboolean(
            "Filter", "filter_images", fallback=False
        )
        self.filter_phrase = self.parser.get("Filter", "filter_phrase", fallback="")

        # Set up the Python executable path - use the one that's running this script
        self.python_executable = sys.executable

        # Set up environment variables
        self.env_vars: Dict[str, str] = {}
        if "Environment" in self.parser.sections():
            for key in self.parser["Environment"]:
                value = self.parser["Environment"][key]
                if value:  # Only add if there's a value
                    self.env_vars[key.upper()] = value

    def load_config(self):
        """Loads the configuration from the INI file."""
        if self.config_path.is_file():
            try:
                self.parser.read(self.config_path, encoding="utf-8")
            except configparser.Error as e:
                console.print(
                    f"[error]Error reading config file {self.config_path}: {e}[/error]"
                )
                # Proceed with defaults if config is corrupt
                self._set_default_config()
        else:
            self._set_default_config()
            self.save_config()  # Save the new default config

    def _set_default_config(self):
        """Sets default configuration values in the parser."""
        self.parser["Paths"] = {
            "liveportrait_repo_path": str(SCRIPT_DIR.parent),
            "driving_video_path": "",
            "default_parent_image_folder": "",
        }
        self.parser["Arguments"] = {
            "device_id": "0",
            "flag_use_half_precision": "true",
            "flag_relative_motion": "true",
            "flag_pasteback": "true",
            "# flag_crop_driving_video": "false",
            "# flag_force_cpu": "false",
            "# flag_normalize_lip": "false",
            "# flag_source_video_eye_retargeting": "false",
            "# flag_video_editing_head_rotation": "false",
            "# flag_eye_retargeting": "false",
            "# flag_lip_retargeting": "false",
            "# flag_stitching": "false",
            "# flag_do_crop": "true",
            "# flag_do_rot": "true",
            "# flag_do_torch_compile": "false",
        }
        self.parser["Batch"] = {
            "processed_suffix": " - done",
            "log_file_name": "liveportrait_batch_log.txt",
            "source_image_prefix": "selfie-",
            "license_image_prefix": "license-",
        }
        self.parser["Filter"] = {
            "filter_images": "false",
            "filter_phrase": "",
        }
        self.parser["Environment"] = {
            "# Example: CUDA_VISIBLE_DEVICES": "",
            "# PYTHONUNBUFFERED": "1",
        }

    def save_config(self):
        """Saves the current configuration to the INI file."""
        try:
            with open(self.config_path, "w", encoding="utf-8") as config_file:
                self.parser.write(config_file)
        except IOError as e:
            console.print(
                f"[error]Could not save configuration to {self.config_path}: {e}[/error]"
            )


# Initialize application configuration
APP_CONFIG = AppConfig(CONFIG_PATH)


def setup_logging():
    """Configure logging to file and console."""
    log_format = "%(asctime)s [%(levelname)s] %(message)s"
    # Ensure log file can be created
    try:
        # Create parent directory if it doesn't exist (for cases where log_file_name might include a path)
        APP_CONFIG.log_file_name.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(APP_CONFIG.log_file_name, encoding="utf-8")
        logging.basicConfig(
            level=logging.INFO,
            format=log_format,
            handlers=[file_handler],
            datefmt="%Y-%m-%d %H:%M:%S",  # More detailed timestamp
        )
    except (OSError, IOError) as e:
        console.print(
            f"[error]Failed to configure logging to file {APP_CONFIG.log_file_name}: {e}[/error]"
        )
        console.print(
            "[warning]Logging will proceed to console only if file logging fails.[/warning]"
        )
        logging.basicConfig(
            level=logging.INFO, format=log_format, datefmt="%Y-%m-%d %H:%M:%S"
        )  # Fallback to basic console logging


def _styled_prompt_text(text: str) -> Text:
    return Text.assemble((text, "prompt_style"))


def _styled_panel(
    content: Union[str, Text, Panel],
    title_text: str,
    subtitle_text: str = "",
    box_style=ROUNDED,
    border_style: str = "panel_border",
    padding_val: Union[int, Tuple[int, int], Tuple[int, int, int, int]] = (1, 2),
    title_style_key: str = "section_title_text",
    expand: bool = False,
):
    """Create a styled panel with title and optional subtitle."""
    title = Text.assemble((title_text, title_style_key))
    actual_subtitle = Text(subtitle_text, style="dimmed") if subtitle_text else None

    return Panel(
        content,
        title=title,
        subtitle=actual_subtitle,
        box=box_style,
        border_style=border_style,
        padding=padding_val,
        expand=expand,
    )


def get_folder_with_gui(
    title: str, initial_dir: Optional[str] = None
) -> Optional[Path]:
    """Open a GUI folder picker dialog with the specified title."""
    if not TKINTER_AVAILABLE or not tk_filedialog:
        if (
            TKINTER_AVAILABLE is None  # Tkinter init itself failed, warn once.
            and not getattr(get_folder_with_gui, "_warned_tkinter_unavailable", False)
        ):
            console.print(
                _styled_panel(
                    "[warning]:warning: GUI dialogs not available (Tkinter/Tk display issue). Please enter paths manually.[/warning]",
                    "GUI Warning",
                    border_style="orange3",
                    expand=True,
                )
            )
            get_folder_with_gui._warned_tkinter_unavailable = True  # type: ignore
        return None

    if _tk_root:
        _tk_root.deiconify()
        _tk_root.lift()
        _tk_root.focus_force()

    if not initial_dir or not Path(initial_dir).is_dir():
        initial_dir = str(Path.home())

    folder_path_str = tk_filedialog.askdirectory(
        title=title,
        initialdir=initial_dir,
    )

    if _tk_root:
        _tk_root.withdraw()

    return Path(folder_path_str) if folder_path_str else None


def get_file_with_gui(
    title: str, filetypes: List[Tuple[str, str]], initial_dir: Optional[str] = None
) -> Optional[Path]:
    """Open a GUI file picker dialog with the specified title and file types."""
    if not TKINTER_AVAILABLE or not tk_filedialog:
        # Warning about general Tkinter unavailability is handled by get_folder_with_gui
        return None

    if _tk_root:
        _tk_root.deiconify()
        _tk_root.lift()
        _tk_root.focus_force()

    if not initial_dir or not Path(initial_dir).is_dir():
        initial_dir = str(Path.home())

    file_path_str = tk_filedialog.askopenfilename(
        title=title,
        filetypes=filetypes,
        initialdir=initial_dir,
    )

    if _tk_root:
        _tk_root.withdraw()

    return Path(file_path_str) if file_path_str else None


def check_and_set_liveportrait_path() -> None:
    """Check the LivePortrait path, prompt to change if valid, or set if invalid."""
    initial_path_is_valid = (
        APP_CONFIG.liveportrait_repo_path.is_dir()
        and (APP_CONFIG.liveportrait_repo_path / "inference.py").is_file()
    )

    should_get_new_path = False

    if initial_path_is_valid:
        console.print(
            Padding(
                Text.assemble(
                    ("Current LivePortrait path: ", "info"),
                    (str(APP_CONFIG.liveportrait_repo_path), "path"),
                ),
                (0, 0, 1, 0),
            )
        )
        change_choice = Prompt.ask(
            _styled_prompt_text("Do you want to change it?"),
            choices=["y", "n"],
            default="n",
        )
        if change_choice.lower() == "y":
            should_get_new_path = True
        else:
            console.print(
                Padding(
                    Text.assemble(
                        ("Keeping LivePortrait at: ", "info"),
                        (str(APP_CONFIG.liveportrait_repo_path), "path"),
                    ),
                    (0, 0, 1, 0),
                )
            )
            return
    else:
        should_get_new_path = True
        if (
            APP_CONFIG.liveportrait_repo_path
            and str(APP_CONFIG.liveportrait_repo_path) != "."
            and str(APP_CONFIG.liveportrait_repo_path) != ""
            and str(APP_CONFIG.liveportrait_repo_path)
            != str(SCRIPT_DIR.parent)  # Avoid warning if it's just the default fallback
        ):
            console.print(
                Padding(
                    Text.assemble(
                        ("Configured LivePortrait path '", "warning"),
                        (str(APP_CONFIG.liveportrait_repo_path), "path"),
                        (
                            "' is invalid (e.g., folder not found or 'inference.py' missing).",
                            "warning",
                        ),
                    ),
                    (0, 0, 1, 0),
                )
            )

    if should_get_new_path:
        console.line()
        console.print(
            Rule(
                title=Text(" LivePortrait Location ", style="section_title_text"),
                style="section_title_text",  # Match title style
                characters="═",  # Heavier line
            )
        )
        console.print(
            _styled_panel(
                Text.assemble(
                    (
                        "Please select the LivePortrait installation folder (which contains ",
                        "default",
                    ),
                    ("inference.py", "highlight"),
                    (").", "default"),
                ),
                "Select LivePortrait Path",
                "Required for processing",
                expand=True,
            )
        )

        gui_initial_dir = str(SCRIPT_DIR.parent)
        if APP_CONFIG.liveportrait_repo_path and str(
            APP_CONFIG.liveportrait_repo_path
        ) not in [".", ""]:
            if APP_CONFIG.liveportrait_repo_path.is_dir():
                gui_initial_dir = str(APP_CONFIG.liveportrait_repo_path)
            elif APP_CONFIG.liveportrait_repo_path.parent.is_dir():
                gui_initial_dir = str(APP_CONFIG.liveportrait_repo_path.parent)

        lp_path_candidate = get_folder_with_gui(
            "Select LivePortrait Directory", gui_initial_dir
        )

        if not lp_path_candidate:
            prompt_default_path_str = (
                str(APP_CONFIG.liveportrait_repo_path)
                if APP_CONFIG.liveportrait_repo_path
                and str(APP_CONFIG.liveportrait_repo_path) not in [".", ""]
                else str(SCRIPT_DIR.parent)
            )
            lp_path_str = Prompt.ask(
                _styled_prompt_text("Enter LivePortrait directory path"),
                default=prompt_default_path_str,
            )
            lp_path_candidate = Path(lp_path_str)

        new_path_is_valid = (
            lp_path_candidate
            and lp_path_candidate.is_dir()
            and (lp_path_candidate / "inference.py").is_file()
        )

        if new_path_is_valid:
            APP_CONFIG.liveportrait_repo_path = lp_path_candidate
            APP_CONFIG.parser.set(
                "Paths", "liveportrait_repo_path", str(lp_path_candidate)
            )
            APP_CONFIG.save_config()
            console.print(
                Text.assemble(
                    ("LivePortrait path set to: ", "success"),
                    (str(lp_path_candidate), "path"),
                )
            )
        else:
            console.print(
                Text.assemble(
                    ("The selected path '", "warning"),
                    (str(lp_path_candidate), "path"),
                    (
                        "' is invalid (folder not found or 'inference.py' missing).",
                        "warning",
                    ),
                )
            )
            if initial_path_is_valid:
                console.print(
                    Text.assemble(
                        (
                            "Reverting to previously confirmed LivePortrait path: ",
                            "info",
                        ),
                        (str(APP_CONFIG.liveportrait_repo_path), "path"),
                    )
                )
            else:
                response = Prompt.ask(
                    _styled_prompt_text(
                        "Continue with this potentially invalid path (y) or exit (n)?"
                    ),
                    choices=["y", "n"],
                    default="n",
                )
                if response.lower() == "n":
                    console.print(
                        "Exiting due to no valid LivePortrait path.", style="danger"
                    )
                    sys.exit(1)
                else:
                    APP_CONFIG.liveportrait_repo_path = lp_path_candidate
                    APP_CONFIG.parser.set(
                        "Paths", "liveportrait_repo_path", str(lp_path_candidate)
                    )
                    APP_CONFIG.save_config()
                    console.print(
                        Text.assemble(
                            (
                                "LivePortrait path set to (potentially invalid): ",
                                "warning",
                            ),
                            (str(lp_path_candidate), "path"),
                        )
                    )


def check_and_set_driving_video() -> None:
    """Check the driving video path, prompt to change if valid, or set if invalid."""
    initial_path_is_valid = APP_CONFIG.driving_video_path.is_file()

    should_get_new_path = False

    if initial_path_is_valid:
        console.print(
            Padding(
                Text.assemble(
                    ("Current driving video: ", "info"),
                    (str(APP_CONFIG.driving_video_path), "path"),
                ),
                (0, 0, 1, 0),
            )
        )
        change_choice = Prompt.ask(
            _styled_prompt_text("Do you want to change it?"),
            choices=["y", "n"],
            default="n",
        )
        if change_choice.lower() == "y":
            should_get_new_path = True
        else:
            console.print(
                Padding(
                    Text.assemble(
                        ("Keeping driving video: ", "info"),
                        (str(APP_CONFIG.driving_video_path), "path"),
                    ),
                    (0, 0, 1, 0),
                )
            )
            return
    else:
        should_get_new_path = True
        if (
            APP_CONFIG.driving_video_path
            and str(APP_CONFIG.driving_video_path) != "."
            and str(APP_CONFIG.driving_video_path) != ""
        ):
            console.print(
                Padding(
                    Text.assemble(
                        ("Configured driving video path '", "warning"),
                        (str(APP_CONFIG.driving_video_path), "path"),
                        ("' is invalid (file not found or not a file).", "warning"),
                    ),
                    (0, 0, 1, 0),
                )
            )

    if should_get_new_path:
        console.line()
        console.print(
            Rule(
                title=Text(" Driving Video Selection ", style="section_title_text"),
                style="section_title_text",
                characters="═",
            )
        )
        console.print(
            _styled_panel(
                "Please select the driving video file (e.g., .mp4, .avi).",
                "Select Driving Video",
                "Required for processing",
                expand=True,
            )
        )

        video_types = [("Video files", "*.mp4 *.avi *.mov *.mkv"), ("All files", "*.*")]

        gui_initial_dir = str(Path.home())
        if APP_CONFIG.driving_video_path and str(APP_CONFIG.driving_video_path) not in [
            ".",
            "",
        ]:
            if APP_CONFIG.driving_video_path.is_file():
                gui_initial_dir = str(APP_CONFIG.driving_video_path.parent)
            elif APP_CONFIG.driving_video_path.parent.is_dir():
                gui_initial_dir = str(APP_CONFIG.driving_video_path.parent)
        elif APP_CONFIG.liveportrait_repo_path.is_dir():  # Fallback to LP repo dir
            gui_initial_dir = str(APP_CONFIG.liveportrait_repo_path)

        video_path_candidate = get_file_with_gui(
            "Select Driving Video", video_types, gui_initial_dir
        )

        if not video_path_candidate:
            prompt_default_path_str = (
                str(APP_CONFIG.driving_video_path)
                if APP_CONFIG.driving_video_path
                and str(APP_CONFIG.driving_video_path) not in [".", ""]
                else ""
            )
            video_path_str = Prompt.ask(
                _styled_prompt_text("Enter the full path to the driving video file"),
                default=prompt_default_path_str,
            )
            video_path_candidate = Path(video_path_str) if video_path_str else Path("")

        new_path_is_valid = video_path_candidate and video_path_candidate.is_file()

        if new_path_is_valid:
            APP_CONFIG.driving_video_path = video_path_candidate
            APP_CONFIG.parser.set(
                "Paths", "driving_video_path", str(video_path_candidate)
            )
            APP_CONFIG.save_config()
            console.print(
                Text.assemble(
                    ("Driving video set to: ", "success"),
                    (str(video_path_candidate), "path"),
                )
            )
        else:
            path_str_for_msg = (
                str(video_path_candidate)
                if video_path_candidate and str(video_path_candidate) != "."
                else "No path selected or path is empty"
            )
            console.print(
                Text.assemble(
                    ("The selected video path '", "warning"),
                    (path_str_for_msg, "path"),
                    ("' is invalid (file not found or not a file).", "warning"),
                )
            )
            if initial_path_is_valid:
                console.print(
                    Text.assemble(
                        ("Reverting to previously confirmed driving video: ", "info"),
                        (str(APP_CONFIG.driving_video_path), "path"),
                    )
                )
            else:
                response = Prompt.ask(
                    _styled_prompt_text(
                        "Continue with this potentially invalid video path (y) or exit (n)?"
                    ),
                    choices=["y", "n"],
                    default="n",
                )
                if response.lower() == "n":
                    console.print(
                        "Exiting due to no valid driving video path.", style="danger"
                    )
                    sys.exit(1)
                else:
                    APP_CONFIG.driving_video_path = video_path_candidate
                    APP_CONFIG.parser.set(
                        "Paths",
                        "driving_video_path",
                        str(video_path_candidate) if video_path_candidate else "",
                    )
                    APP_CONFIG.save_config()
                    console.print(
                        Text.assemble(
                            (
                                "Driving video path set to (potentially invalid): ",
                                "warning",
                            ),
                            (path_str_for_msg, "path"),
                        )
                    )


def get_image_filter_settings() -> None:
    """Prompt the user for image filtering settings."""
    console.line()
    console.print(
        Rule(
            title=Text(" Image Filtering Options ", style="section_title_text"),
            style="section_title_text",
            characters="═",
        )
    )
    console.print(
        _styled_panel(
            "Choose whether to process all images or only those matching a specific phrase in their filename.",
            "Image Filtering Options",
            "Useful for targeting specific images",
            expand=True,
        )
    )

    table = Table(
        show_header=True,
        box=ROUNDED,
        border_style="table_border",
        header_style="table_header",
        title_style="dimmed",
        expand=True,
    )
    table.add_column("Option", style="highlight", justify="center", width=10)
    table.add_column("Description", style="info", ratio=1)
    table.add_row("1", "Process all images in each folder.")
    table.add_row(
        "2", "Only process images containing a specific phrase in their filename."
    )
    console.print(table)

    choice = IntPrompt.ask(
        _styled_prompt_text("Enter your choice"),
        choices=["1", "2"],
        default=1 if not APP_CONFIG.filter_images else 2,
    )

    if "Filter" not in APP_CONFIG.parser.sections():
        APP_CONFIG.parser.add_section("Filter")

    if choice == 2:
        filter_phrase = Prompt.ask(
            _styled_prompt_text("Enter the phrase to filter by (case-insensitive)"),
            default=APP_CONFIG.filter_phrase if APP_CONFIG.filter_phrase else "selfie",
        )
        APP_CONFIG.filter_images = True
        APP_CONFIG.filter_phrase = filter_phrase
        APP_CONFIG.parser.set("Filter", "filter_images", "true")
        APP_CONFIG.parser.set("Filter", "filter_phrase", filter_phrase)
        console.print(
            Text.assemble(
                ("Will only process images containing: ", "info"),
                (filter_phrase, "highlight"),
                (".", "info"),
            )
        )
    else:
        APP_CONFIG.filter_images = False
        APP_CONFIG.filter_phrase = ""
        APP_CONFIG.parser.set("Filter", "filter_images", "false")
        APP_CONFIG.parser.set("Filter", "filter_phrase", "")
        console.print("Will process all images in each folder.", style="info")
    APP_CONFIG.save_config()


def get_user_inputs() -> Tuple[Path, int]:
    """Prompt the user for the parent directory and number of videos to process."""
    console.line()
    console.print(
        Rule(
            title=Text(" Step 1: Input Directory ", style="section_title_text"),
            style="section_title_text",
            characters="═",
        )
    )
    console.print(
        _styled_panel(
            "Select the main directory that contains subfolders with your source images.",
            "Parent Directory for Image Subfolders",
            expand=True,
        )
    )

    initial_dir_str = (
        str(APP_CONFIG.default_parent_image_folder)
        if APP_CONFIG.default_parent_image_folder.is_dir()
        else str(Path.home())
    )
    parent_dir_path_obj = get_folder_with_gui(
        "Select Parent Directory for Image Subfolders", initial_dir_str
    )

    if not parent_dir_path_obj:
        default_path_prompt_str = (
            str(APP_CONFIG.default_parent_image_folder)
            if APP_CONFIG.default_parent_image_folder.is_dir()
            else str(Path.home())
        )
        parent_dir_str = Prompt.ask(
            _styled_prompt_text("Enter parent directory path"),
            default=default_path_prompt_str,
        )
        parent_dir_path_obj = Path(parent_dir_str)

    if not parent_dir_path_obj.is_dir():
        console.print(
            Text.assemble(
                ("Invalid directory: ", "danger"),
                (str(parent_dir_path_obj), "path"),
                (". Exiting.", "danger"),
            )
        )
        sys.exit(1)

    console.print(
        Text.assemble(
            ("Selected parent directory: ", "info"),
            (str(parent_dir_path_obj), "path"),
        )
    )
    console.line()
    console.print(
        Rule(
            title=Text(" Step 2: Batch Size ", style="section_title_text"),
            style="section_title_text",
            characters="═",
        )
    )
    console.print(
        _styled_panel(
            "Specify how many subfolders you want to process in this run.",
            "Number of Folders to Process",
            "Enter 0 to process all available unprocessed folders",
            expand=True,
        )
    )
    num_videos = IntPrompt.ask(
        _styled_prompt_text("Enter number of folders to process"),
        default=0,
        show_default=True,
    )
    return parent_dir_path_obj, num_videos


def get_unprocessed_folders(parent_dir: Path, limit: int = 0) -> List[Path]:
    """Find unprocessed folders in the parent directory.

    When filtering is enabled, this will only return folders that actually contain
    images matching the filter phrase, continuing to search until it finds enough
    matching folders or exhausts all available folders.
    """
    logging.info("Scanning for subfolders in: %s", parent_dir)
    console.print(
        Text.assemble(
            ("Scanning for subfolders in: ", "info"),
            (str(parent_dir), "path"),
        )
    )

    try:
        all_folders = [f for f in parent_dir.iterdir() if f.is_dir()]
    except OSError as e:
        console.print(f"[error]Error accessing directory {parent_dir}: {e}[/error]")
        return []

    unprocessed_folders = [
        folder
        for folder in all_folders
        if not folder.name.endswith(APP_CONFIG.processed_suffix)
    ]
    unprocessed_folders.sort()  # Sort for consistent processing order
    total_unprocessed = len(unprocessed_folders)

    # When filtering is active, pre-check folders for matching images
    if APP_CONFIG.filter_images and APP_CONFIG.filter_phrase:
        folders_with_matching_images = []
        console.print(
            Text.assemble(
                ("Checking ", "info"),
                (str(total_unprocessed), "highlight"),
                (
                    f" unprocessed folder(s) for images containing '{APP_CONFIG.filter_phrase}'...",
                    "info",
                ),
            )
        )

        # Create a progress tracker for folder scanning
        with Progress(
            SpinnerColumn(),
            TextColumn(
                "[progress_desc]Scanning folders for matching images...[/progress_desc]"
            ),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
            transient=True,
        ) as progress:
            scan_task = progress.add_task("Scanning", total=len(unprocessed_folders))

            for folder in unprocessed_folders:
                # Check if this folder contains any images matching the filter
                has_matching_images = False
                all_img_patterns = [
                    folder / "*.jpg",
                    folder / "*.jpeg",
                    folder / "*.png",
                ]
                all_images = []
                for pattern_path in all_img_patterns:
                    all_images.extend(list(folder.glob(pattern_path.name)))

                # Check if any image contains the filter phrase
                for img in all_images:
                    if APP_CONFIG.filter_phrase.lower() in img.name.lower():
                        has_matching_images = True
                        break

                if has_matching_images:
                    folders_with_matching_images.append(folder)
                    logging.info(
                        f"Folder {folder.name} contains images matching '{APP_CONFIG.filter_phrase}'"
                    )

                    # If we've found enough folders, stop searching
                    if limit > 0 and len(folders_with_matching_images) >= limit:
                        break

                progress.update(scan_task, advance=1)

        # Use only folders that have matching images
        total_matching = len(folders_with_matching_images)
        console.print(
            Text.assemble(
                ("Found ", "info"),
                (str(total_matching), "highlight"),
                (
                    f" folder(s) containing images with '{APP_CONFIG.filter_phrase}'.",
                    "info",
                ),
            )
        )

        if total_matching == 0:
            console.print(
                "[warning]No folders contain images matching your filter criteria.[/warning]"
            )
            return []

        unprocessed_folders = folders_with_matching_images

    # Apply folder limit (after filtering if active)
    total_found = len(unprocessed_folders)
    if limit > 0 and limit < total_found:
        unprocessed_folders_to_process = unprocessed_folders[:limit]
        console.print(
            Text.assemble(
                ("Processing ", "info"),
                (str(limit), "highlight"),
                (" of ", "info"),
                (str(total_found), "highlight"),
                (" available folder(s).", "info"),
            )
        )
    else:
        unprocessed_folders_to_process = unprocessed_folders
        console.print(
            Text.assemble(
                ("Processing all ", "info"),
                (str(total_found), "highlight"),
                (" available folder(s).", "info"),
            )
        )

    logging.info(
        "Selected %d folder(s) to process.",
        len(unprocessed_folders_to_process),
    )
    return unprocessed_folders_to_process


def process_single_image(
    target_folder_path: Path,
    image_prefix: str,
    process_env: Dict[str, str],
    is_last_prefix_in_folder: bool,
) -> Tuple[bool, bool, bool]:
    """Process a single image from the target folder.

    Returns:
        Tuple containing:
        - image_found (bool): True if any image was found (before filtering).
        - image_matched_filter (bool): True if an image matched the filter (or if filtering is off).
        - processing_successful (bool): True if the matched image was processed successfully by LivePortrait.
    """
    # These connectors are relative to the "Processing images..." line from process_one_folder
    tree_connector = (
        "  └─ " if is_last_prefix_in_folder else "  ├─ "
    )  # For "No image found" type messages directly under prefix group
    sub_tree_item_connector = "  │  ├─ "  # For "Processing image: X"
    sub_tree_connector = "  │  │  └─ "  # For "OK/FAIL" status of image X

    # Handle three possible cases:
    # 1. Filter is enabled with a specific phrase (find only images with that phrase)
    # 2. Process all images in folder is selected (empty prefix, no filtering)
    # 3. Use specific prefixes (selfie-, license-)

    if APP_CONFIG.filter_images and APP_CONFIG.filter_phrase:
        # Case 1: Look for ALL images matching the filter phrase
        all_img_patterns = [
            target_folder_path / "*.jpg",
            target_folder_path / "*.jpeg",
            target_folder_path / "*.png",
        ]
        all_images: List[Path] = []
        for pattern_path in all_img_patterns:
            all_images.extend(list(target_folder_path.glob(pattern_path.name)))

        if not all_images:
            msg = "No images found in folder."
            logging.info("In folder %s: %s", target_folder_path.name, msg)
            console.print(
                Text.assemble(
                    Text(tree_connector, style="tree_branch"), Text(msg, style="dimmed")
                )
            )
            return (False, False, False)

        # Filter by the user-specified phrase
        filtered_images = [
            img
            for img in all_images
            if APP_CONFIG.filter_phrase.lower() in img.name.lower()
        ]

        if not filtered_images:
            msg = f"No images matching filter '{APP_CONFIG.filter_phrase}' found in folder."
            logging.info("In folder %s: %s", target_folder_path.name, msg)
            console.print(
                Text.assemble(
                    Text(tree_connector, style="tree_branch"),
                    Text(msg, style="dimmed"),
                )
            )
            return (False, False, False)

        logging.info(
            "Folder %s: Found %d images matching filter '%s' out of %d total images",
            target_folder_path.name,
            len(filtered_images),
            APP_CONFIG.filter_phrase,
            len(all_images),
        )
        source_images = filtered_images
        image_matched_filter = True
    elif not image_prefix:  # Case 2: Process all images in folder (empty prefix)
        all_img_patterns = [
            target_folder_path / "*.jpg",
            target_folder_path / "*.jpeg",
            target_folder_path / "*.png",
        ]
        source_images: List[Path] = []
        for pattern_path in all_img_patterns:
            source_images.extend(list(target_folder_path.glob(pattern_path.name)))

        if not source_images:
            msg = "No images found in folder."
            logging.info("In folder %s: %s", target_folder_path.name, msg)
            console.print(
                Text.assemble(
                    Text(tree_connector, style="tree_branch"), Text(msg, style="dimmed")
                )
            )
            return (False, False, False)

        logging.info(
            "Folder %s: Found %d total images to process",
            target_folder_path.name,
            len(source_images),
        )
        image_matched_filter = True
    else:  # Case 3: Original behavior - use the specific prefix
        img_patterns = [
            target_folder_path / f"{image_prefix}*.jpg",
            target_folder_path / f"{image_prefix}*.jpeg",
            target_folder_path / f"{image_prefix}*.png",
        ]
        source_images: List[Path] = []
        for pattern_path in img_patterns:
            source_images.extend(list(target_folder_path.glob(pattern_path.name)))

        if not source_images:
            msg = f"No '{image_prefix}*' image found."
            logging.info("In folder %s: %s", target_folder_path.name, msg)
            console.print(
                Text.assemble(
                    Text(tree_connector, style="tree_branch"), Text(msg, style="dimmed")
                )
            )
            return (False, False, False)

        image_matched_filter = True  # No filter means all images "match" this criteria

    image_found = True
    processing_successful = False
    selected_source_image_path: Optional[Path] = None

    selected_source_image_path = source_images[0]  # Pick the first one
    if len(source_images) > 1:
        # Use tree_connector for this warning as it's a status about the prefix group
        msg_markup = f"Multiple matching images found for '{image_prefix}*'. Using: [highlight]{selected_source_image_path.name}[/highlight]"
        logging.warning(
            "Folder %s: %s",
            target_folder_path.name,
            msg_markup.replace("[highlight]", "").replace("[/highlight]", ""),
        )
        console.print(
            Text.assemble(
                Text(
                    tree_connector, style="tree_branch"
                ),  # Or a dedicated warning connector
                Text.from_markup(f":warning: {msg_markup}"),
            )
        )

    # Connector for "Processing image:" should be sub_tree_item_connector
    # If multiple images were processed in a loop here, subsequent ones might use sub_tree_item_connector too,
    # with the last one potentially using a sub_tree_item_connector variant ending in '└─'.
    # For now, only one image is processed per prefix.
    console.print(
        Text.assemble(
            Text(sub_tree_item_connector, style="tree_branch"),
            Text("Processing image: ", style="info"),
            Text(selected_source_image_path.name, style="highlight"),
        )
    )

    liveportrait_inference_script = APP_CONFIG.liveportrait_repo_path / "inference.py"
    original_cwd = Path.cwd()

    try:
        os.chdir(APP_CONFIG.liveportrait_repo_path)
        logging.info("Changed CWD to: %s", APP_CONFIG.liveportrait_repo_path)

        command = [
            APP_CONFIG.python_executable,
            str(liveportrait_inference_script.name),
            "--source",
            str(selected_source_image_path.resolve()),
            "--driving",
            str(APP_CONFIG.driving_video_path.resolve()),
            "--output-dir",
            str(target_folder_path.resolve()),
        ]

        if "Arguments" in APP_CONFIG.parser:
            for arg_key, arg_value in APP_CONFIG.parser["Arguments"].items():
                if arg_key == "device_id":
                    command.extend(["--device-id", arg_value])
                elif arg_key.startswith("flag_") and APP_CONFIG.parser.getboolean(
                    "Arguments", arg_key
                ):
                    command.append(f"--{arg_key.replace('_', '-')}")

        cmd_str_for_log = " ".join(f'"{arg}"' if " " in arg else arg for arg in command)
        logging.info(
            "Executing in %s: %s", APP_CONFIG.liveportrait_repo_path, cmd_str_for_log
        )

        start_time = time.time()
        current_env = process_env.copy()
        current_env["PYTHONIOENCODING"] = "utf-8"
        if os.name == "nt":
            current_env["PYTHONLEGACYWINDOWSSTDIO"] = "1"
            current_env["PYTHONUTF8"] = "1"

        use_shell = os.name == "nt"

        process = subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=current_env,
            check=False,
            shell=use_shell,
            cwd=APP_CONFIG.liveportrait_repo_path,
        )
        duration = time.time() - start_time

        if process.returncode == 0:
            processing_successful = True
            msg_markup = f"OK [status_ok]:heavy_check_mark:[/status_ok] [path]{selected_source_image_path.name}[/path]. Time: {duration:.2f}s."
            logging.info(
                "Folder %s: %s",
                target_folder_path.name,
                msg_markup.replace("[status_ok]", "")
                .replace("[/status_ok]", "")
                .replace("[path]", "")
                .replace("[/path]", ""),
            )
            console.print(
                Text.assemble(
                    Text(
                        sub_tree_connector, style="tree_branch"
                    ),  # Status of the image processing
                    Text.from_markup(f"{msg_markup}"),
                )
            )
        else:
            processing_successful = False
            msg_markup = f"FAIL [status_fail]:x:[/status_fail] [path]{selected_source_image_path.name}[/path]. Code: {process.returncode}. Time: {duration:.2f}s."
            logging.error(
                "Folder %s: %s",
                target_folder_path.name,
                msg_markup.replace("[status_fail]", "")
                .replace("[/status_fail]", "")
                .replace("[path]", "")
                .replace("[/path]", ""),
            )
            logging.error(
                "LP STDERR for %s:\n%s",
                selected_source_image_path.name,
                process.stderr.strip(),
            )
            if process.stdout.strip():
                logging.debug(
                    "LP STDOUT for %s:\n%s",
                    selected_source_image_path.name,
                    process.stdout.strip(),
                )
            console.print(
                Text.assemble(
                    Text(sub_tree_connector, style="tree_branch"),
                    Text.from_markup(f"{msg_markup}"),
                )
            )
            console.print(
                Text.assemble(
                    Text(
                        sub_tree_connector, style="tree_branch"
                    ),  # Should be indented further, like "  │  │    └─" or similar
                    Text.from_markup(
                        f"[dimmed]  (See [path]{APP_CONFIG.log_file_name.name}[/path] for details)[/dimmed]"
                    ),
                )
            )

    except (subprocess.CalledProcessError, FileNotFoundError, OSError) as e:
        processing_successful = False
        error_msg = f"Error processing [path]{selected_source_image_path.name if selected_source_image_path else 'unknown image'}[/path]: {e}"
        logging.error(
            "Folder %s: %s",
            target_folder_path.name,
            error_msg.replace("[path]", "").replace("[/path]", ""),
            exc_info=True,
        )
        console.print(
            Text.assemble(
                Text(
                    sub_tree_connector, style="tree_branch"
                ),  # Status of image processing
                Text.from_markup(
                    f"[danger]:exclamation: SCRIPT ERROR: {error_msg} :exclamation:[/danger]"
                ),
            )
        )
    except Exception as e:
        processing_successful = False
        error_msg = f"Unexpected error processing [path]{selected_source_image_path.name if selected_source_image_path else 'unknown image'}[/path]: {e}"
        logging.error(
            "Folder %s: %s",
            target_folder_path.name,
            error_msg.replace("[path]", "").replace("[/path]", ""),
            exc_info=True,
        )
        console.print(
            Text.assemble(
                Text(sub_tree_connector, style="tree_branch"),
                Text.from_markup(
                    f"[danger]:exclamation: UNEXPECTED SCRIPT ERROR: {error_msg} :exclamation:[/danger]"
                ),
            )
        )
    finally:
        os.chdir(original_cwd)
        logging.info("Restored CWD to: %s", original_cwd)

    return (image_found, image_matched_filter, processing_successful)


def check_liveportrait_dependencies() -> bool:
    """Check for required LivePortrait dependencies and install if missing.
    Based on the original LivePortrait requirements_base.txt"""
    global console

    # LivePortrait dependencies with exact versions from original requirements
    dependencies = [
        # Core ML packages
        (
            "onnxruntime",
            "onnxruntime",
        ),  # Changed from onnxruntime-gpu to prevent failures
        ("tokenizers==0.19.1", "tokenizers"),
        ("torch", "torch"),
        ("torchvision", "torchvision"),
        ("torchaudio", "torchaudio"),
        ("numpy==1.26.4", "numpy"),
        # File processing
        ("pyyaml==6.0.1", "yaml"),
        ("opencv-python==4.10.0.84", "cv2"),
        ("scipy==1.13.1", "scipy"),
        ("imageio==2.34.2", "imageio"),
        ("lmdb==1.4.1", "lmdb"),
        # Utilities
        ("tqdm==4.66.4", "tqdm"),
        ("rich==13.7.1", "rich"),
        ("ffmpeg-python==0.2.0", "ffmpeg"),
        ("onnx==1.16.1", "onnx"),
        ("scikit-image==0.24.0", "skimage"),
        ("albumentations==1.4.10", "albumentations"),
        ("matplotlib==3.9.0", "matplotlib"),
        ("imageio-ffmpeg==0.5.1", "imageio_ffmpeg"),
        ("tyro==0.8.5", "tyro"),
        ("gradio==4.37.1", "gradio"),
        ("pykalman==0.9.7", "pykalman"),
        ("transformers", "transformers"),
        ("pillow>=10.2.0", "PIL"),
        # Batch script dependencies
        ("pyfiglet", "pyfiglet"),
    ]

    # Check for dependencies
    console.print()
    console.print(
        Rule(
            title=Text(
                " Checking LivePortrait Dependencies ", style="section_title_text"
            ),
            style="section_title_text",
            characters="═",
        )
    )
    console.print("Checking LivePortrait packages from original requirements...")

    dependencies_to_install = []
    for package_spec, import_name in dependencies:
        try:
            __import__(import_name)
            console.print(
                f"✓ [success]{package_spec.split('==')[0].split('>=')[0]}[/success] is installed."
            )
        except ImportError:
            console.print(
                f"✗ [warning]{package_spec.split('==')[0].split('>=')[0]}[/warning] is missing. Will install."
            )
            dependencies_to_install.append(package_spec)

    # Check CUDA availability for torch
    has_cuda = False
    cuda_available = False
    if "torch" not in [
        pkg.split("==")[0].split(">=")[0] for pkg, _ in dependencies_to_install
    ]:
        try:
            import torch

            has_cuda = True
            cuda_available = torch.cuda.is_available()
            if cuda_available:
                console.print(f"✓ [success]CUDA is available[/success] for PyTorch.")
            else:
                console.print(
                    f"ℹ [warning]CUDA is NOT available[/warning] for PyTorch. CPU mode will be used."
                )
                # Ensure we set CPU mode in the config
                if "Arguments" in APP_CONFIG.parser:
                    APP_CONFIG.parser.set("Arguments", "flag_force_cpu", "True")
                    APP_CONFIG.save_config()
                    console.print("[info]Set force_cpu flag to True in config.[/info]")
        except Exception as e:
            console.print(
                f"ℹ [warning]Could not check CUDA availability: {e}[/warning]. Assuming CPU mode is needed."
            )
            if "Arguments" in APP_CONFIG.parser:
                APP_CONFIG.parser.set("Arguments", "flag_force_cpu", "True")
                APP_CONFIG.save_config()
                console.print("[info]Set force_cpu flag to True in config.[/info]")

    # Install missing dependencies
    if dependencies_to_install:
        console.print()
        console.print(
            _styled_panel(
                Text(
                    f"Installing {len(dependencies_to_install)} missing LivePortrait dependencies...",
                    style="info",
                ),
                "Package Installation",
                "This may take a few minutes...",
                border_style="panel_border",
            )
        )

        for package_spec in dependencies_to_install:
            package_name = package_spec.split("==")[0].split(">=")[0]
            try:
                console.print(f"Installing [highlight]{package_spec}[/highlight]...")

                # Add retry logic for package installation
                max_retries = 3
                for retry in range(max_retries):
                    try:
                        # Add --timeout and --user to help with permission issues
                        install_cmd = [
                            sys.executable,
                            "-m",
                            "pip",
                            "install",
                            "--timeout",
                            "60",  # 60 second connection timeout
                            "--user"
                            if package_name in ["transformers", "gradio"]
                            else "",  # Use --user for problematic packages
                            package_spec,
                        ]
                        # Remove empty string elements
                        install_cmd = [cmd for cmd in install_cmd if cmd]

                        result = subprocess.run(
                            install_cmd,
                            capture_output=True,
                            text=True,
                            timeout=300,  # 5 minute timeout per package
                        )

                        if result.returncode == 0:
                            console.print(
                                f"✓ [success]{package_name}[/success] installed successfully."
                            )
                            break  # Success, exit retry loop
                        else:
                            if retry < max_retries - 1:  # Not the last attempt
                                console.print(
                                    f"⚠ [warning]Retry {retry + 1}/{max_retries} for {package_name}[/warning]"
                                )
                                time.sleep(2)  # Brief pause before retry
                            else:  # Last attempt failed
                                console.print(
                                    f"✗ [danger]Failed to install {package_name} after {max_retries} attempts[/danger]"
                                )
                                console.print(
                                    f"[dimmed]Error: {result.stderr.strip()[:100]}...[/dimmed]"
                                )
                                logging.error(
                                    f"Failed to install {package_spec}: {result.stderr}"
                                )
                    except subprocess.TimeoutExpired:
                        if retry < max_retries - 1:  # Not the last attempt
                            console.print(
                                f"⏱ [warning]Timeout, retry {retry + 1}/{max_retries} for {package_name}[/warning]"
                            )
                            time.sleep(2)  # Brief pause before retry
                        else:  # Last attempt timed out
                            console.print(
                                f"⏱ [warning]Timeout installing {package_name} after {max_retries} attempts - continuing...[/warning]"
                            )
                            logging.warning(
                                f"Timeout installing {package_spec} after {max_retries} attempts"
                            )
            except Exception as e:
                console.print(
                    f"✗ [danger]Error installing {package_name}: {e}[/danger]"
                )
                logging.error(f"Exception installing {package_spec}: {e}")

        console.print()
        console.print(
            "[info]Dependency installation completed. Some packages may need manual installation if errors occurred.[/info]"
        )
    else:
        console.print(
            "[success]All required LivePortrait dependencies are already installed.[/success]"
        )

    # Final CUDA check
    if has_cuda:
        try:
            import torch

            cuda_available = torch.cuda.is_available()
            if not cuda_available:
                console.print(
                    "[warning]CUDA is not available with the current PyTorch installation.[/warning]"
                )
                console.print(
                    "[info]LivePortrait will run in CPU mode. For faster processing, install CUDA-compatible PyTorch.[/info]"
                )
                # Force CPU mode in config
                if "Arguments" in APP_CONFIG.parser:
                    APP_CONFIG.parser.set("Arguments", "flag_force_cpu", "True")
                    APP_CONFIG.save_config()
        except Exception:
            pass  # Already handled above

    console.print()
    return len(dependencies_to_install) == 0


def open_folder_in_explorer(folder_path: Path) -> bool:
    """Open the folder in the file explorer."""
    if not folder_path.is_dir():
        logging.warning(
            "Cannot open directory: %s (not a directory or not accessible)", folder_path
        )
        return False

    try:
        abs_path_str = str(folder_path.resolve())
        if os.name == "nt":
            subprocess.run(["explorer", abs_path_str], check=False)
        elif sys.platform == "darwin":  # macOS
            subprocess.run(["open", abs_path_str], check=False)
        else:  # Linux and other POSIX
            subprocess.run(["xdg-open", abs_path_str], check=False)

        logging.info("Attempted to open folder in file explorer: %s", abs_path_str)
        return True
    except Exception as e:
        logging.error("Failed to open folder %s in explorer: %s", folder_path, e)
        # Fallback to webbrowser if system-specific calls fail
        try:
            webbrowser.open(folder_path.as_uri())
            logging.info(
                "Fallback: Opened folder using webbrowser: %s", folder_path.as_uri()
            )
            return True
        except Exception as web_e:
            logging.error(
                "Fallback webbrowser.open also failed for %s: %s", folder_path, web_e
            )
            return False


def show_welcome_info() -> None:
    """Display welcome information and application header."""
    console.line()
    console.print(
        Panel(
            Align.center(Text(LP_BATCH_ASCII_ART, style="ascii_art")),
            border_style="panel_border",
            padding=(1, 1),
            expand=True,
        )
    )
    console.print(
        _styled_panel(
            Text.assemble(
                ("Welcome to the ", "info"),
                ("LivePortrait Batch Image Processor!", "bold highlight"),
                (
                    "\n\nThis tool automates processing multiple folders containing source images ",
                    "info",
                ),
                (
                    "using the LivePortrait model. Follow the prompts to configure your batch job.",
                    "info",
                ),
            ),
            "Welcome to LP BATCH",
            f"Version 1.2.0 (UI Enhanced) - Config: {CONFIG_FILE_NAME}",
            box_style=HEAVY_HEAD,
            title_style_key="header",
            expand=True,
        )
    )
    if TKINTER_AVAILABLE:  # Only show if Tk might be used and hasn't failed init
        console.print(
            Panel(
                ":bell: [bold yellow]GUI Dialogs[/bold yellow] :bell:\nIf a GUI dialog for file/folder selection opens, it might appear [underline]BEHIND[/underline] this terminal window. Please check your taskbar or Alt+Tab if a dialog seems to be missing.",
                title=Text("Desktop Notification", style="section_title_text"),
                border_style="orange3",
                padding=(1, 2),
                expand=True,
            )
        )
    console.line()


def get_processing_mode() -> str:
    """Prompt the user to choose between CPU and GPU (CUDA) processing mode."""
    console.line()
    console.print(
        Rule(
            title=Text(" Hardware Acceleration ", style="section_title_text"),
            style="section_title_text",
            characters="═",
        )
    )

    # Check if CUDA is actually available
    cuda_available = False
    try:
        import torch

        cuda_available = torch.cuda.is_available()
    except ImportError:
        cuda_available = False
    except Exception:
        cuda_available = False

    if not cuda_available:
        console.print(
            _styled_panel(
                "[warning]CUDA is not available on this system.[/warning] CPU mode will be used automatically.",
                "Processing Mode Selection",
                "To use GPU acceleration, install CUDA-compatible PyTorch.",
                expand=True,
                border_style="warning",
            )
        )
        console.print(
            Text.assemble(
                ("Using ", "info"),
                ("CPU mode ", "highlight"),
                ("([italic]flag-force-cpu[/italic] is enabled).", "info"),
            )
        )

        # Set the force CPU flag in config
        if "Arguments" in APP_CONFIG.parser:
            APP_CONFIG.parser.set("Arguments", "flag_force_cpu", "True")
            APP_CONFIG.save_config()

        return "cpu"

    # CUDA is available, allow user choice
    console.print(
        _styled_panel(
            "Select the processing mode. GPU (CUDA) is significantly faster but requires a compatible NVIDIA GPU and drivers.",
            "Processing Mode Selection",
            "CPU mode is slower but universally compatible.",
            expand=True,
        )
    )

    table = Table(
        show_header=True,
        box=ROUNDED,
        border_style="table_border",
        header_style="table_header",
        expand=True,
    )
    table.add_column("Option", style="highlight", justify="center", width=10)
    table.add_column("Mode", style="info", ratio=1)
    table.add_column("Details", style="dimmed", ratio=2)
    table.add_row(
        "1", "GPU (CUDA)", "Faster, requires NVIDIA GPU with compatible drivers."
    )
    table.add_row("2", "CPU Only", "Slower, but works on all systems.")
    console.print(table)

    default_mode = "1"
    if "Arguments" in APP_CONFIG.parser and APP_CONFIG.parser.getboolean(
        "Arguments", "flag_force_cpu", fallback=False
    ):
        default_mode = "2"

    choice = IntPrompt.ask(
        _styled_prompt_text("Enter your choice"),
        choices=["1", "2"],
        default=int(default_mode),
    )

    if choice == 2:
        console.print(
            Text.assemble(
                ("Using ", "info"),
                ("CPU mode ", "highlight"),
                ("([italic]flag-force-cpu[/italic] will be enabled).", "info"),
            )
        )

        # Set the force CPU flag in config
        if "Arguments" in APP_CONFIG.parser:
            APP_CONFIG.parser.set("Arguments", "flag_force_cpu", "True")
            APP_CONFIG.save_config()

        return "cpu"
    else:
        console.print(
            Text.assemble(
                ("Using ", "info"),
                ("GPU/CUDA mode", "highlight"),
                (".", "info"),
            )
        )

        # Ensure force CPU flag is disabled
        if "Arguments" in APP_CONFIG.parser:
            APP_CONFIG.parser.set("Arguments", "flag_force_cpu", "False")
            APP_CONFIG.save_config()

        return "cuda"


def process_one_folder(target_folder_path: Path) -> bool:
    """Process a folder by generating videos for images found.
    Returns True if at least one image (that matched filter, if active) was successfully processed, False otherwise.
    """
    logging.info("--- Processing folder: %s ---", target_folder_path.name)

    if not APP_CONFIG.liveportrait_repo_path or not APP_CONFIG.driving_video_path:
        console.print(
            Text.assemble(
                Text("  └─ ", style="tree_branch"),
                Text.from_markup(
                    f"[danger]Critical paths (LP repo or driving video) not set. Aborting folder [path]{target_folder_path.name}[/path].[/danger]"
                ),
            )
        )
        return False

    liveportrait_inference_script = APP_CONFIG.liveportrait_repo_path / "inference.py"
    if not liveportrait_inference_script.is_file():
        msg = f"LP inference.py not found at: [path]{liveportrait_inference_script}[/path]"
        logging.error(msg.replace("[path]", "").replace("[/path]", ""))
        console.print(
            Text.assemble(
                Text("  └─ ", style="tree_branch"),
                Text.from_markup(f"[danger]{msg}[/danger]"),
            )
        )
        return False

    process_env = os.environ.copy()
    process_env.update(APP_CONFIG.env_vars)

    any_filter_matched_and_processed_successfully = False

    # Use different messages and processing approach based on whether filtering is enabled
    if APP_CONFIG.filter_images and APP_CONFIG.filter_phrase:
        # When filtering, we search for ANY images with the filter phrase
        console.print(
            Text.assemble(
                Text("  ├─ ", style="tree_branch"),
                Text("Looking for images containing '", style="info"),
                Text(APP_CONFIG.filter_phrase, "highlight"),
                Text("'...", style="info"),
            )
        )

        # Process images with the filter phrase
        # We pass an empty prefix because we're not looking for specific prefixes
        images_found, images_matched_filter, processing_success = process_single_image(
            target_folder_path,
            "",  # Empty prefix to signal we want to check all images
            process_env,
            is_last_prefix_in_folder=True,
        )

        if images_matched_filter and processing_success:
            any_filter_matched_and_processed_successfully = True
    else:
        # Handle two cases: 1) user selected "Process all images" or 2) user selected prefix-based processing
        if not APP_CONFIG.filter_images:  # Option 1: Process all images in folder
            console.print(
                Text.assemble(
                    Text("  ├─ ", style="tree_branch"),
                    Text("Processing all images in folder...", style="info"),
                )
            )
            # Process all images in the folder by passing an empty prefix
            images_found, images_matched_filter, processing_success = (
                process_single_image(
                    target_folder_path,
                    "",  # Empty prefix to signal we want to check all images
                    process_env,
                    is_last_prefix_in_folder=True,
                )
            )

            if images_matched_filter and processing_success:
                any_filter_matched_and_processed_successfully = True
        else:  # Option 2: Original behavior - use prefix-based processing
            # Determine main branch characters for prefix sections
            selfie_branch_char = "  ├─ "
            license_branch_char = "  ├─ "  # Not the last child

            # Process Selfie Images
            console.print(
                Text.assemble(
                    Text(selfie_branch_char, style="tree_branch"),
                    Text("Processing '", style="info"),
                    Text(APP_CONFIG.source_image_prefix, "highlight"),
                    Text("' images...", style="info"),
                )
            )
            selfie_found, selfie_matched_filter, selfie_success = process_single_image(
                target_folder_path,
                APP_CONFIG.source_image_prefix,
                process_env,
                is_last_prefix_in_folder=False,
            )
            if selfie_matched_filter and selfie_success:
                any_filter_matched_and_processed_successfully = True

            # Process License Images
            console.print(
                Text.assemble(
                    Text(license_branch_char, style="tree_branch"),
                    Text("Processing '", style="info"),
                    Text(APP_CONFIG.license_image_prefix, "highlight"),
                    Text("' images...", style="info"),
                )
            )
            license_found, license_matched_filter, license_success = (
                process_single_image(
                    target_folder_path,
                    APP_CONFIG.license_image_prefix,
                    process_env,
                    is_last_prefix_in_folder=True,
                )
            )
            if license_matched_filter and license_success:
                any_filter_matched_and_processed_successfully = True

    # Renaming logic and final status message for the folder
    final_status_branch_char = "  └─ "
    should_rename_folder = False

    if any_filter_matched_and_processed_successfully:
        should_rename_folder = True

    if APP_CONFIG.filter_images and APP_CONFIG.filter_phrase:
        if not any_filter_matched_and_processed_successfully:
            console.print(
                Text.assemble(
                    Text(final_status_branch_char, style="tree_branch"),
                    Text.from_markup(
                        f"[dimmed]No images matching filter '{APP_CONFIG.filter_phrase}' were found or processing failed.[/dimmed]"
                    ),
                )
            )
    else:
        # Handle both "process all images" and prefix-based processing for status messages
        if not APP_CONFIG.filter_images:
            # This is the "process all images" option
            if not images_found:
                console.print(
                    Text.assemble(
                        Text(final_status_branch_char, style="tree_branch"),
                        Text.from_markup(
                            "[dimmed]No images found in this folder.[/dimmed]"
                        ),
                    )
                )
            elif not should_rename_folder:
                console.print(
                    Text.assemble(
                        Text(final_status_branch_char, style="tree_branch"),
                        Text.from_markup(
                            "[warning]Processing attempted but no images were successfully completed in this folder.[/warning]"
                        ),
                    )
                )
        else:
            # Original prefix-based processing
            if not (selfie_found or license_found):
                console.print(
                    Text.assemble(
                        Text(final_status_branch_char, style="tree_branch"),
                        Text.from_markup(
                            "[dimmed]No source images found in this folder.[/dimmed]"
                        ),
                    )
                )
            elif not should_rename_folder:
                console.print(
                    Text.assemble(
                        Text(final_status_branch_char, style="tree_branch"),
                        Text.from_markup(
                            "[warning]Processing attempted but no images were successfully completed in this folder.[/warning]"
                        ),
                    )
                )

    if should_rename_folder:
        if not target_folder_path.name.endswith(APP_CONFIG.processed_suffix):
            new_folder_name = target_folder_path.name + APP_CONFIG.processed_suffix
            new_folder_path = target_folder_path.parent / new_folder_name
            try:
                target_folder_path.rename(new_folder_path)
                logging.info("Renamed folder to: %s", new_folder_name)
                console.print(
                    Text.assemble(
                        Text(final_status_branch_char, style="tree_branch"),
                        Text("Folder marked as processed: ", "success"),
                        Text(new_folder_name, "highlight"),
                    )
                )
            except (OSError, PermissionError) as e:
                logging.error(
                    "Failed to rename folder %s: %s", target_folder_path.name, e
                )
                console.print(
                    Text.assemble(
                        Text(final_status_branch_char, style="tree_branch"),
                        Text("Could not rename folder ", "warning"),
                        Text(target_folder_path.name, "highlight"),
                        Text(f": {e}", "warning"),
                    )
                )
            except Exception as e:
                logging.critical(
                    "Unexpected error renaming folder %s: %s",
                    target_folder_path.name,
                    e,
                    exc_info=True,
                )
                console.print(
                    Text.assemble(
                        Text(final_status_branch_char, style="tree_branch"),
                        Text("CRITICAL: Unexpected error renaming folder ", "danger"),
                        Text(target_folder_path.name, "highlight"),
                        Text(f": {e}", "danger"),
                    )
                )
        else:
            logging.info(
                "Folder %s already marked as processed.", target_folder_path.name
            )
            console.print(
                Text.assemble(
                    Text(final_status_branch_char, style="tree_branch"),
                    Text.from_markup(
                        "[dimmed]Folder already marked as processed.[/dimmed]"
                    ),
                )
            )

    return any_filter_matched_and_processed_successfully


def run_batch_processing():
    """Run the batch processing workflow."""
    setup_logging()
    logging.info("===================================================")
    logging.info(
        "LP Batch Processor started at %s", datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    )
    logging.info("Script Path: %s", SCRIPT_DIR)
    logging.info("Config Path: %s", CONFIG_PATH)

    console.clear()
    show_welcome_info()
    console.line()
    console.print(
        Rule(
            title=Text(" Configuration Loading ", style="section_title_text"),
            style="section_title_text",
            characters="═",
        )
    )
    console.print(
        Text.assemble(
            ("Loaded configuration from: ", "info"),
            (str(CONFIG_PATH.resolve()), "path"),
        )
    )
    console.line()

    check_and_set_liveportrait_path()
    console.line()
    check_and_set_driving_video()
    console.line()

    # ENSURE DEPENDENCIES ARE CHECKED BEFORE PROCESSING
    console.print(
        Rule(
            title=Text(" Dependency Check ", style="section_title_text"),
            style="section_title_text",
            characters="═",
        )
    )
    dependencies_ok = check_liveportrait_dependencies()
    if not dependencies_ok:
        console.print(
            "[warning]Some dependencies could not be installed. Processing may fail.[/warning]"
        )
        choice = Prompt.ask("Continue anyway? (y/n)", choices=["y", "n"], default="n")
        if choice.lower() == "n":
            console.print("Exiting due to dependency issues.")
            return

    console.line()
    processing_mode = get_processing_mode()
    console.line()
    get_image_filter_settings()
    console.line()

    parent_dir, num_folders_to_process_limit = get_user_inputs()
    APP_CONFIG.default_parent_image_folder = parent_dir
    APP_CONFIG.parser.set("Paths", "default_parent_image_folder", str(parent_dir))
    APP_CONFIG.save_config()
    console.line()

    folders_to_process_this_run = get_unprocessed_folders(
        parent_dir, num_folders_to_process_limit
    )

    if not folders_to_process_this_run:
        console.print(
            _styled_panel(
                Text(
                    "No unprocessed folders found matching criteria. Nothing to do.",
                    style="warning",
                ),
                "Processing Status",
                border_style="orange3",
                expand=True,
            )
        )
        logging.info("No unprocessed folders found. Nothing to do.")
        return

    if "Arguments" not in APP_CONFIG.parser:
        APP_CONFIG.parser.add_section("Arguments")
    if processing_mode == "cpu":
        APP_CONFIG.parser.set("Arguments", "flag_force_cpu", "true")
    else:
        APP_CONFIG.parser.set("Arguments", "flag_force_cpu", "false")
    APP_CONFIG.save_config()

    batch_start_time = time.time()
    total_folders_processed_count = 0
    successful_folders_count = 0
    console.line()
    console.print(
        Rule(
            title=Text(" Step 3: Processing Batch ", style="section_title_text"),
            style="section_title_text",
            characters="═",
        )
    )
    console.print(
        _styled_panel(
            Text.assemble(
                ("Starting batch processing for ", "default"),
                (str(len(folders_to_process_this_run)), "highlight"),
                (" folder(s).", "default"),
            ),
            "Batch Processing Status",
            "Press Ctrl+C to attempt cancellation (may not be immediate).",
            box_style=DOUBLE_EDGE,
            title_style_key="header",
            expand=True,
        )
    )

    progress = Progress(
        SpinnerColumn(spinner_name="dots"),
        TextColumn(
            "[progress_desc]{task.description}[/progress_desc] [progress_name]{task.fields[name]}[/progress_name]",
            markup=True,
        ),
        BarColumn(
            bar_width=None,
            style="progress_bar_active",
            complete_style="progress_bar_complete",
            finished_style="progress_bar_finished",
        ),
        TaskProgressColumn(),
        TextColumn("({task.completed} of {task.total})"),
        TimeRemainingColumn(),
        console=console,
        transient=False,
        expand=True,
    )

    with progress:
        overall_task_id = progress.add_task(
            "Overall Progress", total=len(folders_to_process_this_run), name=""
        )
        for i, folder_path in enumerate(folders_to_process_this_run):
            progress.update(
                overall_task_id,
                description=f"Folder {i + 1}/{len(folders_to_process_this_run)}:",
                name=f"{folder_path.name}",
            )
            console.print(
                Padding(
                    Text.assemble(
                        (
                            f"╭── Processing folder ({i + 1}/{len(folders_to_process_this_run)}): ",
                            "default",
                        ),
                        (folder_path.name, "highlight"),
                    ),
                    (1, 0, 0, 0),  # Top padding
                )
            )

            folder_success = process_one_folder(folder_path)

            if folder_success:
                successful_folders_count += 1
            total_folders_processed_count += 1
            progress.update(overall_task_id, advance=1)

            if i < len(folders_to_process_this_run) - 1:
                # Visually separate folder processing blocks
                console.print(
                    Padding(Rule(style="rule_style", characters="┈"), (1, 0, 1, 0))
                )  # Light dashed line with padding

    total_batch_duration = time.time() - batch_start_time
    minutes, seconds = divmod(total_batch_duration, 60)

    console.print(
        Rule(
            title=Text(" Batch Processing Finished ", style="header"),
            style="rule_style",
            characters="═",
        )
    )
    console.print(
        _styled_panel(
            Text.assemble(
                ("Processed: ", "info"),
                (f"{total_folders_processed_count}", "bold highlight"),
                (" folder(s).\n", "info"),
                ("Successful: ", "success"),
                (f"{successful_folders_count}", "bold status_ok"),
                (" folder(s) (based on criteria).\n", "success"),
                ("Time Taken: ", "info"),
                (f"{int(minutes)}m {int(seconds)}s", "bold highlight"),
            ),
            "Completion Summary",
            box_style=ROUNDED,
            title_style_key="section_title_text",
            expand=True,
        )
    )
    logging.info(
        "Batch processing complete: %d processed, %d successful (met criteria for renaming), %.2f seconds",
        total_folders_processed_count,
        successful_folders_count,
        total_batch_duration,
    )
    console.line()
    console.print(
        Rule(
            title=Text(" Detailed Results ", style="section_title_text"),
            style="section_title_text",
            characters="═",
        )
    )
    summary_table = Table(
        title=Text("Batch Processing Summary", style="section_title_text"),
        box=ROUNDED,
        border_style="table_border",
        header_style="table_header",
        show_lines=True,
        caption_style="dimmed",
        caption=f"Log file: {APP_CONFIG.log_file_name.resolve()}",
        expand=True,
    )
    summary_table.add_column("Metric", style="info", justify="right", ratio=1)
    summary_table.add_column("Value", style="highlight", justify="left", ratio=1)

    summary_table.add_row("Total Folders Processed", str(total_folders_processed_count))
    summary_table.add_row(
        "Successful Folders (Renamed/Marked Done)",
        Text(str(successful_folders_count), style="status_ok"),
    )
    failure_to_rename_count = total_folders_processed_count - successful_folders_count
    summary_table.add_row(
        "Folders Not Marked Done",
        Text(
            str(failure_to_rename_count),
            style="status_fail" if failure_to_rename_count > 0 else "status_neutral",
        ),
    )
    success_rate_str = (
        f"{(successful_folders_count / total_folders_processed_count) * 100:.1f}% (for marking done)"
        if total_folders_processed_count > 0
        else "N/A"
    )
    summary_table.add_row("Success Rate (Marking Done)", success_rate_str)
    time_str = (
        f"{int(minutes)}m {int(seconds)}s"
        if minutes > 0 or seconds >= 10
        else f"{seconds:.1f}s"
    )
    summary_table.add_row("Total Processing Time", time_str)
    console.print(summary_table)

    logging.info("--- Summary ---")
    logging.info("Total Folders Queued: %d", len(folders_to_process_this_run))
    logging.info("Folders Processed (attempted): %d", total_folders_processed_count)
    logging.info("Folders Successful (renamed): %d", successful_folders_count)
    logging.info("Folders Not Renamed: %d", failure_to_rename_count)
    logging.info("Success Rate (Renaming): %s", success_rate_str)
    logging.info("Total Time: %s", time_str)
    logging.info(
        "LP Batch Processor finished at %s",
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    )
    logging.info("===================================================")

    console.line()
    console.print(
        Panel(
            Align.center(
                Text.from_markup(
                    "[success]:tada: Batch processing complete! :tada:[/success]"
                )
            ),
            padding=1,
            expand=True,
            border_style="green3",
        )
    )

    if total_folders_processed_count > 0 and parent_dir.is_dir():
        console.line()
        open_choice = Prompt.ask(
            _styled_prompt_text(
                f"Open the parent output folder [path]{parent_dir}[/path] in file explorer?"
            ),
            choices=["y", "n"],
            default="y",
        )
        if open_choice.lower() == "y":
            if open_folder_in_explorer(parent_dir):
                console.print(
                    Text.from_markup(
                        f"[info]Attempted to open folder: [path]{parent_dir}[/path][/info]"
                    )
                )
            else:
                console.print(
                    Text.from_markup(
                        f"[warning]Could not automatically open folder: [path]{parent_dir}[/path]. Please open it manually.[/warning]"
                    )
                )


if __name__ == "__main__":
    try:
        run_batch_processing()
    except KeyboardInterrupt:
        console.line(2)
        console.print(
            Panel(
                Text.assemble(
                    (":warning: Process interrupted by user (Ctrl+C). ", "warning"),
                    ("Exiting gracefully.", "info"),
                ),
                title="User Interruption",
                border_style="orange3",
                expand=True,
            )
        )
        logging.warning("Process interrupted by user (Ctrl+C).")
    except configparser.Error as cfg_err:
        console.print(f"[error]A configuration file error occurred: {cfg_err}[/error]")
        console.print_exception(show_locals=False, width=console.width)
        logging.critical("Configuration error: %s", cfg_err, exc_info=True)
    except Exception as main_exception:
        console.print_exception(show_locals=True, width=console.width)
        logging.critical(
            "UNEXPECTED: An unhandled exception occurred in main: %s",
            main_exception,
            exc_info=True,
        )
    finally:
        if _tk_root and _tk_root.winfo_exists():
            try:
                _tk_root.destroy()
            except tkinter.TclError:
                pass
        console.line()
        Prompt.ask(
            _styled_prompt_text("Press Enter to exit the application"),
            show_default=False,
            default="",
        )
        logging.info("Application exiting.")
        sys.exit(0)
