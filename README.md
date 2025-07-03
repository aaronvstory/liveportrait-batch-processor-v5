# LivePortrait Batch Processor

Professional automated batch processing tools for LivePortrait facial animation. Process hundreds of images efficiently with advanced features, smart management, and clean interfaces.

## 🚀 Quick Start

![LivePortrait Batch Processor v5 Interface](lpbatchv5%20screenshot.png)

### Option 1: Enhanced v5 (Recommended)
- **File**: `launch_v5.bat`
- **Features**: PKL templates (10x faster), progress tracking, config memory, restart functionality
- **Best for**: Power users, large batches, professional workflows

### Option 2: Classic v3.5 (Maximum Compatibility)
- **File**: `LPbatchV3.5.bat` 
- **Features**: Simple, reliable, lightweight, proven stability
- **Best for**: Simple workflows, basic batch processing, guaranteed compatibility

## 📊 Version Comparison

| Feature | v3.5 Classic | v5 Enhanced |
|---------|-------------|-------------|
| **UI Style** | Clean ASCII tree display | Rich console with same clean display |
| **Processing** | Sequential only | Sequential + Parallel options |
| **Templates** | Video only | Video + PKL templates (10x faster) |
| **Progress** | Simple, clear text | Detailed tracking with same clarity |
| **Config Memory** | None | Remembers all user preferences |
| **Folder Management** | Basic | Smart skip/reprocess with counters |
| **Error Handling** | Basic | Advanced retry logic + detailed logging |
| **Resume Support** | None | Full session state management |
| **Restart Option** | Exit only | Press Enter to restart, Ctrl+C to exit |
| **Python Detection** | Manual | Auto-detects LivePortrait venv |

## 🛠️ Installation

### Option A: Quick Setup (v5)
1. **Download/Clone** this repository
2. **Configure**: Copy `config_template.ini` to `liveportrait_batch_config.ini`
3. **Edit paths**: Set your LivePortrait installation path
4. **Launch**: Run `launch_v5.bat`

### Option B: Simple Setup (v3.5)
1. **Download/Clone** this repository  
2. **Launch**: Run `LPbatchV3.5.bat`
3. **Follow prompts**: Set paths when asked

## ⚙️ Configuration

### v5 Configuration (`liveportrait_batch_config.ini`):

```ini
[Paths]
liveportrait_repo_path = C:\path\to\LivePortrait
default_parent_image_folder = C:\path\to\images

[Filter]  
filter_images = true
filter_phrase = gen-selfie,gen-3,selfie

[Batch]
max_folders = 0  # 0 = process all
skip_processed = true  # Smart folder management
```

### v3.5 Configuration:
- All configuration done through interactive prompts
- No config files needed
- Automatically detects LivePortrait installation

## 🎯 Key Features

### 🚀 Enhanced v5 Features:
- **⚡ PKL Template Support**: 10x faster processing with pre-processed templates
- **🧠 Smart Configuration**: Remembers all your preferences between sessions
- **📊 Clean Progress Display**: Same clean tree view as v3.5 but with enhanced tracking
- **🔄 Restart Functionality**: Press Enter to restart, Ctrl+C to exit
- **🎯 Advanced Filtering**: Multiple filter terms with clear feedback
- **💾 Session Management**: Resume interrupted batches automatically
- **⚙️ Parallel Processing**: Optional faster processing mode
- **🐍 Smart Python Detection**: Auto-finds LivePortrait's virtual environment

### 🛡️ Classic v3.5 Features:
- **✅ Maximum Stability**: Time-tested, battle-proven processing
- **🎯 Clean Interface**: Simple tree-style progress display  
- **🔧 Universal Compatibility**: Works in all environments
- **⚡ Lightweight**: Minimal resource usage
- **📋 Simple Setup**: No config files required

## 📋 Usage Guide

### Enhanced v5 Workflow:
1. **Launch**: `launch_v5.bat`
2. **Auto-Setup**: Detects LivePortrait installation automatically
3. **Processing Mode**: Auto-detect (recommended) or manual selection
4. **Template Choice**: PKL templates (fast) or video files
5. **Folder Selection**: Smart filtering with memory
6. **Batch Settings**: Configure parallel/sequential processing
7. **Start & Monitor**: Clean progress display like v3.5
8. **Restart**: Press Enter to process more batches

### Classic v3.5 Workflow:
1. **Launch**: `LPbatchV3.5.bat`
2. **Follow Prompts**: Interactive setup
3. **Select Folders**: Simple folder browsing
4. **Process**: Watch clean progress display
5. **Complete**: Simple completion message

## 📁 Project Structure

```
LPbatchV5/
├── 🚀 Enhanced v5
│   ├── launch_v5.bat              # v5 launcher with restart functionality
│   ├── enhanced_lp_batch_v5.py    # Main v5 application
│   └── config_template.ini        # Configuration template
├── 🛡️ Classic v3.5  
│   ├── LPbatchV3.5.bat           # v3.5 launcher
│   └── LPbatchV3.5.py            # Classic v3.5 application
├── 📁 Shared Resources
│   ├── driving_templates/         # PKL template storage (v5)
│   ├── logs/                     # Application logs
│   └── debug_lp_command.py       # Debugging utilities
└── 📖 Documentation
    └── README.md                 # This comprehensive guide
```

## 🎬 Advanced Features (v5 Only)

### ⚡ PKL Templates (10x Speed Boost):
```
driving_templates/
├── template1.pkl    # Fast emotion template
├── template2.pkl    # Speaking template  
└── template3.pkl    # Custom animation
```

### 💾 Session Recovery:
- Automatically saves progress during processing
- Resume interrupted batches with exact state
- Smart skip of already processed folders

### 🔄 Restart Functionality:
- **Enter**: Restart application for new batch
- **Ctrl+C**: Exit application completely
- No more confusing exit prompts

### ⚙️ Smart Configuration Memory:
- Processing mode preferences
- Template selections
- Filter settings
- Folder management preferences
- Batch processing options

## 🚨 Troubleshooting

### 🔧 Common Solutions:

**v5 Issues:**
```bash
# WinError 87 (Fixed in latest version)
✅ Auto-detects LivePortrait virtual environment
✅ Uses correct Python executable

# Config Issues  
✅ Copy config_template.ini to liveportrait_batch_config.ini
✅ Set liveportrait_repo_path to your installation
```

**v3.5 Issues:**
```bash
# Python Detection
✅ Automatically finds LivePortrait venv
✅ Falls back to system Python if needed

# Path Issues
✅ Interactive prompts guide you through setup
✅ Validates paths before processing
```

**General Issues:**
1. **"No face detected"**: Use images with clear, front-facing faces
2. **Processing timeout**: Images processed successfully but with timeouts
3. **Memory issues**: Reduce parallel tasks or use sequential mode
4. **Template not found**: Ensure PKL files are in `driving_templates/` folder

### 🩺 Debug Tools:
- `debug_lp_command.py` - Test LivePortrait commands directly  
- Log files in project directory
- Enhanced error reporting in v5

## 💡 Pro Tips

### 🎯 For Maximum Speed (v5):
1. Use PKL templates instead of videos (10x faster)
2. Enable parallel processing for multiple folders
3. Use filtering to process only needed images
4. Keep frequently used templates in `driving_templates/`

### 🛡️ For Maximum Reliability (v3.5):
1. Use sequential processing always
2. Process smaller batches (50-100 folders)
3. Monitor progress closely
4. Simple video-based driving for compatibility

### 🔄 For Workflow Efficiency:
1. Set up filter phrases for your image naming
2. Use restart functionality to process multiple batches
3. Enable smart folder management to skip completed work
4. Save different config files for different projects

## 🎨 Display Improvements

Both versions now feature the same clean, readable display:

```
Processing folder (6/10): user_241622
├─ Looking for images containing 'gen-selfie'...
├─ Processing image: selfie.jpg
└─ OK ✓selfie.jpg. Time: 52.9s
└─ Folder marked as processed: user_241622- done
```

**v5 Benefits**: All the advanced features with v3.5's proven clean interface!

## 🤝 Contributing

We welcome contributions! Please feel free to:
- Submit bug reports and feature requests
- Improve documentation
- Add new PKL templates
- Enhance error handling

## 📄 License

This project is provided as-is for educational and personal use with LivePortrait.

---

**Both versions are production-ready with clean interfaces and reliable processing!**
