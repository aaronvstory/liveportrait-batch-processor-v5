# LivePortrait Batch Processor v5.0 - Production Release

🎭 **Professional-grade AI facial animation batch processing tool** - Process hundreds of images with detailed progress tracking, PKL template support, and smart folder management.

## 🚀 Quick Start

### 1. Initial Setup (First Time Only)
```bash
# Clone the repository
git clone [your-repo-url]
cd LivePortrait-Batch-Processor

# Run the setup script (creates virtual environment and installs dependencies)
setup.bat
```

### 2. Configuration
Edit `liveportrait_batch_config.ini` with your paths:
- **LivePortrait repository path** (required)
- **Image folders location** (required)  
- **PKL templates directory** (optional, for 10x speed boost)

### 3. Collect Templates (Recommended)
```bash
# Gather existing PKL templates for faster processing
collect_templates.bat
```

### 4. Start Processing
```bash
# Launch the main application
launch_v5.bat
```

## ✨ Key Features

### 🎯 Professional Performance
- **PKL Template Support**: 10x faster processing (4-8 seconds vs 40+ seconds per image)
- **Smart Progress Tracking**: Real-time file-by-file completion with timing
- **Parallel & Sequential Modes**: Choose stability or speed based on your needs
- **Resume Functionality**: Continue interrupted batches seamlessly

### 🔧 Advanced Controls
- **Smart Folder Management**: Skip processed folders or reprocess everything
- **Image Filtering**: Process only images matching specific filename patterns
- **Batch Limiting**: Control how many folders to process at once
- **Retry Logic**: Automatic retry with configurable attempts and delays

### 🖥️ Enhanced User Experience
- **Beautiful Console UI**: Rich terminal interface with progress bars and colors
- **Cross-platform Support**: Works on Windows with proper virtual environment isolation
- **Detailed Logging**: Comprehensive logs for debugging and monitoring
- **Configuration Memory**: Remembers your preferences between sessions

## 📁 Project Structure

```
LivePortrait-Batch-Processor/
├── enhanced_lp_batch_v5.py      # Main processor application
├── template_manager.py          # PKL template collection utility
├── driving_templates/           # PKL templates storage (auto-created)
├── launch_v5.bat               # Main launcher with virtual environment
├── setup.bat                   # Initial setup and dependency installer
├── collect_templates.bat       # Template collection launcher
├── requirements.txt            # Python dependencies
├── config_template.ini         # Configuration template
├── liveportrait_batch_config.ini # Your configuration (created from template)
└── README.md                   # This file
```

## ⚙️ Configuration Options

### Processing Modes
- **GPU (CUDA)**: Fastest processing with NVIDIA GPUs
- **CPU**: Universal compatibility, slower but reliable
- **Auto-detect**: Automatically chooses best available option

### Driving Types
- **PKL Templates**: Pre-processed templates for maximum speed (recommended)
- **Video Files**: Traditional video-based driving (slower but flexible)

### Batch Settings
- **Sequential Mode**: Process one image at a time with detailed progress (recommended)
- **Parallel Mode**: Multi-threaded processing for speed (less detailed progress)
- **Folder Limiting**: Set maximum number of folders to process
- **Skip Processed**: Automatically skip folders already marked as complete

## 🎮 Usage Examples

### Basic Workflow
1. Place your images in organized folders
2. Configure LivePortrait path in `liveportrait_batch_config.ini`
3. Run `collect_templates.bat` to gather PKL templates
4. Launch `launch_v5.bat` and follow the prompts
5. Monitor real-time progress with file-by-file completion times

### Advanced Filtering
```ini
[Filter]
filter_images = true
filter_phrase = selfie,portrait,headshot
```
Only processes images containing these terms in their filenames.

### Performance Optimization
- Use PKL templates instead of videos for 10x speed improvement
- Enable parallel processing for faster batch completion
- Set appropriate folder limits to avoid memory issues

## 🔧 Troubleshooting

### Common Issues
- **Virtual environment errors**: Run `setup.bat` again
- **Missing dependencies**: Ensure `requirements.txt` is installed properly
- **GPU detection fails**: Check CUDA installation or use CPU mode
- **Path errors**: Use absolute paths in configuration

### Getting Help
- Check `liveportrait_batch_log.txt` for detailed error information
- Enable sequential mode for better error visibility
- Use filtering to test with smaller batches first

## 📊 Performance Metrics

### Speed Comparisons
- **PKL Templates**: 4-8 seconds per image
- **Video Files**: 40+ seconds per image
- **Parallel Mode**: 2-3x faster than sequential (system dependent)

### Resource Usage
- **Memory**: Optimized for large batches without memory leaks
- **CPU**: Efficient threading with configurable parallel limits
- **Storage**: Automatic cleanup of temporary files

## 🛠️ Development & Deployment

### Requirements
- Python 3.8+
- Rich terminal library (auto-installed)
- LivePortrait installation
- NVIDIA GPU (optional, for CUDA acceleration)

### Virtual Environment
The application automatically manages its Python virtual environment:
- Created during `setup.bat`
- Activated automatically in `launch_v5.bat`
- Isolated dependencies prevent conflicts

### Cross-Platform Notes
- Designed for Windows batch script compatibility
- Uses relative paths for portability
- Environment variables prevent hardcoded paths

## 📝 Version History

### v5.0 - Production Release
- ✅ Fixed parallel processing reliability
- ✅ Enhanced progress display with file-by-file tracking
- ✅ Added PKL template support for 10x speed improvement
- ✅ Implemented smart folder management (skip processed)
- ✅ Added comprehensive error handling and retry logic
- ✅ Created professional deployment scripts and virtual environment support
- ✅ Polished UI with cleaner progress display and reduced redundancy

### Previous Versions
- v4.x: Parallel processing experiments
- v3.x: Basic batch functionality
- v2.x: Initial automation scripts

## 🤝 Contributing

This project follows production-ready standards:
- Virtual environment isolation
- Comprehensive error handling
- Cross-platform compatibility
- Professional logging and monitoring
- Clean, maintainable code structure

---

**PAPESLAY Confirmed**: Production-ready deployment with professional UI, virtual environment management, and comprehensive cross-platform support.
