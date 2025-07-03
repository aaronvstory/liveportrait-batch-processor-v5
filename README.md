# LivePortrait Batch Processor v5.0 Enhanced

## 🎯 Key Improvements

### ✅ Fixed Issues
- **Parallel Processing**: Simplified approach that actually works on Windows
- **Progress Tracking**: Detailed progress showing individual file completion times and filenames  
- **Skip/Reprocess**: Option to skip already processed folders or reprocess everything
- **Error Handling**: Enhanced recovery and retry mechanisms
- **Memory Management**: Better resource usage for large batches

### 🚀 New Features
- **Sequential Mode**: Process one image at a time with detailed progress (recommended)
- **Parallel Mode**: Optional multi-threaded processing for speed
- **Template Support**: Use PKL templates for 10x faster processing (4-8s vs 40+s)
- **Real-time Updates**: Live progress with completion times and file names
- **Folder Management**: Automatic marking of completed folders

## 📁 Project Structure

```
LPbatchV5/
├── enhanced_lp_batch_v5.py      # Main enhanced processor
├── template_manager.py          # PKL template collection tool
├── driving_templates/           # Folder for PKL templates (auto-created)
├── launch_v5.bat               # Main launcher
├── collect_templates.bat       # Template collection launcher
├── liveportrait_batch_config.ini # Configuration file
├── batch_state.json            # Resume state (auto-created)
└── liveportrait_batch_log.txt  # Processing log (auto-created)
```

## 🎮 Usage

### 1. First Time Setup
Run `collect_templates.bat` to gather existing PKL templates from LivePortrait temp folders.

### 2. Main Processing
Run `launch_v5.bat` and follow the prompts:
- **Processing Mode**: CUDA/CPU/Auto-detect
- **Driving Type**: Video file or PKL template (much faster)
- **Filtering**: Process all images or filter by filename phrase
- **Processing Settings**:
  - Sequential (recommended): Stable with detailed progress
  - Parallel: Faster but less detailed progress
- **Folder Handling**: Skip processed folders or reprocess all

### 3. Progress Monitoring
The enhanced version shows:
```
✅ [1/23] selfie.jpg - 4.2s
✅ [2/23] license.jpg - 3.8s
❌ [3/23] photo.jpg - Failed: File not found
✅ [4/23] selfie_2.jpg - 4.1s
```

## 🔧 Configuration

The `liveportrait_batch_config.ini` file contains all settings:
- **Paths**: LivePortrait repo, input folders, templates
- **Arguments**: GPU settings, processing flags
- **Batch**: Retry logic, parallel settings, filename prefixes
- **Filter**: Image filtering options

## ⚡ Performance Tips

### Template Processing (Recommended)
- Use PKL templates instead of video files
- 10x faster: 4-8 seconds per image vs 40+ seconds
- Run `collect_templates.bat` to gather existing templates

### Processing Modes
- **Sequential**: More stable, shows detailed progress per file
- **Parallel**: Faster processing but less detailed feedback
- **Templates**: Always faster than video files regardless of mode

### Folder Management
- **Skip Processed**: Only process new folders (faster)
- **Reprocess All**: Process everything including completed folders

## 🐛 Troubleshooting

### Common Issues
1. **No templates found**: Run `collect_templates.bat` first
2. **GUI dialogs hidden**: Check behind the console window
3. **Parallel processing fails**: Use Sequential mode instead
4. **Path errors**: Ensure all paths are absolute and exist

### Debugging
- Check `liveportrait_batch_log.txt` for detailed error logs
- Use Sequential mode for better error visibility
- Enable filtering to test with fewer images first

## 🔄 Resume Functionality

The processor automatically saves progress and offers to resume interrupted batches:
- State saved in `batch_state.json`
- Resume prompt shows progress and failed count
- Can continue from where it left off

## 📊 Results Display

Enhanced results show:
- **Overall Statistics**: Total, completed, failed, success rate, duration
- **Per-Folder Breakdown**: Success rates and average times by folder
- **Failed Images List**: Detailed error information for troubleshooting
- **Performance Metrics**: Average processing time per image

PAPESLAY confirmed - v5.0 addresses all requested improvements with detailed progress tracking and fixed parallel processing!
