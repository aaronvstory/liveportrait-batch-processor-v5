# Troubleshooting Guide - LivePortrait Batch Processor v5.0

## Path Configuration Issues

### Problem: All Processing Failing with Command Errors

**Symptoms:**
- All folders show "Failed to process any images"
- Error messages like "Command '[...Scripts/LPbatc..." truncated
- 0/N folders processed successfully
- Commands failing immediately

**Solutions:**

1. **Run Debug Test**
   ```bash
   python debug_test.py
   # This will test LivePortrait command execution and show detailed errors
   ```

2. **Check Python Environment**
   ```bash
   # Make sure you're using the right Python for LivePortrait
   # Add to config if LivePortrait needs specific Python:
   [Paths]
   python_executable = C:\path\to\liveportrait\python.exe
   ```

3. **Verify LivePortrait Installation**
   ```bash
   # Test LivePortrait manually first:
   cd F:\DF\LivePortraitPortablev110
   python inference.py --help
   ```

4. **Check Command Arguments**
   - LivePortrait flag names may have changed
   - Verify flag syntax matches your LivePortrait version
   - Check if inference.py accepts the flags being used

### Problem: "Invalid LivePortrait path" Error

**Symptoms:**
- Application shows "Invalid LivePortrait path. Please ensure the directory contains inference.py"
- Path configuration fails even with correct directory

**Solutions:**

1. **Verify LivePortrait Installation**
   ```
   ✅ Check that LivePortrait is properly installed
   ✅ Ensure the directory contains 'inference.py' file
   ✅ Look for the main LivePortrait folder (not a subdirectory)
   ```

2. **Use Absolute Paths**
   ```
   ❌ Bad: .\LivePortrait
   ❌ Bad: ~\Downloads\LivePortrait  
   ✅ Good: C:\AI\LivePortrait
   ✅ Good: F:\Tools\LivePortraitPortablev110
   ```

3. **GUI Dialog Issues**
   - If folder picker doesn't appear, check behind the console window
   - If GUI fails, the app will prompt for manual path entry
   - You get 3 attempts to configure each path correctly

### Problem: Configuration Not Saving

**Symptoms:**
- Paths reset every time you run the application
- "Current path" not remembered between sessions

**Solutions:**

1. **Check File Permissions**
   ```bash
   # Ensure the application can write to config file
   # Right-click folder → Properties → Security
   # Make sure your user has "Full control"
   ```

2. **Run Setup Again**
   ```bash
   setup.bat
   # This recreates the virtual environment and config files
   ```

3. **Manual Config Edit**
   ```bash
   # Edit liveportrait_batch_config.ini directly
   # Replace empty paths with your absolute paths
   ```

### Problem: Virtual Environment Issues

**Symptoms:**
- "Virtual environment not found" error
- Dependencies not installing properly

**Solutions:**

1. **Run Setup Script**
   ```bash
   setup.bat
   # This creates virtual environment and installs dependencies
   ```

2. **Manual Virtual Environment Creation**
   ```bash
   python -m venv .venv
   .venv\Scripts\activate.bat
   pip install -r requirements.txt
   ```

3. **Python Version Check**
   ```bash
   python --version
   # Ensure Python 3.8+ is installed
   ```

## Common Path Examples

### Windows Paths
```ini
[Paths]
liveportrait_repo_path = C:\AI\LivePortrait
default_parent_image_folder = C:\Users\YourName\Pictures\ToProcess
driving_template_path = C:\Scripts\LPbatchV5\driving_templates\your_template.pkl
```

### Portable Installation Paths
```ini
[Paths]
liveportrait_repo_path = F:\PortableAI\LivePortraitPortablev110
default_parent_image_folder = F:\Images\BatchProcess
```

## Step-by-Step Path Configuration

### 1. LivePortrait Path
- Navigate to your LivePortrait installation
- Look for the main folder containing `inference.py`
- **NOT** the subfolder with models or assets
- Example: `C:\AI\LivePortrait` (contains inference.py)

### 2. Input Directory  
- Choose parent folder containing image folders
- Each subfolder should contain images to process
- Structure: `Parent\Folder1\image1.jpg, image2.jpg`

### 3. Templates (Optional)
- Run `collect_templates.bat` first to gather PKL files
- Choose from existing templates in `driving_templates\` folder
- Templates provide 10x faster processing than videos

## Advanced Troubleshooting

### Reset Configuration
```bash
# Delete config file to start fresh
del liveportrait_batch_config.ini
# Run application - it will recreate default config
launch_v5.bat
```

### Debug Mode
```bash
# Check logs for detailed error information
type liveportrait_batch_log.txt
```

### Environment Variables
```bash
# Check Python environment
where python
echo %PATH%
```

## Getting Help

1. **Check Log Files**
   - `liveportrait_batch_log.txt` - Detailed processing logs
   - Check for specific error messages

2. **Verify Installation**
   - Ensure LivePortrait works independently
   - Test with a single image first

3. **File Permissions**
   - Ensure write access to project directory
   - Run as Administrator if needed (for first-time setup only)

## Performance Tips

### Optimal Configuration
- Use PKL templates instead of videos (10x faster)
- Place input images on fast storage (SSD)
- Keep image folders organized and not too large (< 100 images per folder)

### Memory Management
- Close other applications during large batch processing
- Use sequential mode for stability with large batches
- Monitor system resources during processing

---

**Need More Help?**
- Check the main README.md for complete setup instructions
- Verify your LivePortrait installation works independently
- Ensure all paths use absolute directory references
