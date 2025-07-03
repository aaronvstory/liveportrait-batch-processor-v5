#!/usr/bin/env python3
"""
Debug script for LivePortrait Batch Processor v5.0
Tests LivePortrait command execution and helps identify issues
"""

import os
import sys
import subprocess
import configparser
from pathlib import Path

def test_liveportrait_command():
    """Test LivePortrait command execution with current configuration"""
    
    print("🔍 LivePortrait Command Test - Debug Mode")
    print("=" * 60)
    
    # Load configuration
    config_path = Path(__file__).parent / "liveportrait_batch_config.ini"
    if not config_path.exists():
        print("❌ Configuration file not found!")
        print(f"   Expected: {config_path}")
        return False
    
    config = configparser.ConfigParser()
    config.read(config_path)
    
    # Get paths
    lp_repo = Path(config.get("Paths", "liveportrait_repo_path"))
    driving_template = Path(config.get("Paths", "driving_template_path"))
    input_dir = Path(config.get("Paths", "default_parent_image_folder"))
    python_exe = config.get("Paths", "python_executable") or sys.executable
    
    print(f"📁 LivePortrait repo: {lp_repo}")
    print(f"🎭 Driving template: {driving_template}")
    print(f"📂 Input directory: {input_dir}")
    print(f"🐍 Python executable: {python_exe}")
    print()
    
    # Validate paths
    if not lp_repo.exists():
        print("❌ LivePortrait repository path does not exist!")
        return False
    
    inference_py = lp_repo / "inference.py"
    if not inference_py.exists():
        print("❌ inference.py not found in LivePortrait repository!")
        print(f"   Expected: {inference_py}")
        return False
    
    if not driving_template.exists():
        print("❌ Driving template file does not exist!")
        return False
    
    if not input_dir.exists():
        print("❌ Input directory does not exist!")
        return False
    
    print("✅ All paths validated successfully")
    print()
    
    # Find a test image (prefer selfie/front images)
    test_image = None
    for folder in input_dir.iterdir():
        if folder.is_dir():
            # Look for images with face-indicating names first
            face_patterns = ['*selfie*', '*front*', '*gen-selfie*', '*gen-3*']
            for pattern in face_patterns:
                images = list(folder.glob(pattern + '.jpg')) + list(folder.glob(pattern + '.jpeg')) + list(folder.glob(pattern + '.png'))
                if images:
                    test_image = images[0]
                    break
            
            # If no face-specific images, try any image except back/license
            if not test_image:
                for ext in ['*.jpg', '*.jpeg', '*.png']:
                    images = list(folder.glob(ext))
                    for img in images:
                        # Skip back/license images that likely don't have faces
                        if not any(word in img.name.lower() for word in ['back', 'license', 'rear', 'behind']):
                            test_image = img
                            break
                    if test_image:
                        break
            if test_image:
                break
    
    if not test_image:
        print("❌ No test images found in input directory!")
        return False
    
    print(f"🖼️  Test image: {test_image}")
    
    # Create test output directory
    output_dir = Path(__file__).parent / "debug_output"
    output_dir.mkdir(exist_ok=True)
    
    # Build command
    command = [
        python_exe,
        str(inference_py),
        "--source", str(test_image),
        "--driving", str(driving_template),
        "--output-dir", str(output_dir),
    ]
    
    # Add config flags
    try:
        if config.getboolean("Arguments", "flag_force_cpu"):
            command.append("--flag-force-cpu")
    except:
        pass
    try:
        if config.getboolean("Arguments", "flag_use_half_precision"):
            command.append("--flag-use-half-precision")
    except:
        pass
    try:
        if config.getboolean("Arguments", "flag_relative_motion"):
            command.append("--flag-relative-motion")
    except:
        pass
    try:
        if config.getboolean("Arguments", "flag_pasteback"):
            command.append("--flag-pasteback")
    except:
        pass
    
    print()
    print("🚀 Executing LivePortrait command:")
    command_str = ' '.join(f'"{arg}"' if ' ' in str(arg) else str(arg) for arg in command)
    print("   " + command_str)
    print(f"📏 Command length: {len(command_str)} characters")
    
    if len(command_str) > 8000:
        print("⚠️  WARNING: Command line may be too long for Windows!")
    print()
    
    # Execute command
    try:
        # Always use subprocess.run with direct command array to avoid shell issues
        result = subprocess.run(
            command,
            cwd=lp_repo,
            capture_output=True,
            text=True,
            timeout=60  # 1 minute timeout for test
        )
        
        print(f"📊 Return code: {result.returncode}")
        
        if result.stdout:
            print("📤 STDOUT:")
            print(result.stdout)
        
        if result.stderr:
            print("📥 STDERR:")
            print(result.stderr)
        
        if result.returncode == 0:
            print("✅ Command executed successfully!")
            # Check for output files
            output_files = list(output_dir.glob("*.mp4"))
            if output_files:
                print(f"🎬 Output file created: {output_files[0]}")
            else:
                print("⚠️  No output file found")
        else:
            print("❌ Command failed!")
            
        return result.returncode == 0
        
    except subprocess.TimeoutExpired:
        print("⏰ Command timed out!")
        return False
    except Exception as e:
        print(f"💥 Exception: {e}")
        return False

if __name__ == "__main__":
    test_liveportrait_command()
    print("\n🔧 Debug complete. Check the output above for issues.")
    input("Press Enter to exit...")
