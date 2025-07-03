#!/usr/bin/env python3
"""Template Manager - Automatically collect .pkl files from LivePortrait temp folder"""

import shutil
from pathlib import Path
from datetime import datetime

def get_templates_folder():
    """Get the driving templates folder."""
    script_dir = Path(__file__).parent.resolve()
    templates_dir = script_dir / "driving_templates"
    templates_dir.mkdir(exist_ok=True)
    return templates_dir

def scan_liveportrait_temp():
    """Scan LivePortrait temp folders for .pkl files."""
    # Common LivePortrait temp locations
    temp_locations = [
        Path("F:/DF/LivePortraitPortablev110/tmp/gradio"),
        Path("C:/LivePortrait/tmp/gradio"),
        Path("./tmp/gradio"),
    ]
    
    pkl_files = []
    for lp_temp in temp_locations:
        if lp_temp.exists():
            print(f"Found LivePortrait temp: {lp_temp}")
            for temp_folder in lp_temp.iterdir():
                if temp_folder.is_dir():
                    for pkl_file in temp_folder.glob("*.pkl"):
                        pkl_files.append(pkl_file)
            break
    else:
        print("No LivePortrait temp folders found in common locations:")
        for loc in temp_locations:
            print(f"  - {loc}")
    
    return pkl_files

def copy_templates():
    """Copy new .pkl files to templates folder."""
    templates_dir = get_templates_folder()
    existing_templates = {t.name for t in templates_dir.glob("*.pkl")}
    
    temp_pkl_files = scan_liveportrait_temp()
    copied_count = 0
    
    print("🔍 Scanning for new .pkl templates...")
    print(f"📁 Templates folder: {templates_dir}")
    print(f"📦 Found {len(temp_pkl_files)} .pkl files in LivePortrait temp")
    
    for pkl_file in temp_pkl_files:
        # Create a clean name
        clean_name = pkl_file.name.replace(" ", "_")
        target_path = templates_dir / clean_name
        
        # Skip if already exists
        if clean_name in existing_templates:
            continue
            
        try:
            shutil.copy2(pkl_file, target_path)
            print(f"✅ Copied: {clean_name}")
            copied_count += 1
        except Exception as e:
            print(f"❌ Failed to copy {pkl_file.name}: {e}")
    
    if copied_count > 0:
        print(f"\n🎉 Successfully copied {copied_count} new templates!")
    else:
        print("\n💡 No new templates found.")
    
    # Show current templates
    current_templates = list(templates_dir.glob("*.pkl"))
    if current_templates:
        print(f"\n📋 Available templates ({len(current_templates)}):")
        for template in sorted(current_templates):
            size_kb = template.stat().st_size / 1024
            print(f"   • {template.name} ({size_kb:.1f} KB)")
    
    return copied_count

if __name__ == "__main__":
    print("🚀 LivePortrait Template Manager v5.0")
    print("=" * 50)
    
    try:
        copied = copy_templates()
        print("\n✨ Template collection complete!")
        
        if copied > 0:
            print("\n💡 Tip: Run the batch processor and select 'Template (.pkl)' for faster processing!")
            
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
    
    input("\nPress Enter to exit...")
