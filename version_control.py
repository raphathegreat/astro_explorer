#!/usr/bin/env python3
"""
Version Control System for ISS Speed Analysis Dashboard
Creates timestamped backups before making changes
"""

import os
import shutil
import datetime
from pathlib import Path

def create_version_backup(description="Manual backup"):
    """
    Create a timestamped backup of the current working files
    """
    # Get current timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create version folder name
    version_folder = f"v{timestamp}_{description.replace(' ', '_')}"
    version_path = Path("versions") / version_folder
    
    # Create the version directory
    version_path.mkdir(parents=True, exist_ok=True)
    
    # Files to backup
    files_to_backup = [
        "iss_speed_html_dashboard_v2_clean.py",
        "templates/dashboard_v2_clean.html",
        "realistic_test.py",
        "DEVELOPMENT_RULES.md",
        "README_CLEAN_VERSION.md"
    ]
    
    # Create templates subdirectory if it doesn't exist
    templates_path = version_path / "templates"
    templates_path.mkdir(exist_ok=True)
    
    # Copy files
    backed_up_files = []
    for file_path in files_to_backup:
        source = Path(file_path)
        if source.exists():
            if file_path.startswith("templates/"):
                dest = version_path / file_path
            else:
                dest = version_path / source.name
            
            shutil.copy2(source, dest)
            backed_up_files.append(file_path)
            print(f"✅ Backed up: {file_path}")
        else:
            print(f"⚠️  File not found: {file_path}")
    
    # Create version info file
    version_info = {
        "timestamp": timestamp,
        "description": description,
        "files_backed_up": backed_up_files,
        "created_by": "version_control.py"
    }
    
    import json
    with open(version_path / "version_info.json", "w") as f:
        json.dump(version_info, f, indent=2)
    
    print(f"\n🎯 Version backup created: {version_folder}")
    print(f"📁 Location: {version_path}")
    print(f"📄 Files backed up: {len(backed_up_files)}")
    
    return version_path

def list_versions():
    """
    List all available versions
    """
    versions_dir = Path("versions")
    if not versions_dir.exists():
        print("❌ No versions directory found")
        return []
    
    versions = []
    for version_folder in sorted(versions_dir.iterdir()):
        if version_folder.is_dir():
            version_info_file = version_folder / "version_info.json"
            if version_info_file.exists():
                import json
                with open(version_info_file, "r") as f:
                    info = json.load(f)
                versions.append((version_folder.name, info))
            else:
                versions.append((version_folder.name, {"description": "Legacy backup"}))
    
    print("\n📋 Available Versions:")
    print("-" * 80)
    for version_name, info in versions:
        timestamp = info.get("timestamp", "Unknown")
        description = info.get("description", "No description")
        files_count = len(info.get("files_backed_up", []))
        print(f"📁 {version_name}")
        print(f"   📅 {timestamp} | 📝 {description} | 📄 {files_count} files")
        print()
    
    return versions

def restore_version(version_name):
    """
    Restore files from a specific version
    """
    version_path = Path("versions") / version_name
    if not version_path.exists():
        print(f"❌ Version {version_name} not found")
        return False
    
    # Read version info
    version_info_file = version_path / "version_info.json"
    if version_info_file.exists():
        import json
        with open(version_info_file, "r") as f:
            info = json.load(f)
        files_to_restore = info.get("files_backed_up", [])
    else:
        # Legacy backup - restore all files in the directory
        files_to_restore = []
        for file_path in version_path.rglob("*"):
            if file_path.is_file() and file_path.name != "version_info.json":
                relative_path = file_path.relative_to(version_path)
                files_to_restore.append(str(relative_path))
    
    print(f"🔄 Restoring version: {version_name}")
    print(f"📄 Files to restore: {len(files_to_restore)}")
    
    # Restore files
    restored_files = []
    for file_path in files_to_restore:
        source = version_path / file_path
        if source.exists():
            # Create destination directory if needed
            dest = Path(file_path)
            dest.parent.mkdir(parents=True, exist_ok=True)
            
            shutil.copy2(source, dest)
            restored_files.append(file_path)
            print(f"✅ Restored: {file_path}")
        else:
            print(f"⚠️  File not found in backup: {file_path}")
    
    print(f"\n🎯 Version restored: {version_name}")
    print(f"📄 Files restored: {len(restored_files)}")
    
    return True

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python version_control.py backup [description]")
        print("  python version_control.py list")
        print("  python version_control.py restore <version_name>")
        sys.exit(1)
    
    command = sys.argv[1]
    
    if command == "backup":
        description = sys.argv[2] if len(sys.argv) > 2 else "Manual backup"
        create_version_backup(description)
    
    elif command == "list":
        list_versions()
    
    elif command == "restore":
        if len(sys.argv) < 3:
            print("❌ Please specify version name to restore")
            sys.exit(1)
        version_name = sys.argv[2]
        restore_version(version_name)
    
    else:
        print(f"❌ Unknown command: {command}")
        sys.exit(1)
