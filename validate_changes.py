#!/usr/bin/env python3
"""
File Change Validation Script
Prevents changes from being applied to wrong files
"""

import os
import sys
import subprocess
from pathlib import Path

def validate_file_change(target_file, change_type, description):
    """
    Validate that changes are being applied to the correct file
    
    Args:
        target_file: Path to the file that should be modified
        change_type: Type of change (ui, backend, test, config)
        description: Description of the change being made
    """
    
    print(f"🔍 Validating file change: {description}")
    print(f"📁 Target file: {target_file}")
    print(f"🔧 Change type: {change_type}")
    
    # Check if file exists
    if not os.path.exists(target_file):
        print(f"❌ ERROR: Target file does not exist: {target_file}")
        return False
    
    # Check file type matches change type
    file_extension = Path(target_file).suffix.lower()
    
    type_mapping = {
        'ui': ['.html', '.css', '.js'],
        'backend': ['.py'],
        'test': ['.py'],
        'config': ['.json', '.md', '.txt', '.yml', '.yaml']
    }
    
    if change_type in type_mapping:
        if file_extension not in type_mapping[change_type]:
            print(f"❌ ERROR: File type mismatch!")
            print(f"   Expected: {type_mapping[change_type]}")
            print(f"   Actual: {file_extension}")
            return False
    
    # Check file is readable
    try:
        with open(target_file, 'r') as f:
            content = f.read(100)  # Read first 100 chars
        print(f"✅ File is readable")
    except Exception as e:
        print(f"❌ ERROR: Cannot read file: {e}")
        return False
    
    # Show file info
    stat = os.stat(target_file)
    print(f"📊 File size: {stat.st_size} bytes")
    print(f"📅 Last modified: {stat.st_mtime}")
    
    print(f"✅ File validation passed!")
    return True

def check_git_status():
    """Check git status to see what files are staged/modified"""
    try:
        result = subprocess.run(['git', 'status', '--porcelain'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            if lines and lines[0]:
                print("📋 Git status:")
                for line in lines:
                    print(f"   {line}")
            else:
                print("📋 No changes in git status")
        else:
            print(f"⚠️  Could not check git status: {result.stderr}")
    except Exception as e:
        print(f"⚠️  Error checking git status: {e}")

def validate_commit():
    """Validate that the correct files are being committed"""
    try:
        # Check staged files
        result = subprocess.run(['git', 'diff', '--cached', '--name-only'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            staged_files = result.stdout.strip().split('\n')
            if staged_files and staged_files[0]:
                print("📦 Files staged for commit:")
                for file in staged_files:
                    print(f"   {file}")
            else:
                print("📦 No files staged for commit")
        else:
            print(f"⚠️  Could not check staged files: {result.stderr}")
    except Exception as e:
        print(f"⚠️  Error checking staged files: {e}")

def main():
    """Main validation function"""
    if len(sys.argv) < 4:
        print("Usage: python validate_changes.py <target_file> <change_type> <description>")
        print("Change types: ui, backend, test, config")
        print("Example: python validate_changes.py templates/dashboard_v2_clean.html ui 'Update filter descriptions'")
        sys.exit(1)
    
    target_file = sys.argv[1]
    change_type = sys.argv[2]
    description = sys.argv[3]
    
    print("🛡️  FILE CHANGE VALIDATION")
    print("=" * 50)
    
    # Validate the file change
    if not validate_file_change(target_file, change_type, description):
        print("\n❌ VALIDATION FAILED - DO NOT PROCEED")
        sys.exit(1)
    
    print("\n📋 Current git status:")
    check_git_status()
    
    print("\n✅ VALIDATION PASSED - Safe to proceed")
    print("\n💡 Remember to:")
    print("   1. Verify changes after applying")
    print("   2. Check git status before committing")
    print("   3. Validate commit includes correct files only")

if __name__ == "__main__":
    main()
