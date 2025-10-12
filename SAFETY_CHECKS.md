# 🛡️ Safety Checks for File Modifications

## Critical Issue Prevention

### Problem Identified
- **Issue**: Changes were committed to wrong file (Python file instead of HTML template)
- **Impact**: UI changes not visible, confusion, wasted time
- **Root Cause**: No validation of which file is being modified

### Mandatory Safety Checks

#### 1. **File Path Validation**
```bash
# ALWAYS verify the target file before making changes
echo "Target file: $(pwd)/templates/dashboard_v2_clean.html"
ls -la templates/dashboard_v2_clean.html
```

#### 2. **Pre-Change Verification**
```bash
# Check current content before modification
grep -n "search_string" templates/dashboard_v2_clean.html
```

#### 3. **Post-Change Verification**
```bash
# Verify changes were applied to correct file
git diff templates/dashboard_v2_clean.html
git status
```

#### 4. **Commit Validation**
```bash
# Check what files are being committed
git diff --cached --name-only
git show --name-only HEAD
```

### File Type Guidelines

| Change Type | Target File | Verification |
|-------------|-------------|--------------|
| UI/Frontend | `templates/dashboard_v2_clean.html` | Check HTML structure |
| Backend Logic | `iss_speed_html_dashboard_v2_clean.py` | Check Python syntax |
| Tests | `realistic_test.py` | Check test structure |
| Configuration | `*.json`, `*.md` | Check file format |

### Emergency Recovery

If wrong file is modified:
```bash
# 1. Check what was changed
git show HEAD

# 2. Revert if needed
git reset --soft HEAD~1

# 3. Apply to correct file
# (make changes to correct file)

# 4. Commit with correct message
git add correct_file
git commit -m "fix: Apply changes to correct file"
```

### Prevention Checklist

Before any file modification:
- [ ] Verify target file path
- [ ] Check file exists and is readable
- [ ] Confirm file type matches change type
- [ ] Preview changes before applying
- [ ] Verify changes after applying
- [ ] Check git status before committing
- [ ] Validate commit includes correct files only

### File Modification Protocol

1. **Identify Target**: Clearly specify which file needs changes
2. **Verify Path**: Confirm absolute path to target file
3. **Check Content**: Preview current content around change area
4. **Apply Changes**: Make modifications with clear feedback
5. **Validate Results**: Verify changes in correct file
6. **Commit Safely**: Check git status before committing
7. **Test Changes**: Verify changes work as expected

### Warning Signs

Stop and verify if you see:
- Tool reports "file not found" but changes seem to apply
- Commit shows different file than expected
- Changes don't appear in UI after restart
- Git status shows unexpected files modified
- Search results show matches in wrong file type

### Recovery Commands

```bash
# Check what files were actually changed
git diff HEAD~1 HEAD --name-only

# See what was changed in each file
git diff HEAD~1 HEAD

# Revert specific file if wrong
git checkout HEAD~1 -- wrong_file.py

# Apply to correct file
# (make changes to correct file)

# Commit fix
git add correct_file.html
git commit -m "fix: Apply changes to correct file"
```
