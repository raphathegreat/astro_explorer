# Version Control Workflow

## 🛡️ **Safety First - Always Backup Before Changes**

### 📋 **Before Making ANY Code Changes:**

1. **Create a backup:**
   ```bash
   python3 version_control.py backup "Description of what you're about to change"
   ```

2. **Verify backup was created:**
   ```bash
   python3 version_control.py list
   ```

### 🔄 **Version Control Commands:**

#### **Create Backup:**
```bash
python3 version_control.py backup "Fix statistics calculation bug"
python3 version_control.py backup "Add new feature X"
python3 version_control.py backup "Debug logging issue"
```

#### **List All Versions:**
```bash
python3 version_control.py list
```

#### **Restore Previous Version:**
```bash
python3 version_control.py restore v20251011_113303_Before_fixing_statistics_calculation_bug
```

### 📁 **Version Structure:**
```
versions/
├── v20251011_113303_Before_fixing_statistics_calculation_bug/
│   ├── iss_speed_html_dashboard_v2_clean.py
│   ├── templates/
│   │   └── dashboard_v2_clean.html
│   ├── realistic_test.py
│   ├── DEVELOPMENT_RULES.md
│   ├── README_CLEAN_VERSION.md
│   └── version_info.json
└── v20251011_114500_Add_new_feature/
    └── ...
```

### 🎯 **Best Practices:**

1. **Always backup before changes** - No exceptions
2. **Use descriptive backup names** - "Fix bug X" not "backup"
3. **Test changes thoroughly** - Before making more changes
4. **Keep recent versions** - Don't delete recent backups
5. **Document what changed** - In backup description

### 🚨 **Emergency Rollback:**

If something breaks:
1. **Stop the application**
2. **Restore last working version:**
   ```bash
   python3 version_control.py restore v20251011_113303_Before_fixing_statistics_calculation_bug
   ```
3. **Restart application**
4. **Verify it's working**

### 📊 **Version Info File:**
Each backup includes a `version_info.json` with:
- Timestamp
- Description
- List of backed up files
- Creation metadata

## 🔧 **Integration with Development Rules:**

This version control system supports **Rule 2: Version Archiving** from `DEVELOPMENT_RULES.md` by providing:
- Automated timestamped backups
- Easy rollback capability
- Organized version history
- Safety net for experimentation
