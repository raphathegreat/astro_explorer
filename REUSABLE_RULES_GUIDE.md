# 📋 Reusable Development Rules Guide

This document identifies which rules files from this project can be reused in other projects, and which need customization.

## ✅ **Highly Reusable Files (Use as-is or with minor customization)**

### 1. **DEVELOPMENT_RULES.md** ⭐⭐⭐⭐⭐
**Reusability: 90%**

**What to reuse:**
- ✅ Branch workflow (always use feature branches)
- ✅ Branch naming conventions (feature/, fix/, enhancement/, hotfix/)
- ✅ Commit message standards (feat:, fix:, enhancement:, etc.)
- ✅ Merge to production process
- ✅ Testing requirements workflow
- ✅ Prohibited actions list
- ✅ Quality gates and checklists

**What to customize:**
- ❌ Test file name (`realistic_test.py` → your test file)
- ❌ Test count (51 tests → your test count)
- ❌ Version file format (`version.py` → your versioning system)
- ❌ Deployment platform (Railway → your platform)
- ❌ Specific file paths

**How to adapt:**
1. Replace `realistic_test.py` with your test command
2. Update test count references
3. Remove or adapt version.py section if not using it
4. Update deployment platform references

---

### 2. **GITHUB_WORKFLOW.md** ⭐⭐⭐⭐
**Reusability: 85%**

**What to reuse:**
- ✅ Daily development process
- ✅ Branch creation workflow
- ✅ Testing before push
- ✅ Merge process
- ✅ Git commands and examples

**What to customize:**
- ❌ Test command (`python realistic_test.py` → your test command)
- ❌ Project-specific file names

**How to adapt:**
1. Replace test commands with your project's test suite
2. Update any project-specific references

---

### 3. **SAFETY_CHECKS.md** ⭐⭐⭐⭐⭐
**Reusability: 95%**

**What to reuse:**
- ✅ File path validation process
- ✅ Pre/post-change verification
- ✅ Commit validation
- ✅ File modification protocol
- ✅ Emergency recovery commands
- ✅ Prevention checklist

**What to customize:**
- ❌ Specific file paths (templates/, Python files)
- ❌ File type guidelines (HTML, Python → your file types)

**How to adapt:**
1. Update file type guidelines table with your project's file types
2. Replace example file paths with your project structure

---

## ⚠️ **Partially Reusable Files (Need significant customization)**

### 4. **SAFE_DEVELOPMENT_WORKFLOW.md** ⭐⭐⭐
**Reusability: 60%**

**What to reuse:**
- ✅ Core safety concepts
- ✅ Validation workflow
- ✅ Recovery procedures

**What to customize:**
- ❌ Specific file names (`templates/dashboard_v2_clean.html`)
- ❌ Validation script (`validate_changes.py`)
- ❌ Project-specific file structure

**How to adapt:**
1. Remove references to specific files
2. Generalize validation process
3. Update with your project's file structure

---

### 5. **VERSION_CONTROL_WORKFLOW.md** ⭐⭐
**Reusability: 40%**

**What to reuse:**
- ✅ Backup before changes concept
- ✅ Version structure concept

**What to customize:**
- ❌ `version_control.py` script (project-specific)
- ❌ Specific file paths in version structure
- ❌ Backup tool implementation

**Note:** This is very project-specific. Consider using git tags or a generic backup system instead.

---

## 🎯 **Recommended Reusable Template**

### **GENERIC_DEVELOPMENT_RULES.md** (Create this for new projects)

```markdown
# Development Rules Template

## Core Workflow

### Rule 1: Always Work on Feature Branches
- NEVER make changes directly on `main`/`master`
- ALWAYS create a new branch for any changes
- Use descriptive branch names: `feature/description`, `fix/issue`, `enhancement/feature`

### Rule 2: Branch Creation Process
1. Ensure main is up to date: `git checkout main && git pull origin main`
2. Create new branch: `git checkout -b feature/your-feature-name`
3. Make your changes
4. Test your changes: `[YOUR_TEST_COMMAND]`
5. Commit: `git commit -m "feat: Description"`
6. Push: `git push origin feature/your-feature-name`

### Rule 3: Testing Requirements
- MUST run test suite before merging
- MUST achieve 100% test pass rate
- MUST test locally
- MUST verify deployment (if applicable)

### Rule 4: Merge to Production
1. Ensure all tests pass
2. Switch to main: `git checkout main && git pull origin main`
3. Merge: `git merge feature/your-feature-name`
4. Push: `git push origin main`
5. Clean up: `git branch -d feature/your-feature-name`

## Branch Naming Conventions
- `feature/description` - New features
- `fix/issue` - Bug fixes
- `enhancement/feature` - Improvements
- `hotfix/critical` - Urgent production fixes

## Commit Message Standards
Format: `type: Brief description`

Types:
- `feat:` - New feature
- `fix:` - Bug fix
- `enhancement:` - Improvement
- `refactor:` - Code refactoring
- `test:` - Test improvements
- `docs:` - Documentation
- `chore:` - Maintenance

## Prohibited Actions
- ❌ Direct commits to main
- ❌ Merge without tests
- ❌ Force push to main
- ❌ Multiple features in one branch
```

---

## 📦 **Files to Copy to New Projects**

### **Essential Files (Copy & Customize):**
1. ✅ `DEVELOPMENT_RULES.md` - Core workflow (customize test commands)
2. ✅ `SAFETY_CHECKS.md` - File modification safety (customize file types)
3. ✅ `GITHUB_WORKFLOW.md` - Daily workflow (customize test commands)

### **Optional Files (Copy if relevant):**
4. ⚠️ `SAFE_DEVELOPMENT_WORKFLOW.md` - If you have similar file structure issues
5. ⚠️ `VERSION_CONTROL_WORKFLOW.md` - Only if using similar version control system

### **Don't Copy (Project-Specific):**
- ❌ `DEPLOYMENT_GUIDE.md` - Too project-specific
- ❌ `ML_MODEL_SETUP.md` - Project-specific
- ❌ `RASPBERRY_PI_SETUP.md` - Project-specific
- ❌ `CLEANUP_SUMMARY.md` - Project-specific cleanup

---

## 🚀 **Quick Start: Setting Up Rules for New Project**

### Step 1: Copy Core Files
```bash
# Copy to your new project
cp DEVELOPMENT_RULES.md ../new-project/
cp SAFETY_CHECKS.md ../new-project/
cp GITHUB_WORKFLOW.md ../new-project/
```

### Step 2: Customize for Your Project
1. **Update test commands:**
   - Replace `python3 realistic_test.py` with your test command
   - Update test count references

2. **Update file types:**
   - In `SAFETY_CHECKS.md`, update the file type guidelines table
   - Replace example file paths with your project structure

3. **Update deployment:**
   - Replace Railway references with your deployment platform
   - Update deployment verification steps

4. **Remove project-specific sections:**
   - Remove version.py references if not using it
   - Remove project-specific file paths

### Step 3: Add Project-Specific Rules
Add any project-specific rules to the files:
- Technology stack specifics
- Project file structure
- Team-specific workflows

---

## 📊 **Reusability Summary**

| File | Reusability | Customization Needed |
|------|-------------|---------------------|
| `DEVELOPMENT_RULES.md` | ⭐⭐⭐⭐⭐ 90% | Test commands, version system |
| `SAFETY_CHECKS.md` | ⭐⭐⭐⭐⭐ 95% | File types, file paths |
| `GITHUB_WORKFLOW.md` | ⭐⭐⭐⭐ 85% | Test commands |
| `SAFE_DEVELOPMENT_WORKFLOW.md` | ⭐⭐⭐ 60% | File structure, validation |
| `VERSION_CONTROL_WORKFLOW.md` | ⭐⭐ 40% | Backup system, file paths |

---

## 💡 **Best Practices for Reusing**

1. **Start with the core workflow** - Copy `DEVELOPMENT_RULES.md` and customize
2. **Add safety checks** - Copy `SAFETY_CHECKS.md` and update file types
3. **Adapt to your stack** - Update test commands, deployment platform
4. **Keep it simple** - Don't copy everything, only what you need
5. **Evolve over time** - Add project-specific rules as you learn

---

## 🎯 **Recommended Minimal Set**

For a new project, start with these 3 files:
1. `DEVELOPMENT_RULES.md` (customized)
2. `SAFETY_CHECKS.md` (customized)
3. `GITHUB_WORKFLOW.md` (customized)

This gives you:
- ✅ Complete branch workflow
- ✅ Testing requirements
- ✅ Commit standards
- ✅ Merge process
- ✅ Safety checks
- ✅ File modification protocols

---

**Remember:** These rules are proven to work well. Adapt them to your project's needs, but keep the core principles:
- Always use feature branches
- Always test before merging
- Always follow commit message standards
- Always verify changes before committing




