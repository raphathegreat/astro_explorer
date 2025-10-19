# 🚀 AstroPi Explorer - Quick Reference Guide

## 🔥 **Most Common Tasks**

### 🆕 **Starting New Feature**
```bash
git checkout main
git pull origin main
git checkout -b feature/your-feature-name
# Make changes...
python3 realistic_test.py  # MUST pass all tests
git add .
git commit -m "feat: Your feature description"
git push origin feature/your-feature-name
```

### 🐛 **Fixing Bug**
```bash
git checkout main
git pull origin main
git checkout -b fix/bug-description
# Fix the bug...
python3 realistic_test.py  # MUST pass all tests
git add .
git commit -m "fix: Bug description"
git push origin fix/bug-description
```

### 🔀 **Merging to Production**
```bash
# After feature is tested and ready
git checkout main
git pull origin main
git merge feature/your-feature-name
git push origin main
git branch -d feature/your-feature-name
git push origin --delete feature/your-feature-name
```

### 🧪 **Testing**
```bash
# Run full test suite
python3 realistic_test.py

# Expected output: 51 tests, 100% pass rate
# If tests fail, fix the issues, don't modify tests
```

### 🚀 **Railway Deployment Check**
```bash
# Check if Railway is deploying latest changes
curl https://astropi-explorer.up.railway.app/health

# Expected: {"status": "healthy", "message": "AstroPi Explorer Dashboard is running"}
```

## 📋 **Branch Types & Naming**

| Type | Prefix | Example | Use Case |
|------|--------|---------|----------|
| Feature | `feature/` | `feature/add-dark-mode` | New functionality |
| Bug Fix | `fix/` | `fix/dropdown-bug` | Bug fixes |
| Enhancement | `enhancement/` | `enhancement/improve-ui` | Improvements |
| Hotfix | `hotfix/` | `hotfix/critical-fix` | Urgent production fixes |

## 🎯 **Commit Message Template**
```
type: Brief description (50 chars max)

- What was changed
- Why it was changed  
- Any important notes
- Test results: "All 51 tests passing"
```

## ⚡ **Quick Commands**

### 🔍 **Check Status**
```bash
git status                    # See current changes
git log --oneline -5         # See recent commits
git branch -a                # See all branches
```

### 🧹 **Clean Up**
```bash
git branch -d old-branch     # Delete local branch
git push origin --delete old-branch  # Delete remote branch
git checkout main && git pull origin main  # Reset to latest main
```

### 🚨 **Emergency Rollback**
```bash
# If Railway deployment is broken
git checkout main
git revert HEAD              # Undo last commit
git push origin main         # Deploy rollback
```

## 📊 **Testing Checklist**

### ✅ **Before Every Merge**
- [ ] `python3 realistic_test.py` → 51/51 tests pass
- [ ] Local UI testing completed
- [ ] Railway deployment verified
- [ ] No console errors in browser
- [ ] All sections working correctly

### 🔍 **Railway Verification**
- [ ] Health endpoint: `/health` returns healthy
- [ ] Main page loads correctly
- [ ] UI matches local version
- [ ] File upload works (if applicable)
- [ ] Dropdown populates correctly

## 🎨 **UI Development**

### 📁 **Templates**
- `templates/dashboard_v2_clean.html` - Modern UI (preferred)
- `templates/dashboard.html` - Traditional UI

### 🔧 **Switching Templates**
```python
# In iss_speed_html_dashboard_v2_clean.py
@app.route('/')
def index():
    return render_template('dashboard_v2_clean.html')  # Modern
    # return render_template('dashboard.html')         # Traditional
```

## 🚫 **Never Do These**
- ❌ Commit directly to `main`
- ❌ Merge without running tests
- ❌ Skip testing for "small" changes
- ❌ Force push to `main`
- ❌ Work on multiple features in one branch

## 🆘 **When Things Go Wrong**

### 🧪 **Tests Failing**
1. Read error messages carefully
2. Check if it's a global state issue
3. Fix the root cause, don't modify tests
4. Run tests again until 51/51 pass

### ☁️ **Railway Issues**
1. Check Railway dashboard for build logs
2. Verify health endpoint: `/health`
3. Check if latest commit was deployed
4. Look for environment variable issues

### 🔄 **Merge Conflicts**
1. `git checkout main && git pull origin main`
2. `git checkout your-branch`
3. `git merge main` (resolve conflicts)
4. Test thoroughly before pushing

---

## 🎯 **Remember: Test First, Deploy Second!**

**Every change must pass the test suite before reaching production.**
