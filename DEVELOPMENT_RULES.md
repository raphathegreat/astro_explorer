# 🚀 AstroPi Explorer - Development Rules

## 📋 **Core Development Workflow**

### 🎯 **Rule 1: Always Work on Feature Branches**
- **NEVER** make changes directly on the `main` branch
- **ALWAYS** create a new branch from `main` for any changes
- Use descriptive branch names: `feature/description`, `fix/issue`, `enhancement/feature`

### 🔄 **Rule 2: Branch Creation Process**
```bash
# 1. Ensure main is up to date
git checkout main
git pull origin main

# 2. Create and switch to new branch
git checkout -b feature/your-feature-name

# 3. Make your changes
# ... edit files ...

# 4. Test your changes
python3 realistic_test.py

# 5. Commit changes
git add .
git commit -m "feat: Add your feature description"

# 6. Push branch to GitHub
git push origin feature/your-feature-name
```

### ✅ **Rule 3: Testing Requirements**
- **MUST** run the test suite before merging: `python3 realistic_test.py`
- **MUST** achieve 100% test pass rate (51/51 tests)
- **MUST** test locally to ensure UI/functionality works as expected
- **MUST** verify Railway deployment works (if applicable)

### 🔀 **Rule 4: Merge to Production Process**
```bash
# 1. Ensure all tests pass
python3 realistic_test.py

# 2. Switch to main branch
git checkout main
git pull origin main

# 3. Merge feature branch
git merge feature/your-feature-name

# 4. Push to production
git push origin main

# 5. Clean up feature branch
git branch -d feature/your-feature-name
git push origin --delete feature/your-feature-name
```

## 🏷️ **Branch Naming Conventions**

### 📝 **Feature Branches**
- `feature/dashboard-ui-improvements`
- `feature/add-new-algorithm`
- `feature/performance-optimization`

### 🐛 **Bug Fix Branches**
- `fix/dropdown-not-working`
- `fix/railway-deployment-issue`
- `fix/cache-corruption-bug`

### ⚡ **Enhancement Branches**
- `enhancement/improve-test-coverage`
- `enhancement/optimize-image-processing`
- `enhancement/add-user-guidance`

### 🔧 **Hotfix Branches** (for urgent production fixes)
- `hotfix/critical-security-patch`
- `hotfix/railway-crash-fix`

## 📊 **Testing Standards**

### 🧪 **Test Suite Requirements**
- **All 51 tests MUST pass** before merging to main
- **No test modifications** to make tests pass - fix the actual issues
- **Test isolation** - ensure tests don't interfere with each other
- **Coverage validation** - maintain ~90% code coverage

### 🔍 **Local Testing Checklist**
- [ ] Run full test suite: `python3 realistic_test.py`
- [ ] Test UI functionality manually
- [ ] Verify file upload works (if applicable)
- [ ] Check dropdown functionality
- [ ] Test algorithm processing
- [ ] Verify cache system works

### ☁️ **Railway Testing Checklist**
- [ ] Verify deployment succeeds
- [ ] Test health endpoint: `/health`
- [ ] Verify UI matches local version
- [ ] Test file upload functionality
- [ ] Check all sections work correctly

## 📝 **Commit Message Standards**

### 🎯 **Format**
```
type: Brief description (50 chars max)

Detailed explanation of what was changed and why.
Include any breaking changes or important notes.
```

### 📋 **Types**
- `feat:` - New feature
- `fix:` - Bug fix
- `enhancement:` - Improvement to existing feature
- `refactor:` - Code refactoring
- `test:` - Test improvements
- `docs:` - Documentation updates
- `chore:` - Maintenance tasks

### ✅ **Examples**
```
feat: Add file upload functionality for Railway deployment

- Implemented multi-file upload with progress tracking
- Added environment detection for local vs Railway
- Improved user guidance with helpful messaging
- All 51 tests passing
```

```
fix: Resolve dropdown population issue on Railway

- Fixed empty dropdown when no photos-* folders exist
- Added helpful messaging for users
- Improved file upload UI styling
- Verified with comprehensive testing
```

## 🚫 **Prohibited Actions**

### ❌ **Never Do These**
- Make direct commits to `main` branch
- Merge branches without running tests
- Skip testing for "small" changes
- Force push to `main` branch
- Delete the `main` branch
- Work on multiple features in the same branch

### ⚠️ **Emergency Exceptions**
- Only for critical production issues
- Must create `hotfix/` branch immediately after
- Must run tests before merging
- Must document the emergency in commit message

## 🔄 **Railway Deployment Process**

### 🚀 **Automatic Deployment**
- Railway automatically deploys when `main` branch is updated
- No manual deployment needed
- Monitor Railway logs for deployment status

### 🔍 **Deployment Verification**
1. Check Railway dashboard for build status
2. Test health endpoint: `https://astropi-explorer.up.railway.app/health`
3. Verify UI matches local version
4. Test core functionality

## 📚 **Documentation Requirements**

### 📖 **Code Documentation**
- Document complex algorithms and functions
- Include docstrings for all public functions
- Maintain README.md with setup instructions
- Update DEPLOYMENT_GUIDE.md for deployment changes

### 🧪 **Test Documentation**
- Document test scenarios and expected outcomes
- Maintain test coverage reports
- Document any test modifications or additions

## 🎯 **Quality Gates**

### ✅ **Pre-Merge Checklist**
- [ ] All tests pass (51/51)
- [ ] Code follows style guidelines
- [ ] Documentation updated
- [ ] Commit messages follow standards
- [ ] Local testing completed
- [ ] Railway deployment verified (if applicable)

### 🚀 **Post-Merge Verification**
- [ ] Railway deployment successful
- [ ] Health endpoint responding
- [ ] UI matches expected design
- [ ] Core functionality working
- [ ] No regressions detected

## 🔧 **Development Environment Setup**

### 💻 **Local Development**
```bash
# Clone repository
git clone https://github.com/raphathegreat/astro_explorer.git
cd astro_explorer

# Create feature branch
git checkout -b feature/your-feature-name

# Run tests
python3 realistic_test.py

# Start development server
python3 iss_speed_html_dashboard_v2_clean.py
```

### ☁️ **Railway Integration**
- Connected to GitHub repository
- Auto-deploys from `main` branch
- Environment variables configured
- Health checks enabled

## 📞 **Support and Escalation**

### 🆘 **When to Ask for Help**
- Tests failing and unsure how to fix
- Railway deployment issues
- Complex algorithm or UI problems
- Performance optimization needs

### 📋 **Information to Provide**
- Branch name and commit hash
- Test output and error messages
- Railway deployment logs
- Steps to reproduce issues

---

## 🎉 **Remember: Quality First, Speed Second**

**Every change should be:**
- ✅ **Tested** thoroughly
- ✅ **Documented** properly  
- ✅ **Reviewed** before merging
- ✅ **Deployed** safely to production

**This workflow ensures:**
- 🛡️ **Stable production** environment
- 🧪 **Reliable testing** process
- 🔄 **Smooth deployments** to Railway
- 👥 **Team collaboration** best practices