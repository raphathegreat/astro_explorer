# Development Rules and Guidelines

This document contains the development rules and guidelines for the ISS Speed Analysis Dashboard project.

## Rule 1: Test-Driven Development (TDD)

- Create test cases before writing code
- Execute tests to validate the code
- Run the full test pack and ensure it's updated
- **Never amend tests to make code pass** - tests define requirements

## Rule 2: Version Archiving

- **ALWAYS create a backup before making ANY code changes** using the version control system
- Use: `python3 version_control.py backup "Description of changes"`
- See `VERSION_CONTROL_WORKFLOW.md` for detailed instructions
- Use: `python3 version_control.py restore <version_name>` to rollback if needed
- Maintain organized version history in `versions/` folder with timestamped backups

## Rule 3: Test Pack

- `realistic_test.py` is the designated test pack
- Execute and update it with new test cases
- Use it to validate all changes
- Ensure comprehensive test coverage

## Rule 4: Cache Maintenance for Section 2 Changes

When making changes to section 2 (algorithm parameters, processing options, or any configuration that affects data processing), the caching system must be updated accordingly if required.

This includes:
- Updating cache key generation to include new parameters
- Ensuring cache invalidation works with new parameter combinations
- Testing cache functionality with the new parameters
- Updating the comprehensive cache tests to cover new parameter combinations

The cache system must remain consistent with all section 2 functionality.

## Rule 5: Issue Investigation Protocol

**When the user raises an issue or reports a problem, ALWAYS check the application logs from the last restart before attempting to diagnose or fix the issue.**

### Log Files to Check:
- `dashboard_application.log` - Backend API calls, data processing, statistics
- `app_output.log` - Application startup and general output

### Benefits:
- Complete understanding of what happened without requiring user to copy-paste any data or console logs
- Logs provide exact API calls, parameters, data processing steps, and results
- Faster debugging and more accurate fixes

### Implementation:
- Logs are automatically generated when the application runs
- All API endpoints log their calls and responses
- Frontend actions are logged to browser console with `🎬 ACTION:` prefix
- Backend processing is logged with structured information

## Rule 6: GSD Configuration Default Behavior

- GSD (Ground Sample Distance) configuration should be **disabled by default**
- Users must explicitly enable custom GSD to override the default value
- This prevents unintended speed recalculations when no filters are applied
- Ensures consistent behavior when clicking refresh without applying filters

## Rule 7: Filter Application Logic

- Only send enabled filters to the backend
- Empty filter objects should be sent when no filters are applied
- This ensures the backend uses raw data when no filters are intended
- Prevents unintended filter applications

---

## How to Use These Rules

1. **Before making any changes**: Review relevant rules
2. **During development**: Follow TDD principles (Rule 1)
3. **When changing Section 2**: Update caching accordingly (Rule 4)
4. **When user reports issues**: Check logs first (Rule 5)
5. **After changes**: Run the test pack to validate (Rule 3)
6. **Before deployment**: Archive working versions (Rule 2)

## Log File Locations

- **Backend Logs**: `/Users/astropi/Backup_esaiss/dashboard_application.log`
- **Application Output**: `/Users/astropi/Backup_esaiss/app_output.log`
- **Frontend Logs**: Browser console (Developer Tools → Console)

## Quick Reference

| Rule | Purpose | When to Apply |
|------|---------|---------------|
| 1 | TDD | Before writing any code |
| 2 | Version Control | After successful changes |
| 3 | Testing | After any modifications |
| 4 | Cache Management | When changing Section 2 |
| 5 | Issue Debugging | When user reports problems |
| 6 | GSD Defaults | UI/UX consistency |
| 7 | Filter Logic | Frontend-backend communication |
