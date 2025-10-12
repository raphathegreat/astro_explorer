#!/bin/bash

# Quick Enhancement Workflow Script
# Usage: ./quick_enhancement.sh "Description of your enhancement"

if [ $# -eq 0 ]; then
    echo "Usage: ./quick_enhancement.sh 'Description of your enhancement'"
    exit 1
fi

DESCRIPTION="$1"
BRANCH_NAME="enhancement/$(date +%Y%m%d_%H%M)_$(echo "$DESCRIPTION" | tr ' ' '_' | tr '[:upper:]' '[:lower:]' | cut -c1-30)"

echo "🚀 Starting quick enhancement workflow..."
echo "📝 Description: $DESCRIPTION"
echo "🌿 Branch: $BRANCH_NAME"

# Create and switch to branch
git checkout -b "$BRANCH_NAME"
echo "✅ Created branch: $BRANCH_NAME"

echo ""
echo "📝 Make your changes now, then press Enter to continue..."
read -r

# Run tests
echo "🧪 Running tests..."
if python realistic_test.py; then
    echo "✅ Tests passed!"
    
    # Commit changes
    echo "📦 Committing changes..."
    git add .
    git commit -m "feat: $DESCRIPTION"
    
    # Push to GitHub
    echo "🚀 Pushing to GitHub..."
    git push origin "$BRANCH_NAME"
    
    # Merge to main
    echo "🔄 Merging to main..."
    git checkout main
    git merge "$BRANCH_NAME"
    git push origin main
    
    # Clean up
    echo "🧹 Cleaning up..."
    git branch -d "$BRANCH_NAME"
    
    echo ""
    echo "🎉 Enhancement completed successfully!"
    echo "📊 View your changes at: https://github.com/raphathegreat/astro_explorer"
    
else
    echo "❌ Tests failed! Please fix the issues before committing."
    echo "🔄 You're still on branch: $BRANCH_NAME"
    echo "💡 Fix the issues and run this script again, or:"
    echo "   git checkout main && git branch -D $BRANCH_NAME"
fi
