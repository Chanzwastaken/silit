# Git Commit Script for SILIT
# This script commits only the important files, excluding the dataset

# Add .gitignore first
git add .gitignore

# Add Python code
git add model/*.py
git add app/*.py

# Add documentation
git add README.md
git add SHOWCASE.md
git add PROJECT_SUMMARY.md
git add SOCIAL_MEDIA.md
git add requirements.txt

# Add assets (demo.gif and README)
git add assets/

# Check what will be committed
Write-Host "`nFiles to be committed:" -ForegroundColor Green
git status --short

# Commit
$commitMessage = Read-Host "`nEnter commit message (or press Enter for default)"
if ([string]::IsNullOrWhiteSpace($commitMessage)) {
    $commitMessage = "Initial commit: SILIT - Real-time Sign Language Translator"
}

git commit -m $commitMessage

Write-Host "`nCommit complete!" -ForegroundColor Green
Write-Host "Files committed: " -NoNewline
git diff --stat HEAD~1 HEAD

Write-Host "`n`nTo push to GitHub:" -ForegroundColor Yellow
Write-Host "  git push origin main" -ForegroundColor Cyan
