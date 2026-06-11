@echo off
chcp 65001 >nul
title 上传项目到 GitHub

echo ============================================
echo       上传项目到 GitHub
echo ============================================
echo.
echo 正在删除残留的 .git 文件夹...
if exist "D:\church\Documents\学习6\Teamwork\.git" (
    rmdir /s /q "D:\church\Documents\学习6\Teamwork\.git" 2>nul
    if exist "D:\church\Documents\学习6\Teamwork\.git" (
        echo ⚠ 无法自动删除 .git，请手动删除
        echo 打开 D:\church\Documents\学习6\Teamwork
        echo 找到 .git 文件夹按 Delete
        pause
        exit /b
    )
)
echo ✅ .git 已清除

echo.
echo 正在初始化 Git 仓库...
cd /d "D:\church\Documents\学习6\Teamwork"
git init
if %ERRORLEVEL% NEQ 0 (
    echo ❌ Git 初始化失败
    pause
    exit /b
)
echo ✅ Git 初始化成功

echo.
echo 正在添加文件...
git add -A
git commit -m "初始提交：团队协作项目"
echo ✅ 提交成功

echo.
echo 正在连接 GitHub 仓库...
git remote add origin https://github.com/Church309/teamwork.git 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo ⚠ remote 已存在，更新地址
    git remote set-url origin https://github.com/Church309/teamwork.git
)
echo ✅ 连接成功

echo.
echo 正在推送到 GitHub...
echo 如果弹出登录窗口：
echo   用户名: Church309
echo   密码:   粘贴你的 Token
echo.
git push -u origin main

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ============================================
    echo   ✅ 上传成功！
    echo   打开 https://github.com/Church309/teamwork
    echo   查看你的项目
    echo ============================================
) else (
    echo.
    echo ❌ 推送失败，试试手动操作：
    echo.
    echo cd /d "D:\church\Documents\学习6\Teamwork"
    echo git push -u origin main
)

pause
