@echo off
:: 激活 Conda 环境
call activate.bat webapp

:: 检查激活是否成功
if "%CONDA_DEFAULT_ENV%" NEQ "webapp" (
    echo Failed to activate environment "webapp".
    exit /b 1
)

:: 执行 Python 脚本
python trans_excel.py multi 100

:: 退出 Conda 环境
call deactivate.bat

echo Done.
pause
