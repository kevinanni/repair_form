@echo off
setlocal enabledelayedexpansion

REM 定义项目根目录
set PROJECT_ROOT=%~dp0

REM 创建项目根目录下的子目录
mkdir "%PROJECT_ROOT%src"
mkdir "%PROJECT_ROOT%data"
mkdir "%PROJECT_ROOT%tests"
mkdir "%PROJECT_ROOT%docs"
mkdir "%PROJECT_ROOT%static"

REM 创建src目录下的子目录
mkdir "%PROJECT_ROOT%src\business_logic"
mkdir "%PROJECT_ROOT%src\models"
mkdir "%PROJECT_ROOT%src\services"
mkdir "%PROJECT_ROOT%src\templates"
mkdir "%PROJECT_ROOT%src\templates\pages"

REM 创建data目录下的子目录
mkdir "%PROJECT_ROOT%data\raw"
mkdir "%PROJECT_ROOT%data\processed"
mkdir "%PROJECT_ROOT%data\models"

REM 创建static目录下的子目录
mkdir "%PROJECT_ROOT%static\images"
mkdir "%PROJECT_ROOT%static\uploads"
mkdir "%PROJECT_ROOT%static\css"
mkdir "%PROJECT_ROOT%static\js"

REM 创建tests目录下的子目录
REM mkdir "%PROJECT_ROOT%tests"

REM 在相应目录下创建空文件
echo. > "%PROJECT_ROOT%app.py"
echo. > "%PROJECT_ROOT%requirements.txt"
echo. > "%PROJECT_ROOT%.gitignore"
echo. > "%PROJECT_ROOT%README.md"

echo. > "%PROJECT_ROOT%src\__init__.py"
echo. > "%PROJECT_ROOT%src\main.py"

echo. > "%PROJECT_ROOT%src\business_logic\__init__.py"
echo. > "%PROJECT_ROOT%src\business_logic\data_handling.py"
echo. > "%PROJECT_ROOT%src\business_logic\model_management.py"
echo. > "%PROJECT_ROOT%src\business_logic\utils.py"

echo. > "%PROJECT_ROOT%src\models\__init__.py"
echo. > "%PROJECT_ROOT%src\models\train.py"
echo. > "%PROJECT_ROOT%src\models\predict.py"

echo. > "%PROJECT_ROOT%src\services\__init__.py"
echo. > "%PROJECT_ROOT%src\services\api.py"
echo. > "%PROJECT_ROOT%src\services\utils.py"

echo. > "%PROJECT_ROOT%src\templates\__init__.py"
echo. > "%PROJECT_ROOT%src\templates\pages\dashboard.py"
echo. > "%PROJECT_ROOT%src\templates\pages\results.py"

echo. > "%PROJECT_ROOT%tests\__init__.py"
echo. > "%PROJECT_ROOT%tests\test_business_logic.py"
echo. > "%PROJECT_ROOT%tests\test_models.py"
echo. > "%PROJECT_ROOT%tests\test_services.py"

echo. > "%PROJECT_ROOT%docs\architecture.md"
echo. > "%PROJECT_ROOT%docs\deployment_guide.md"
echo. > "%PROJECT_ROOT%docs\user_manual.md"

echo Directory structure created successfully.
pause
endlocal