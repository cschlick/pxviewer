:: conda-build script for pxviewer (Windows). pxviewer is noarch, so the package is
:: normally built on Linux/macOS; this exists only so a Windows build can succeed when
:: the frontend bundle was produced beforehand (the vendored esbuild binary is not
:: available on Windows, so this does not build the frontend itself).
setlocal EnableDelayedExpansion

if not exist "%SRC_DIR%\frontend\build\index.js" (
    echo frontend\build\index.js is missing. Build it first ^(scripts\build_frontend.sh on
    echo Linux/macOS^), then run conda build. pxviewer is noarch — prefer building there.
    exit /b 1
)

set "PKG_FE=%SRC_DIR%\python\pxviewer\frontend"
mkdir "%PKG_FE%\build"
copy /y "%SRC_DIR%\frontend\index.html"     "%PKG_FE%\"        || exit /b 1
copy /y "%SRC_DIR%\frontend\app.html"       "%PKG_FE%\"        || exit /b 1
copy /y "%SRC_DIR%\frontend\favicon.png"    "%PKG_FE%\"        || exit /b 1
copy /y "%SRC_DIR%\frontend\build\index.js" "%PKG_FE%\build\"  || exit /b 1

%PYTHON% -m pip install .\python --no-deps --no-build-isolation -vv
if errorlevel 1 exit /b 1
