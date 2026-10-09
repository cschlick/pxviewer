@echo off
:: conda-build invokes bld.bat on Windows; rattler-build (recipe.yaml) looks for
:: build.bat instead. Keep this shim so both build frontends share one script.
call "%~dp0build.bat"
