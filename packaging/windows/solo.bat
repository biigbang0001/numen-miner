@echo off

rem ##########################
rem ##    Your settings     ##
rem ##########################

set "NODE=ws://127.0.0.1:9944"
set "OPTIONS="

rem ##########################
rem ## End of your settings ##
rem ##########################

cd /d "%~dp0"
numen-miner.exe --url %NODE% %OPTIONS% %*
pause
