@echo off

rem ##########################
rem ##    Your settings     ##
rem ##########################

rem replace the wallet addresses with your own
set "WALLET=nu3Numen1pYQqAfigZh4LiK2VF2b1cRBggkKB4gQcLFpBDhGp"
set "WORKER=rig1"
set "POOL=stratum+ssl://stratum.nitopool.online:3027"
set "OPTIONS="

rem ##########################
rem ## End of your settings ##
rem ##########################

cd /d "%~dp0"
numen-miner.exe -o %POOL% -u %WALLET%.%WORKER% %OPTIONS% %*
pause
