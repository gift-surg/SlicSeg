setlocal EnableDelayedExpansion
echo Running Windows build and testscript  %~dp0..\matlab\CIBuildAndTest.m
echo Path: %PATH%
set MATCOMMAND="try, run('%~dp0..\matlab\CIBuildAndTest.m'), catch ex, system(['ECHO Exception during CIBuildAndTest.m: ' ex.message]), exit(1), end, exit(0);"
"C:\Program Files\MATLAB\R2015b\bin\matlab.exe" -wait -nodisplay -nosplash -nodesktop -logfile ci-output.log -r %MATCOMMAND%
set LEVEL=!ERRORLEVEL!
type ci-output.log

if not "!LEVEL!" == "0" (
    echo ERROR: Exit Code = !LEVEL!
	exit /b 1
)
