setlocal EnableDelayedExpansion
echo Running Windows build and testscript  %~dp0..\matlab\CIBuildAndTest.m
echo Path: %PATH%
set MATCOMMAND="try, run('%~dp0..\matlab\CIBuildAndTest.m'), catch ex, system(['ECHO Exception during CIBuildAndTest.m: ' ex.message]), exit(1), end, exit(0);"
"C:\Program Files\MATLAB\R2015b\bin\matlab.exe" -wait -nodisplay -nosplash -nodesktop -logfile ci-output.log -r %MATCOMMAND%
set LEVEL=!ERRORLEVEL!
if exist ci-output.log (
    type ci-output.log
) else (
    echo Log file not found
)


if not "!LEVEL!" == "0" (
    echo ERROR: Exit Code = !LEVEL!
	exit /b 1
)
