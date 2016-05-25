#!/bin/bash
/Applications/MATLAB_R2015b.app/bin/matlab -nodisplay -nosplash -nodesktop -r "try, run('$(pwd)/matlab/CIBuildAndTest.m'), catch, exit(1), end, exit(0); "
#mvn install -B
if [ $? -eq 0 ]; then
	echo "Success running CIBuildAndTest.m"
	exit 0;
else
	echo "Failure running CIBuildAndTest.m"
	echo 1;
fi