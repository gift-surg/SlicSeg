#!/bin/bash
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 /usr/local/R2015b/bin/matlab -nodisplay -nosplash -nodesktop -r "try, run('$(pwd)/matlab/CIBuildAndTest.m'), catch ex, disp(['Exception during CIBuildAndTest.m: ' ex.message]), exit(1), end, exit(0);"
#mvn install -B
if [ $? -eq 0 ]; then
	echo "Success running CIBuildAndTest.m"
	exit 0;
else
	echo "Failure running CIBuildAndTest.m"
	exit 1;
fi