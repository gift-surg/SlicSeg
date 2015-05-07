function setup()
curDir=pwd;
path([curDir],path);
path([curDir '/ui/'],path);
path([curDir '/packages/'],path);

rwDir1=[curDir '/../../software/graphAnalysisToolbox-1.0/'];
rwDir2=[curDir '/../../software/random_walker_matlab_code/'];
if(exist(rwDir1,'dir') & exist(rwDir2,'dir')),
  path(rwDir1,path);
  path(rwDir2,path);
else
  fprintf('Warning: Cant find random walker code, RW will not work\n');
end
