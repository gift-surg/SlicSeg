function compile_mex()

cwd=miscFns.extractDirPath(mfilename('fullpath'));

mexCmds=cell(0,1);

mexCmds{end+1}=sprintf('mex -O %s+cpp/perform_front_propagation_2d_color.cpp -outdir %s+cpp/',cwd,cwd);
mexCmds{end+1}=sprintf('mex -O %s+gmm/vag_norm_fast.cpp -outdir %s+gmm/',cwd,cwd);

for i=1:length(mexCmds)
  fprintf('Executing %s\n',mexCmds{i});
  eval(mexCmds{i});
end
