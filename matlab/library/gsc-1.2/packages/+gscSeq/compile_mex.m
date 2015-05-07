function compile_mex()

cwd=miscFns.extractDirPath(mfilename('fullpath'));

mexCmds=cell(0,1);

mexCmds{end+1}=sprintf('mex -O %s+cpp/mex_setupTransductionGraph.cpp -outdir %s+cpp/',cwd,cwd);
mexCmds{end+1}=sprintf('mex -O %s+cpp/mex_mapEdges.cpp -outdir %s+cpp/',cwd,cwd);
mexCmds{end+1}=sprintf('mex -O -I%s+cpp/graphCut/ %s+cpp/graphCut/mexDGC.cpp %s+cpp/graphCut/graph.cpp -outdir %s+cpp/',cwd,cwd,cwd,cwd);
mexCmds{end+1}=sprintf('mex -O %s+gmm/vag_norm_fast.cpp -outdir %s+gmm/',cwd,cwd);
mexCmds{end+1}=sprintf('mex -O %s+cpp/sp/mex_spNormalized_constrained.cpp %s+cpp/sp/heaps/f_heap.cc -outdir %s+cpp/',cwd,cwd,cwd);

for i=1:length(mexCmds)
  fprintf('Executing %s\n',mexCmds{i});
  eval(mexCmds{i});
end
