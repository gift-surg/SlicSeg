function [E_n,E_w]=convertMatrix_toEdgePairs(W)
% Function to take sparse matrix representations of the
% edge energies and convert it to a form acceptable by the mexGraphCut function

% W -> is a symmetric matrix with the edge weights (diagnoal should be 0 , otherwise problems!)
%      (double)
% IMP: E_n and E_w are nEdges x 2 array (as opposed to the usual 2 x nEdges array)

W=triu(W);
[rowsW,colsW,wts]=find(W);
rowsW=uint32(rowsW);colsW=uint32(colsW);
E_n=[rowsW colsW];
E_w=zeros([length(rowsW) 2],'int32');
E_w(:,1)=int32(wts);
E_w(:,2)=E_w(:,1);

