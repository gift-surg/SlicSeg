function [E_n,E_w]=convertMatrix_toEdgePairs(W,Wstar)
% Function to take sparse matrix representations of the
% edge energies and convert it to a form acceptable by the mexGraphCut function

% W -> is a symmetric matrix with the edge weights (diagnoal should be 0 , otherwise problems!)
%      (double)
% IMP: E_n and E_w are nEdges x 2 array (as opposed to the usual 2 x nEdges array)

maskEdges=(W|Wstar);

% symettrize maskEdges
maskEdges=maskEdges|maskEdges';
maskEdges=triu(maskEdges);
[rows,cols]=find(maskEdges);

rows=uint32(rows);
cols=uint32(cols);
E_n=[rows cols];

W=triu(W);
[rowsW,colsW,wts]=find(W);
rowsW=uint32(rowsW);colsW=uint32(colsW);
mapToEdges=gsc.cpp.mex_mapEdges(rows,cols,rowsW,colsW);
E_w=zeros([length(rows) 2],'int32');
E_w(mapToEdges,1)=int32(wts);
E_w(:,2)=E_w(:,1);

E_starShape=logical(zeros([length(rows) 2],'int32'));

[rowsW,colsW,wts]=find(triu(Wstar));
rowsW=uint32(rowsW);colsW=uint32(colsW);
mapToEdges=gsc.cpp.mex_mapEdges(rows,cols,rowsW,colsW);
E_starShape(mapToEdges,1)=wts;

Wstar=Wstar';
[rowsW,colsW,wts]=find(triu(Wstar));
rowsW=uint32(rowsW);colsW=uint32(colsW);
mapToEdges=gsc.cpp.mex_mapEdges(rows,cols,rowsW,colsW);
E_starShape(mapToEdges,2)=wts;

E_w(E_starShape(:,1),1)=intmax-E_w(E_starShape(:,1),2)-1; % Subtracting to prevent overflow in graphCuts
E_w(E_starShape(:,2),2)=intmax-E_w(E_starShape(:,2),1)-1;
% sometimes both edges can be infinity, so correct that!

bothInfinity=E_starShape(:,1)&E_starShape(:,2);
E_w(bothInfinity,1)=intmax/2-1;
E_w(bothInfinity,2)=intmax/2-1;
