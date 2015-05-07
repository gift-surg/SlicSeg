% Copyright (C) 2009-10 Varun Gulshan
function segmenterH=segOptions(data)
% Function to make the opts structure for segmentation
%
import miscFns.*

% ------ Overwrite with options which were given from the ui ---------

switch(data.segMethod)
  case 'RW'
    segOpts=rw.segOpts();
    segmenterH=rw.segEngine(data.debugLevel,segOpts);
  case 'SP-SIG'
    segOpts=sp.segOpts();
    segOpts.spImg='imgSmoothed';
    segOpts.sigma=3;
    segmenterH=sp.segEngine(data.debugLevel,segOpts);
  case 'SP-IG'
    segOpts=sp.segOpts();
    segOpts.spImg='imgSmoothed';
    segOpts.sigma=0;
    segmenterH=sp.segEngine(data.debugLevel,segOpts);
  case 'SP-LIG'
    segOpts=sp.segOpts();
    segmenterH=sp.segEngine(data.debugLevel,segOpts);
  case 'ESCseq'
    segOpts=gscSeq.segOpts();
    segOpts.gcGamma=data.gamma;
    segOpts.geoGamma=0;
    segmenterH=gscSeq.segEngine(data.debugLevel,segOpts);
  case 'ESC'
    segOpts=gsc.segOpts();
    segOpts.gcGamma=data.gamma;
    segOpts.geoGamma=0;
    segmenterH=gsc.segEngine(data.debugLevel,segOpts);
  case 'GSCseq'
    segOpts=gscSeq.segOpts();
    segOpts.gcGamma=data.gamma;
    segOpts.geoGamma=data.geoGamma;
    segmenterH=gscSeq.segEngine(data.debugLevel,segOpts);
   case 'GSC'
    segOpts=gsc.segOpts();
    segOpts.gcGamma=data.gamma;
    segOpts.geoGamma=data.geoGamma;
    segmenterH=gsc.segEngine(data.debugLevel,segOpts);
  case 'BJ'
    segOpts=bj.segOpts();
    segOpts.gcGamma=data.gamma;
    segmenterH=bj.segEngine(data.debugLevel,segOpts);
  case 'PP'
    segOpts=bj.segOpts();
    segOpts.gcGamma=data.gamma;
    segOpts.postProcess=1;
    segmenterH=bj.segEngine(data.debugLevel,segOpts);
 otherwise
   error('Invalid method: %s\n',data.segMethod);

end
