% Copyright (C) 2009-10 Varun Gulshan
% This class implements GSC segmentation
% It inherits the abstract imgSegmenter class and implements its functions
classdef segEngine < imgSegmenter
  properties (SetAccess=private, GetAccess=public)
    state % string = 'init','pped','started'
    opts % object of type bj.segOpts
    img % double img, b/w [0,1]
    features
    W % sparse matrix of edge weights
    posteriorImage % data terms for graph cut
    debugLevel % inherited property
    seg % inheritied property
    starInfo % debug information of star shape
  end

  methods
    function obj=segEngine(debugLevel,segOpts)
      obj.debugLevel=debugLevel;
      obj.opts=segOpts;
      obj.state='init';
      obj.seg=[];
      obj.img=[];
    end
    postImg=viewPosterior(obj)
  end
end
