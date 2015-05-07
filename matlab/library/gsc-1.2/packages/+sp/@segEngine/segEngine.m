% Copyright (C) 2009-10 Varun Gulshan
% This class implements SP segmentation
% It inherits the abstract imgSegmenter class and implements its functions
classdef segEngine < imgSegmenter
  properties (SetAccess=private, GetAccess=public)
    state % string = 'init','pped','started'
    opts % object of type sp.segOpts
    img % double img, b/w [0,1]
    features
    debugLevel % inherited property
    seg % inheritied property
    posteriorImage % in case likelihoods are computed
    smoothedImg % in case shortest paths are computed on smoothed gradients
  end

  methods
    function obj=segEngine(debugLevel,segOpts)
      obj.debugLevel=debugLevel;
      obj.opts=segOpts;
      obj.state='init';
      obj.seg=[];
      obj.img=[];
    end
  end
end
