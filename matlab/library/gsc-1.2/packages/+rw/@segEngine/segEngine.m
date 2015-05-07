% Copyright (C) 2009-10 Varun Gulshan
% This class implements RW segmentation
% It inherits the abstract imgSegmenter class and implements its functions
classdef segEngine < imgSegmenter
  properties (SetAccess=private, GetAccess=public)
    state % string = 'init','pped','started'
    opts % object of type rw.segOpts
    img % double img, b/w [0,1]
    debugLevel % inherited property
    seg % inheritied property
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
