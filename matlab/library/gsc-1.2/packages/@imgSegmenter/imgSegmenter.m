% Copyright (C) 2009-10 Varun Gulshan
% This abstract class defines the interface for an image segmentation method
% It inherits from the handle class, so that means you can maintain state inside
% an object of this class
classdef imgSegmenter < handle
  properties (Abstract, SetAccess=private, GetAccess=public)
    debugLevel
    seg % uint8 segmentation mask (255=fg,0=bg)
  end

  methods(Abstract)
    preProcess(obj,img) % Call this function first to preprocess image if needed
                        % img should be double
    startOk=start(obj,labelImg) % Call this function after pre-processing to get initial seg
                                % labelImg denotes user strokes, labelImg=0 is empty
                                % labelImg=1 is FG
                                % labelImg=2 is BG
    updateSeg(obj,labelImg) % Call this function once user makes edits
  end
end
