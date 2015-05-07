% Copyright (C) 2009-10 Varun Gulshan

% This class defines the data fields stored in the user interface
classdef uidata
  properties (SetAccess = public, GetAccess = public)
    baseLineCanvasPosition % original position of canvas
    uiState % string denoting current state of UI
    drag % variable to denote if mouse is dragging
    strokeType
    canvasView % string to denote current view setting of canvas, it is one of:
    % 'drawing_segBoundary', 'gt', 'fg', 'bg', 'posterior'
    brushRad
    brushMask
    brushType
    %boxStarted
    segBoundaryColors
    segBoundaryMask
    segBoundaryColor_inside
    segBoundaryColor_outside
    segBoundaryWidth
    gamma
    geoGamma
    origCanvasPosition
    segMethod
    amap
    cmap
    imgReScale % scaling factor from orignal size to canvas
    labelImg % resolution = canvas
    labelImgHandle
    inImg  % resolution = original image, type=double
    curImg % resolution = canvas
    curSeg % resolution = canvas
    labelImg_orig % resolution = original img
    defSaveLocation
    defSaveLocation_seg
    hCanvas
    wCanvas
    segmenterH % handle to a image segmentation class
    inFileName 
    pathname 
    debugLevel
    labelMask % CHECK
    labelBoundaryColors % CHECK
    cwd
  end

  methods
    function obj=uidata()
      % Not initializing anything here, everything gets set to [] as default
    end
  end
end
