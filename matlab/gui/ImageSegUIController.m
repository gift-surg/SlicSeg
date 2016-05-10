classdef ImageSegUIController < CoreBaseClass
    % ImageSegUIController A controller class for the ImageSegUI user interface
    % which handles axes drawing and mouse input
    %
    %
    % Author: Guotai Wang
    % Copyright (c) 2015-2016 University College London, United Kingdom. All rights reserved.
    % Distributed under the BSD-3 licence. Please see the file licence.txt 
    %
    % This software is not certified for clinical use.
    %
    
    properties
        leftMouseSelectsForeground = true
    end
    
    properties (SetObservable)
        currentViewImageIndex % The currently displayed slice number
    end
    
    properties (Access = private)
        labelImage
        imageAxes
        slicSeg
        mouseIsDown = false
    end

    methods
        function obj = ImageSegUIController(currentFigure, imageAxes)
            % Creates a new ImageSegUIController for the given figure and axes
            obj.imageAxes = imageAxes;
            obj.slicSeg = SlicSegAlgorithm();
            
            % Listen for mouse events
            set(currentFigure, 'WindowButtonDownFcn', @obj.mouseDown);
            set(currentFigure, 'WindowButtonMotionFcn', {@obj.mouseMove});
            set(currentFigure, 'WindowButtonUpFcn', {@obj.mouseUp});
            
            % Listen for changes in the segmented slice and the currently
            % viewed slice
            obj.AddEventListener(obj.slicSeg, 'SegmentationProgress', @obj.UpdateSegmentationProgress);
            obj.AddPostSetListener(obj, 'currentViewImageIndex', @obj.sliceNumberChanged);
        end
        
        function maxValue = getMaxSliceNumber(obj)
            % Returns the number of slices in the currently loaded image
            imageSize = obj.slicSeg.volumeImage.getImageSize;
            maxValue = max(1, imageSize(3));
        end
        
        function selectAndLoad(obj)
            % Opens a dialog allowing the user to select a directory from
            % which images will be loaded
            reset(obj.imageAxes);
            cla(obj.imageAxes);
            [~, imgFolderName, ~] = uigetfile('*.png','select a file');
            obj.slicSeg.volumeImage = OpenPNGImage(imgFolderName);
            maxSliceNumber = obj.getMaxSliceNumber;
            currentSliceNumber = max(1, min(maxSliceNumber, round(maxSliceNumber/2)));
            imgSize = obj.slicSeg.volumeImage.getImageSize;
            obj.resetLabelImage(imgSize);
            obj.currentViewImageIndex = currentSliceNumber; % Note this will trigger a redraw
        end
            
        function segment(obj)
            % Segment the current slice
            obj.slicSeg.startIndex = obj.currentViewImageIndex;
            obj.slicSeg.seedImage = obj.labelImage;
            obj.slicSeg.StartSliceSegmentation();
            obj.showResult();            
        end
        
        function propagate(obj, minSlice, maxSlice)
            % Propagate segmentation to neighbouring slices
            obj.slicSeg.sliceRange = [minSlice, maxSlice];
            obj.slicSeg.SegmentationPropagate();
        end
        
        function reset(obj)
            % Delete the current seed points and segmentations
            obj.resetLabelImage(size(obj.labelImage));
            obj.slicSeg.ResetSegmentationResult();
            obj.slicSeg.ResetSeedPoints();
            obj.showResult();            
        end
        
        function set.currentViewImageIndex(obj, newSliceNumber)
            newSliceNumber = max(1, min(newSliceNumber, obj.getMaxSliceNumber));
            obj.currentViewImageIndex = newSliceNumber;
        end
    end
    
    methods (Access = private)
        function mouseDown(obj, ~, ~)
            obj.mouseIsDown = true;
        end
        
        function mouseUp(obj, ~, ~)
            obj.mouseIsDown = false;
        end
        
        function mouseMove(obj, src, ~)
            if(~obj.mouseIsDown)
                return;
            end

            select_foreground = obj.leftMouseSelectsForeground;
            selection_type = get(src, 'SelectionType');
            if strcmp(selection_type, 'alt')
                select_foreground = ~select_foreground;
            end

            coords = get(gca, 'currentpoint');
            x = floor(coords(1,2));
            y = floor(coords(1,1));
            
            if select_foreground
                labelIndex = 127;
                marker = '.r';
            else
                labelIndex = 255;
                marker = '.b';
            end
            
            hold on;
            plot(coords(1,1), coords(1,2), marker, 'MarkerSize', 10);

            radius=2;
            imin = max(1, x-radius);
            imax = min(size(obj.labelImage,1), x+radius);
            jmin = max(1, y-radius);
            jmax = min(size(obj.labelImage,2), y+radius);
            for i = imin : imax
                for j = jmin : jmax
                    obj.labelImage(i, j) = labelIndex;
                end
            end
        end
        
        function UpdateSegmentationProgress(obj, ~, eventData)
            % When a slice is segmented we change the current slice number,
            % which will force a redraw
            obj.currentViewImageIndex = eventData.OrgValue;
        end
        
        function sliceNumberChanged(obj, ~, ~, ~)
            obj.showResult();
        end
    
        function showResult(obj)
            I = obj.slicSeg.volumeImage.get2DSlice(obj.currentViewImageIndex, obj.slicSeg.orientation);
            showI = repmat(I,1,1,3);
            
            segI = obj.slicSeg.segImage.get2DSlice(obj.currentViewImageIndex, obj.slicSeg.orientation);
            if(~isempty(find(segI,1)))
                showI=obj.addContourToImage(showI,segI);
            end
            if(obj.currentViewImageIndex==obj.slicSeg.startIndex)
                showI=obj.addSeedsToImage(showI,obj.slicSeg.seedImage);
            end
            axes(obj.imageAxes);
            imshow(showI);
        end
        
        function resetLabelImage(obj, imgSize)
            obj.labelImage = uint8(zeros([imgSize(1), imgSize(2)]));
        end
        
        function iSeg = addContourToImage(obj, iRGB, label)
            isize=size(iRGB);
            iSeg=iRGB;
            for i=1:isize(1)
                for j=1:isize(2)
                    if(i==1 || i==isize(1) || j==1 || j==isize(2))
                        continue;
                    end
                    if(label(i,j)~=0 && ~(label(i-1,j)~=0 && label(i+1,j)~=0 && label(i,j-1)~=0 && label(i,j+1)~=0))
                        for di=-1:0
                            for dj=-1:0
                                idi=i+di;
                                jdj=j+dj;
                                if(idi>0 && idi<=isize(1) && jdj>0 && jdj<=isize(2))
                                    iSeg(idi,jdj,1)=0;
                                    iSeg(idi,jdj,2)=255;
                                    iSeg(idi,jdj,3)=0;
                                end
                            end
                        end
                    end
                end
            end
        end
        
        function iout = addSeedsToImage(obj, iRGB, label)
            isize = size(iRGB);
            iout = iRGB;
            for i=1:isize(1)
                for j=1:isize(2)
                    if(i==1 || i==isize(1) || j==1 || j==isize(2))
                        continue;
                    end
                    if(label(i,j)==127)
                        iout(i,j,1)=255;
                        iout(i,j,2)=0;
                        iout(i,j,3)=0;
                    elseif(label(i,j)==255)
                        iout(i,j,1)=0;
                        iout(i,j,2)=0;
                        iout(i,j,3)=255;
                    end
                end
            end
        end
    end
end