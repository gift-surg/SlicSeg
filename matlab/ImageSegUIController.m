classdef ImageSegUIController < CoreBaseClass
    
    properties (SetObservable)
        currentViewImageIndex
    end
    
    properties (SetAccess = private)
        ILabel
    end
    
    properties (Access = private)
        imageAxes
        slicSeg
        mouseIsDown = false
        foreground = true
    end

    methods
        function obj = ImageSegUIController(currentFigure, imageAxes)
            obj.imageAxes = imageAxes;
            obj.slicSeg = SlicSegAlgorithm();
            set(currentFigure, 'WindowButtonDownFcn', {@obj.mouseDown});
            set(currentFigure, 'WindowButtonMotionFcn', {@obj.mouseMove});
            set(currentFigure, 'WindowButtonUpFcn', {@obj.mouseUp});
            obj.AddEventListener(obj.slicSeg, 'SegmentationProgress', @obj.UpdateSegmentationProgress);
            obj.AddPostSetListener(obj, 'currentViewImageIndex', @obj.sliceNumberChanged);
        end
        
        function maxValue = getMaxSliceNumber(obj)
            imageSize = obj.slicSeg.volumeImage.getImageSize;
            maxValue = max(1, imageSize(3));
        end
        
        function selectForeground(obj)
            obj.foreground = true;
        end
        
        function selectBackground(obj)
            obj.foreground = false;
        end
        
        function resetLabelImage(obj, imgSize)
            obj.ILabel = uint8(zeros([imgSize(1), imgSize(2)]));
        end
        
        function reset(obj)
            obj.resetLabelImage(size(obj.ILabel));
            obj.slicSeg.ResetSegmentationResult();
            obj.slicSeg.ResetSeedPoints();
            obj.showResult();            
        end
        
        function set.currentViewImageIndex(obj, newSliceNumber)
            newSliceNumber = max(1, min(newSliceNumber, obj.getMaxSliceNumber));
            obj.currentViewImageIndex = newSliceNumber;
        end
        
        function selectAndLoad(obj)
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
            obj.slicSeg.startIndex = obj.currentViewImageIndex;
            obj.slicSeg.seedImage = obj.ILabel;
            obj.slicSeg.StartSliceSegmentation();
            obj.showResult();            
        end
        
        function propagate(obj, minSlice, maxSlice)
            obj.slicSeg.sliceRange = [minSlice, maxSlice];
            obj.slicSeg.SegmentationPropagate();
        end
    end
    
    methods (Access = private)
        function mouseDown(obj, imagefig, varargins)
            obj.mouseIsDown = true;
        end
        
        function mouseUp(obj, imagefig, varargins)
            obj.mouseIsDown = false;
        end
        
        function mouseMove(obj, imagefig, varargins)
            if(~obj.mouseIsDown)
                return;
            end
            radius=2;
            temp = get(gca, 'currentpoint');
            x=floor(temp(1,2));
            y=floor(temp(1,1));
            if(obj.foreground)
                hold on;
                plot(temp(1,1),temp(1,2),'.r','MarkerSize',10);
                
                for i=-radius:radius
                    for j=-radius:radius
                        obj.ILabel(x+i,y+j)=127;
                    end
                end
            else
                temp = get(gca,'currentpoint');
                hold on;
                plot(temp(1,1),temp(1,2),'.b','MarkerSize',10);
                for i=-radius:radius
                    for j=-radius:radius
                        obj.ILabel(x+i,y+j)=255;
                    end
                end
            end
        end
        
        function UpdateSegmentationProgress(obj, eventSrc,eventData)
            obj.currentViewImageIndex=eventData.OrgValue;
            obj.showResult();
        end
        
        function sliceNumberChanged(obj, ~, ~, ~)
            obj.showResult();
        end
    
        function showResult(obj)
            I = obj.slicSeg.volumeImage.get2DSlice(obj.currentViewImageIndex, obj.slicSeg.orientation);
            showI = repmat(I,1,1,3);
            
            segI = obj.slicSeg.segImage.get2DSlice(obj.currentViewImageIndex, obj.slicSeg.orientation);
            if(~isempty(find(segI,1)))
                showI=addContourToImage(showI,segI);
            end
            if(obj.currentViewImageIndex==obj.slicSeg.startIndex)
                showI=addSeedsToImage(showI,obj.slicSeg.seedImage);
            end
            axes(obj.imageAxes);
            imshow(showI);
        end

        function ISeg=addContourToImage(obj, IRGB,Label)
            Isize=size(IRGB);
            ISeg=IRGB;
            for i=1:Isize(1)
                for j=1:Isize(2)
                    if(i==1 || i==Isize(1) || j==1 || j==Isize(2))
                        continue;
                    end
                    if(Label(i,j)~=0 && ~(Label(i-1,j)~=0 && Label(i+1,j)~=0 && Label(i,j-1)~=0 && Label(i,j+1)~=0))
                        for di=-1:0
                            for dj=-1:0
                                idi=i+di;
                                jdj=j+dj;
                                if(idi>0 && idi<=Isize(1) && jdj>0 && jdj<=Isize(2))
                                    ISeg(idi,jdj,1)=0;
                                    ISeg(idi,jdj,2)=255;
                                    ISeg(idi,jdj,3)=0;
                                end
                            end
                        end
                    end
                end
            end
        end
        
        function Iout=addSeedsToImage(obj, IRGB,Label)
            Isize=size(IRGB);
            Iout=IRGB;
            for i=1:Isize(1)
                for j=1:Isize(2)
                    if(i==1 || i==Isize(1) || j==1 || j==Isize(2))
                        continue;
                    end
                    if(Label(i,j)==127)
                        Iout(i,j,1)=255;
                        Iout(i,j,2)=0;
                        Iout(i,j,3)=0;
                    elseif(Label(i,j)==255)
                        Iout(i,j,1)=0;
                        Iout(i,j,2)=0;
                        Iout(i,j,3)=255;
                    end
                end
            end
        end
    end
end