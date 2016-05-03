classdef SlicSegAlgorithm < handle
    % Interactive segmentation algorithm of Slic-Seg
    % The user selects one start slice and draws some scribbles in that
    % slice to start segmentation.
    properties
        volumeImage       % 3D input volume image
        seedImage         % 2D seed image containing user-provided scribbles in the start slice
        
        orientation       % The index of the dimension perpendicular to the seedImage slice
        startIndex        % start slice index
        sliceRange        % 2x1 matrix to store the minimum and maximum slice index. Leave empty to use first and last slices
        
        lambda   = 5.0    % parameter for max-flow; controls the weight of unary term and binary term
        sigma    = 3.5    % parameter for max-flow; controls the sensitivity of intensity difference
        innerDis = 5      % radius of erosion when generating new training data
        outerDis = 6      % radius of dilation when generating new training data  
    end
    
    properties (SetAccess = private)
        segImage          % 3D image for segmentation result
        probabilityImage  % 3D image of probability of being foreground
    end
    
    events
        SegmentationProgress % Event fired after each image slice has been segmented
    end
    
    properties (Access = private)
        randomForest = RandomForestWrapper()  % Random Forest to learn and predict
    end
    
    methods
        function RunSegmention(obj)
            % Runs the full segmentation. The seed image and start index must be set before calling this method.

            obj.StartSliceSegmentation();
            obj.SegmentationPropagate();
        end
        
        function StartSliceSegmentation(obj)
            % Creates a segmentation for the image slice specified in
            % startIndex. The seed image and start index must be set before calling this method.
            
            if(isempty(obj.startIndex) || isempty(obj.seedImage))
                error('startIndex and seedImage must be set before calling StartSliceSegmentation()');
            end
            if((obj.startIndex < 1) || (obj.startIndex > obj.volumeImage.ImageSize(obj.orientation)))
                 error('startIndex is not set to a valid value in the range for this image size and orientation');
            end
            seedLabels = obj.GetSeedLabelImage();
            currentSegIndex = obj.startIndex;
            volumeSlice = obj.volumeImage.Get2DSlice(currentSegIndex, obj.orientation);
            obj.randomForest.Train(seedLabels, volumeSlice);
            [probabilitySlice, segmentationSlice] = obj.randomForest.PredictUsingConnectivity(seedLabels, volumeSlice, obj.lambda, obj.sigma);
            obj.UpdateResults(currentSegIndex, segmentationSlice, probabilitySlice);
        end
        
        function SegmentationPropagate(obj)
            % Propagates the segmentation obtained from StartSliceSegmentation() to the remaining slices
            
            maxSliceIndex = obj.volumeImage.ImageSize(obj.orientation);
            
            % If no slice range has been specified we use the image limits
            if isempty(obj.sliceRange)
                currentSliceRange = [1, maxSliceIndex];
            else
                currentSliceRange = obj.sliceRange;
                if (currentSliceRange(1) < 1) || (currentSliceRange(2) > maxSliceIndex)
                    error('Slice index is out of range for the current image orientation');
                end
            end
            
            currentSegIndex=obj.startIndex;
            for i=1:obj.startIndex-currentSliceRange(1)
                priorSegIndex=currentSegIndex;
                currentSegIndex=currentSegIndex-1;
                obj.TrainAndPropagate(i>1,currentSegIndex,priorSegIndex);
            end
            
            % propagate to following slices
            currentSegIndex=obj.startIndex;
            notify(obj,'SegmentationProgress',SegmentationProgressEventDataClass(currentSegIndex));
            for i=obj.startIndex:currentSliceRange(2)-1
                priorSegIndex=currentSegIndex;
                currentSegIndex=currentSegIndex+1;
                obj.TrainAndPropagate(i>obj.startIndex,currentSegIndex,priorSegIndex);
            end
        end

        function set.volumeImage(obj, volumeImage)
            obj.volumeImage = ImageWrapper(volumeImage);
            obj.ResetSegmentationResult();
        end

        function OpenScribbleImage(obj,labelFileName)
            % read scribbles in the start slice (*.png rgb file)
            rgbLabel=imread(labelFileName);
            ISize=size(rgbLabel);
            ILabel=uint8(zeros(ISize(1),ISize(2)));
            for i=1:ISize(1)
                for j=1:ISize(2)
                    if(rgbLabel(i,j,1)==255 && rgbLabel(i,j,2)==0 && rgbLabel(i,j,3)==0)
                        ILabel(i,j)=127;
                    elseif(rgbLabel(i,j,1)==0 && rgbLabel(i,j,2)==0 && rgbLabel(i,j,3)==255)
                        ILabel(i,j)=255;
                    end
                end
            end
            obj.seedImage = ILabel;
            disp('seed image has been loaded successfully');
        end

        function ResetSegmentationResult(obj)
            % Deletes the current segmentation results
            fullImageSize = obj.volumeImage.getImageSize;
            obj.segImage = ImageWrapper(zeros(fullImageSize, 'uint8'));
            obj.probabilityImage = ImageWrapper(zeros(fullImageSize));
        end
        
        function ResetSeedPoints(obj)
            % Deletes the current seed points
            sliceSize = obj.volumeImage.get2DSliceSlize(obj.orientation);
            obj.seedImage = zeros(sliceSize, 'uint8');
        end        
    end
    
    methods (Access=private)
        function TrainAndPropagate(obj, train, currentSegIndex, priorSegIndex)
            priorSegmentedSlice = obj.segImage.get2DSlice(priorSegIndex, obj.orientation);
            [currentSeedLabel, currentTrainLabel] = SlicSegAlgorithm.UpdateSeedLabel(priorSegmentedSlice, obj.innerDis, obj.outerDis);
            if(train)
                priorVolumeSlice = obj.volumeImage.Get2DSlice(priorSegIndex, obj.orientation);
                obj.randomForest.Train(currentTrainLabel, priorVolumeSlice);
            end
            currentVolumeSlice = obj.volumeImage.Get2DSlice(currentSegIndex, obj.orientation);
            [probabilitySlice, segmentationSlice] = obj.randomForest.PredictUsingPrior(currentSeedLabel, currentVolumeSlice, priorSegmentedSlice, obj.lambda, obj.sigma);
            obj.UpdateResults(currentSegIndex, segmentationSlice, probabilitySlice);
        end
        
        function UpdateResults(obj, currentSegIndex, segmentationSlice, probabilitySlice)
            obj.segImage.replaceImageSlice(segmentationSlice, currentSegIndex, obj.orientation);
            obj.probabilityImage.replaceImageSlice(probabilitySlice, currentSegIndex, obj.orientation);
            notify(obj,'SegmentationProgress',SegmentationProgressEventDataClass(currentSegIndex));
        end
        
        function label = GetSeedLabelImage(obj)
            label = obj.seedImage;
            [H,W] = size(label);
            for i = 5:5:H-5
                label(i,5)=255;
                label(i,W-5)=255;
            end
            for j = 5:5:W-5
                label(5,j)=255;
                label(H-5,j)=255;
            end
        end
    end
    
    methods (Static, Access=private)
        function [currentSeedLabel, currentTrainLabel] = UpdateSeedLabel(currentSegImage, fgr, bgr)
            % generate new training data (for random forest) and new seeds
            % (hard constraint for max-flow) based on segmentation in last slice
            
            tempSegLabel=currentSegImage;
            fgSe1=strel('disk',fgr);
            fgMask=imerode(tempSegLabel,fgSe1);
            if(length(find(fgMask>0))<100)
                fgMask=bwmorph(tempSegLabel,'skel',Inf);
            else
                fgMask=bwmorph(fgMask,'skel',Inf);
            end
            bgSe1=strel('disk',bgr);
            bgSe2=strel('disk',bgr+1);
            fgDilate1=imdilate(tempSegLabel,bgSe1);
            fgDilate2=imdilate(tempSegLabel,bgSe2);
            bgMask=fgDilate2-fgDilate1;
            currentTrainLabel=uint8(zeros(size(tempSegLabel)));
            currentTrainLabel(fgMask>0)=127;
            currentTrainLabel(bgMask>0)=255;
            
            bgMask=1-fgDilate1;
            currentSeedLabel=uint8(zeros(size(tempSegLabel)));
            currentSeedLabel(fgMask>0)=127;
            currentSeedLabel(bgMask>0)=255;
        end
    end
end
