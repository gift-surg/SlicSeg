classdef SlicSegAlgorithm < CoreBaseClass
    % SlicSegAlgorithm: implementation of the Slic-Seg interactive segmentation algorithm
    %
    % For a description of Slic-Seg see Wang et al 2006: Slic-Seg: A Minimally Interactive Segmentation
    % of the Placenta from Sparse and Motion-Corrupted Fetal MRI in Multiple Views
    %
    % To run the algorithm:
    %   - create a SlicSegAlgorithm object
    %   - set the volumeImage property to a raw 3D dataset
    %   - set the startIndex property to select a start slice
    %   - set the seedImage property to user-generated scribbles for the start slice
    %   - call StartSliceSegmentation() to segement the initial slice
    %   - set the sliceRange property to the minimum and maximum slice numbers for the propagation
    %   - call SegmentationPropagate() to propagate the start slice segmentation to neighbouring slices in the range set by sliceRange
    %
    %
    % Author: Guotai Wang
    % Copyright (c) 2015-2016 University College London, United Kingdom. All rights reserved.
    % Distributed under the BSD-3 licence. Please see the file licence.txt 
    %
    % This software is not certified for clinical use.
    %
    
    properties (SetObservable)
        volumeImage       % 3D input volume image
        seedImage         % 2D seed image containing user-provided scribbles in the start slice
        
        orientation = 3   % The index of the dimension perpendicular to the seedImage slice
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
        function obj = SlicSegAlgorithm()
            % When these properties are changed, we invalidate the seed image and the segmentation results
            obj.AddPostSetListener(obj, 'volumeImage', @obj.ResetSeedAndSegmentationResultCallback);
            obj.AddPostSetListener(obj, 'orientation', @obj.ResetSeedAndSegmentationResultCallback);

            % When these properties are changed, we invalidate just the segmentation results
            obj.AddPostSetListener(obj, 'seedImage', @obj.ResetSegmentationResultCallback);
            obj.AddPostSetListener(obj, 'startIndex', @obj.ResetSegmentationResultCallback);
            obj.AddPostSetListener(obj, 'sliceRange', @obj.ResetSegmentationResultCallback);
            obj.AddPostSetListener(obj, 'lambda', @obj.ResetSegmentationResultCallback);
            obj.AddPostSetListener(obj, 'sigma', @obj.ResetSegmentationResultCallback);
            obj.AddPostSetListener(obj, 'innerDis', @obj.ResetSegmentationResultCallback);
            obj.AddPostSetListener(obj, 'outerDis', @obj.ResetSegmentationResultCallback);
        end
        
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
            imageSize = obj.volumeImage.getImageSize;
            if((obj.startIndex < 1) || (obj.startIndex > imageSize(obj.orientation)))
                 error('startIndex is not set to a valid value in the range for this image size and orientation');
            end
            seedLabels = obj.GetSeedLabelImage();
            currentSegIndex = obj.startIndex;
            volumeSlice = obj.volumeImage.get2DSlice(currentSegIndex, obj.orientation);
            obj.randomForest.Train(seedLabels, volumeSlice);
            [probabilitySlice, segmentationSlice] = obj.randomForest.PredictUsingConnectivity(seedLabels, volumeSlice, obj.lambda, obj.sigma);
            obj.UpdateResults(currentSegIndex, segmentationSlice, probabilitySlice);
        end
        
        function SegmentationPropagate(obj)
            % Propagates the segmentation obtained from StartSliceSegmentation() to the remaining slices
            
            maxSliceIndex = obj.volumeImage.ImageSize(obj.orientation);
            
            % If no slice range has been specified we use the image limits
            if isempty(obj.sliceRange)
                minSlice = 1;
                maxSlice = maxSliceIndex;
            else
                minSlice = obj.sliceRange(1);
                maxSlice = obj.sliceRange(2);
                if (currentSliceRange(1) < 1) || (currentSliceRange(2) > maxSliceIndex)
                    error('Slice index is out of range for the current image orientation');
                end
            end
            
            currentSegIndex = obj.startIndex;
            
            % First we propagate to the neighbours of the initial slice. We
            % do this first because the algorithm is trained on this initial slice
            if currentSegIndex > minSlice
                obj.TrainAndPropagate(false, currentSegIndex - 1, currentSegIndex);                
            end
            if currentSegIndex < maxSlice
                obj.TrainAndPropagate(false, currentSegIndex + 1, currentSegIndex);                
            end
            
            % Propagate backwards from the backwards neighbour of the initial slice
            priorSegIndex = obj.startIndex - 1;
            for currentSegIndex = obj.startIndex-1 : -1 : minSlice
                obj.TrainAndPropagate(true, currentSegIndex, priorSegIndex);
                priorSegIndex=currentSegIndex;
            end
            
            % Propagate forwards from forwards neighbour of the initial slice
            priorSegIndex = obj.startIndex + 1;
            for currentSegIndex = obj.startIndex+1 : maxSlice
                obj.TrainAndPropagate(true, currentSegIndex, priorSegIndex);
                priorSegIndex=currentSegIndex;
            end
        end

        function ResetSegmentationResult(obj)
            % Deletes the current segmentation results
            fullImageSize = obj.volumeImage.getImageSize;
            obj.segImage = ImageWrapper(zeros(fullImageSize, 'uint8'));
            obj.probabilityImage = ImageWrapper(zeros(fullImageSize));
        end
        
        function ResetSeedPoints(obj)
            % Deletes the current seed points
            sliceSize = obj.volumeImage.get2DSliceSize(obj.orientation);
            obj.seedImage = zeros(sliceSize, 'uint8');
        end
        
        function set.volumeImage(obj, volumeImage)
            % Custom setter method to ensure existing results are invalidated by a change of image
            obj.volumeImage = ImageWrapper(volumeImage);
            obj.ResetSegmentationResult();
            obj.ResetSeedPoints();
        end
    end
    
    methods (Access=private)
        function TrainAndPropagate(obj, train, currentSegIndex, priorSegIndex)
            priorSegmentedSlice = obj.segImage.get2DSlice(priorSegIndex, obj.orientation);
            [currentSeedLabel, currentTrainLabel] = obj.UpdateSeedLabel(priorSegmentedSlice);
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
            notify(obj,'SegmentationProgress', SegmentationProgressEventDataClass(currentSegIndex));
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
        
        function [currentSeedLabel, currentTrainLabel] = UpdateSeedLabel(obj, currentSegImage)
            % generate new training data (for random forest) and new seeds
            % (hard constraint for max-flow) based on segmentation in last slice
            
            fgr = obj.innerDis;
            bgr = obj.outerDis;
            
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
        
        function ResetSeedAndSegmentationResultCallback(obj, ~, ~, ~)
            obj.ResetSeedPoints();
            obj.ResetSegmentationResult();
        end
        
        function ResetSegmentationResultCallback(obj, ~, ~, ~)
            obj.ResetSegmentationResult();
        end
    end
end
