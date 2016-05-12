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
        volumeImage = ImageWrapper()      % 3D input volume image
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
        randomForest       % Random Forest to learn and predict
    end
    
    methods
        function obj = SlicSegAlgorithm()
            % When these properties are changed, we invalidate the seed image and the segmentation results
            obj.AddPostSetListener(obj, 'volumeImage', @obj.ResetSeedAndSegmentationResultCallback);
            obj.AddPostSetListener(obj, 'orientation', @obj.ResetSeedAndSegmentationResultCallback);

            % When these properties are changed, we invalidate just the segmentation results
            obj.AddPostSetListener(obj, 'seedImage', @obj.ResetSegmentationResultCallback);
            obj.AddPostSetListener(obj, 'startIndex', @obj.ResetSegmentationResultCallback);
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
            obj.Train(seedLabels, volumeSlice);
            [probabilitySlice, segmentationSlice] = obj.PredictUsingConnectivity(seedLabels, volumeSlice, obj.lambda, obj.sigma);
            obj.UpdateResults(currentSegIndex, segmentationSlice, probabilitySlice);
        end
        
        function SegmentationPropagate(obj)
            % Propagates the segmentation obtained from StartSliceSegmentation() to the remaining slices
            
            maxSliceIndex = obj.volumeImage.getMaxSliceNumber(obj.orientation);
            
            % If no slice range has been specified we use the image limits
            if isempty(obj.sliceRange)
                minSlice = 1;
                maxSlice = maxSliceIndex;
            else
                minSlice = obj.sliceRange(1);
                maxSlice = obj.sliceRange(2);
                if (minSlice < 1) || (maxSlice > maxSliceIndex)
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
        
        function Reset(obj)
            % Resets the random forest and results
            obj.randomForest = [];
            obj.volumeImage = ImageWrapper();
            obj.ResetSegmentationResult();
            obj.ResetSegmentationResult();
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
        function Train(obj, currentTrainLabel, volumeSlice)
            % train the random forest using scribbles in on slice
            
            featureMatrix = image2FeatureMatrix(volumeSlice);
            if(isempty(currentTrainLabel) || isempty(find(currentTrainLabel>0)))
                error('the training set is empty');
            end
            foreground=find(currentTrainLabel==127);
            background=find(currentTrainLabel==255);
            totalseeds=length(foreground)+length(background);
            if(totalseeds==0)
                error('the training set is empty');
            end
            TrainingSet=zeros(totalseeds,size(featureMatrix,2));
            TrainingLabel=zeros(totalseeds,1);
            TrainingSet(1:length(foreground),:)=featureMatrix(foreground,:);
            TrainingLabel(1:length(foreground))=1;
            TrainingSet(length(foreground)+1:length(foreground)+length(background),:)=featureMatrix(background,:);
            TrainingLabel(length(foreground)+1:length(foreground)+length(background))=0;
            TrainingDataWithLabel=[TrainingSet,TrainingLabel];
            obj.getRandomForest.Train(TrainingDataWithLabel');
        end
        
        function [probabilitySlice, segmentationSlice] = PredictUsingPrior(obj, currentSeedLabel, volumeSlice, priorSegSlice, lambda, sigma)
            % get the probability in one slice
            P0 = obj.Predict(volumeSlice);
            probabilitySlice=SlicSegAlgorithm.ProbabilityProcessUsingShapePrior(P0,priorSegSlice);
            segmentationSlice = SlicSegAlgorithm.GetSingleSliceSegmentation(currentSeedLabel, volumeSlice, probabilitySlice, lambda, sigma);
        end
        
        function [probabilitySlice, segmentationSlice] = PredictUsingConnectivity(obj, currentSeedLabel, volumeSlice, lambda, sigma)
            % get the probability in one slice
            P0 = obj.Predict(volumeSlice);
            probabilitySlice = SlicSegAlgorithm.ProbabilityProcessUsingConnectivity(currentSeedLabel, P0, volumeSlice);
            segmentationSlice = SlicSegAlgorithm.GetSingleSliceSegmentation(currentSeedLabel, volumeSlice, probabilitySlice, lambda, sigma);            
        end
        
        function randomForest = getRandomForest(obj)
            if isempty(obj.randomForest)
                obj.randomForest = ForestWrapper();
                obj.randomForest.Init(20,8,20);        
            end
            randomForest = obj.randomForest;
        end
        
        function TrainAndPropagate(obj, train, currentSegIndex, priorSegIndex)
            priorSegmentedSlice = obj.segImage.get2DSlice(priorSegIndex, obj.orientation);
            [currentSeedLabel, currentTrainLabel] = obj.UpdateSeedLabel(priorSegmentedSlice);
            if(train)
                priorVolumeSlice = obj.volumeImage.get2DSlice(priorSegIndex, obj.orientation);
                obj.Train(currentTrainLabel, priorVolumeSlice);
            end
            currentVolumeSlice = obj.volumeImage.get2DSlice(currentSegIndex, obj.orientation);
            [probabilitySlice, segmentationSlice] = obj.PredictUsingPrior(currentSeedLabel, currentVolumeSlice, priorSegmentedSlice, obj.lambda, obj.sigma);
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
        
        function P0 = Predict(obj, volumeSlice)
            featureMatrix = image2FeatureMatrix(volumeSlice);
            Prob = obj.getRandomForest.Predict(featureMatrix');
            P0 = reshape(Prob, size(volumeSlice,1), size(volumeSlice,2));
        end        
    end
    
    methods (Static, Access = private) 
        function P = ProbabilityProcessUsingShapePrior(P0,lastSeg)
            Isize=size(lastSeg);
            dis=zeros(Isize);
            se= strel('disk',1);
            temp0=lastSeg;
            temp1=imerode(temp0,se);
            currentdis=0;
            while(~isempty(find(temp1>0)))
                dis0=temp0-temp1;
                currentdis=currentdis+1;
                dis(dis0>0)=currentdis;
                temp0=temp1;
                temp1=imerode(temp0,se);
            end
            maxdis=currentdis;
            
            P=P0;
            outsideIndex=intersect(find(dis==0),find(P>0.5));
            P(outsideIndex)=0.4*P(outsideIndex);
            insideIndex=intersect(find(dis>0) , find(P<0.8));
            P(insideIndex)=P(insideIndex)+0.2*dis(insideIndex)/maxdis;
        end
        
        function P = ProbabilityProcessUsingConnectivity(currentSeedLabel,P0,I)
            PL=P0>=0.5;
            pSe= strel('disk',3);
            pMask=imclose(PL,pSe);
            [H,W]=size(P0);
            HW=H*W;
            indexHW=uint32(zeros(HW,1));
            seedsIndex=find(currentSeedLabel==127);
            seeds=length(seedsIndex);
            indexHW(1:seeds)=seedsIndex(1:seeds);
            L=uint8(zeros(H,W));
            P=P0;
            L(seedsIndex)=1;
            P(seedsIndex)=1.0;
            
            
            fg=I(seedsIndex);
            fg_mean=mean(fg);
            fg_std=sqrt(var(double(fg)));
            fg_min=fg_mean-fg_std*3;
            fg_max=fg_mean+fg_std*2;
            
            current=1;
            while(current<=seeds)
                currentIndex=indexHW(current);
                NeighbourIndex=[currentIndex-1,currentIndex+1,...
                    currentIndex+H,currentIndex+H-1,currentIndex+H+1,...
                    currentIndex-H,currentIndex-H-1,currentIndex-H+1];
                for i=1:8
                    tempIndex=NeighbourIndex(i);
                    if(tempIndex>0 && tempIndex<HW && L(tempIndex)==0 && pMask(tempIndex)>0 && I(tempIndex)>fg_min && I(tempIndex)<fg_max)
                        L(tempIndex)=1;
                        seeds=seeds+1;
                        indexHW(seeds,1)=tempIndex;
                    end
                end
                current=current+1;
            end
            
            Lindex=find(L==0);
            P(Lindex)=P(Lindex)*0.4;
        end
        
        function seg = GetSingleSliceSegmentation(currentSeedLabel, currentI, currentP, lambda, sigma)
            % use max flow to get the segmentation in one slice
            
            currentSeed = currentSeedLabel;
            [flow, currentSegLabel] = wgtmaxflowmex(currentI, currentSeed, currentP, lambda, sigma);
            currentSegLabel = 1-currentSegLabel;
            se = strel('disk', 2);
            currentSegLabel = imclose(currentSegLabel, se);
            currentSegLabel = imopen(currentSegLabel, se);
            seg = currentSegLabel(:,:);
        end 
    end    
end
