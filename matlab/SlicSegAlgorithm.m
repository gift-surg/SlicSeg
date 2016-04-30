classdef SlicSegAlgorithm < handle
    % Interactive segmentation algorithm of Slic-Seg
    % The user selects one start slice and draws some scribbles in that
    % slice to start segmentation.
    properties
        startIndex       % start slice index
        sliceRange       % 2x1 matrix to store the minimum and maximum slice index
        
        seedImage         % 2D seed image containing user-provided scribbles in the start slice
        volumeImage       % 3D input volume image
        segImage          % 3D image for segmentation result
        probabilityImage  % 3D image of probability of being foreground
        
        lambda            % parameter for max-flow; controls the weight of unary term and binary term
        sigma             % parameter for max-flow; controls the sensitivity of intensity difference
        innerDis          % radius of erosion when generating new training data
        outerDis          % radius of dilation when generating new training data
    end
    
    properties (Access = private)
        randomForest      % Random Forest to learn and predict
    end
    
    properties (Dependent)
        imageSize        % 3x1 matrix, size of image (width, height, slices)
    end
    
    events
        SegmentationProgress
    end
    
    methods
        function obj = SlicSegAlgorithm
            obj.startIndex=0;
            obj.sliceRange=[0,0];
            obj.randomForest=RandomForestWrapper;
            
            obj.lambda=5.0;
            obj.sigma=3.5;
            obj.innerDis=5;
            obj.outerDis=6;
        end
        
        function imageSize = get.imageSize(obj)
            if isempty(obj.volumeImage)
                imageSize=[0,0,0];
            else
                imageSize=size(obj.volumeImage);
            end
        end
        
        function set.volumeImage(obj,volumeImage)
            obj.volumeImage=volumeImage;
            obj.ResetSegmentationResult();
        end
        
        function SetMultipleProperties(obj,varargin)
            argin=varargin;
            while(length(argin)>=2)
                obj.(argin{1})=argin{2};
                argin=argin(3:end);
            end
        end
        
        function val=Get2DSlice(obj,dataName, sliceIndex)
            switch dataName
                case 'volumeImage'
                    val=obj.volumeImage(:,:,sliceIndex);
                case 'probabilityImage'
                    val=obj.probabilityImage(:,:,sliceIndex);
                case 'segImage'
                    val=obj.segImage(:,:,sliceIndex);
                otherwise
                    error([prop_name,'is not a valid image']);
            end
        end
        
        function OpenImage(obj,imgFolderName)
            % read volume image from a folder, which contains a chain of
            % *.png images indexed from 1 to the number of slices.
            dirinfo=dir(fullfile(imgFolderName,'*.png'));
            sliceNumber=length(dirinfo);
            
            longfilename=fullfile(imgFolderName,'1.png');
            I=imread(longfilename);
            size2d=size(I);
            size3d=[size2d, sliceNumber];
            volume=uint8(zeros(size3d));
            for i=1:sliceNumber
                tempfilename=fullfile(imgFolderName,[num2str(i) '.png']);
                tempI=imread(tempfilename);
                volume(:,:,i)=tempI(:,:);
            end
            obj.volumeImage = volume;
            
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
            obj.seedImage=uint8(zeros(obj.imageSize(1),obj.imageSize(2)));
            obj.segImage=uint8(zeros(obj.imageSize));
            obj.probabilityImage=zeros(obj.imageSize);
        end
        
        function SaveSegmentationResult(obj,segSaveFolder)
            for index=1:obj.imageSize(3)
                segFileName=fullfile(segSaveFolder,[num2str(index) '_seg.png']);
                imwrite(obj.segImage(:,:,index)*255,segFileName);
            end
            
        end
        
        function StartSliceSegmentation(obj)
            if(obj.startIndex==0)
                error('slice index should not be 0');
            end
            % segmentation in the start slice
            SeedLabel=obj.GetSeedLabelImage();
            currentSeedLabel  = SeedLabel;
            currentTrainLabel = SeedLabel;
            currentSegIndex   = obj.startIndex;
            obj.Train(currentTrainLabel,SlicSegAlgorithm.GetSliceFeature(obj.volumeImage(:,:,currentSegIndex)));
            obj.probabilityImage(:,:,currentSegIndex)=obj.randomForest.PredictUsingConnectivity(currentSeedLabel,obj.volumeImage(:,:,currentSegIndex));
            obj.segImage(:,:,currentSegIndex)=SlicSegAlgorithm.GetSingleSliceSegmentation(currentSeedLabel,obj.volumeImage(:,:,currentSegIndex),obj.probabilityImage(:,:,currentSegIndex),obj.lambda,obj.sigma);
        end
        
        function SegmentationPropagate(obj)
            % propagate to previous slices
            if(obj.sliceRange(1)==0 || obj.sliceRange(2)==0)
                error('index range should not be 0');
            end
            currentSegIndex=obj.startIndex;
            for i=1:obj.startIndex-obj.sliceRange(1)
                priorSegIndex=currentSegIndex;
                currentSegIndex=currentSegIndex-1;
                obj.TrainAndPropagate(i>1,currentSegIndex,priorSegIndex);
            end
            
            % propagate to following slices
            currentSegIndex=obj.startIndex;
            notify(obj,'SegmentationProgress',SegmentationProgressEventDataClass(currentSegIndex));
            for i=obj.startIndex:obj.sliceRange(2)-1
                priorSegIndex=currentSegIndex;
                currentSegIndex=currentSegIndex+1;
                obj.TrainAndPropagate(i>obj.startIndex,currentSegIndex,priorSegIndex);
            end
        end
        
        function RunSegmention(obj)
            obj.StartSliceSegmentation();
            obj.SegmentationPropagate();
        end
    end
    
    methods (Access=private)
        function TrainAndPropagate(obj,train,currentSegIndex,priorSegIndex)
            [currentSeedLabel,currentTrainLabel]=SlicSegAlgorithm.UpdateSeedLabel(obj.segImage(:,:,priorSegIndex),obj.innerDis,obj.outerDis);
            if(train)
                obj.randomForest.Train(currentTrainLabel,SlicSegAlgorithm.GetSliceFeature(obj.volumeImage(:,:,priorSegIndex)));
            end
            obj.probabilityImage(:,:,currentSegIndex)=obj.randomForest.PredictUsingPrior(obj.volumeImage(:,:,currentSegIndex),obj.segImage(:,:,priorSegIndex));
            obj.segImage(:,:,currentSegIndex)=SlicSegAlgorithm.GetSingleSliceSegmentation(currentSeedLabel,obj.volumeImage(:,:,currentSegIndex),obj.probabilityImage(:,:,currentSegIndex),obj.lambda,obj.sigma);
            notify(obj,'SegmentationProgress',SegmentationProgressEventDataClass(currentSegIndex));            
        end
        
        function Label=GetSeedLabelImage(obj)
            Label=obj.seedImage;
            [H,W]=size(Label);
            for i=5:5:H-5
                Label(i,5)=255;
                Label(i,W-5)=255;
            end
            for j=5:5:W-5
                Label(5,j)=255;
                Label(H-5,j)=255;
            end
        end
    end
    
    methods (Static, Access=private)
        function featureMatrix=GetSliceFeature(I)
            % get the feature matrix for given slice
            dwtFeature=image2DWTfeature(I);
            hogFeature=image2HOGFeature(I);
            %             lbpFeature=image2LBPFeature(I);
            intensityFeature=image2IntensityFeature(I);
            % glmcfeatures=image2GLCMfeature(I);
            % featureMatrix=[intensityFeature dwtFeature];% glmcfeatures];
            featureMatrix=[intensityFeature hogFeature dwtFeature];
        end
        
        function seg = GetSingleSliceSegmentation(currentSeedLabel,currentI,currentP,lambda,sigma)
            % use max flow to get the segmentatio in one slice
            currentSeed=currentSeedLabel;
            
            [flow, currentSegLabel]=wgtmaxflowmex(currentI,currentSeed,currentP,lambda,sigma);
            currentSegLabel=1-currentSegLabel;
            se= strel('disk',2);
            currentSegLabel=imclose(currentSegLabel,se);
            currentSegLabel=imopen(currentSegLabel,se);
            seg=currentSegLabel(:,:);
        end
        
        function [currentSeedLabel,currentTrainLabel] = UpdateSeedLabel(currentSegImage,fgr,bgr)
            % generate new training data (for random forest) and new seeds
            % (hard constraint for max-flow) based on segmentation in last slice
            
            tempSegLabel=currentSegImage;
            fgSe1= strel('disk',fgr);
            fgMask=imerode(tempSegLabel,fgSe1);
            if(length(find(fgMask>0))<100)
                fgMask=bwmorph(tempSegLabel,'skel',Inf);
            else
                fgMask=bwmorph(fgMask,'skel',Inf);
            end
            bgSe1= strel('disk',bgr);
            bgSe2= strel('disk',bgr+1);
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

