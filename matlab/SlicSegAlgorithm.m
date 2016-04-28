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
        function d=SlicSegAlgorithm
            d.startIndex=0;
            d.sliceRange=[0,0];
            d.randomForest = RandomForestWrapper;
            
            d.lambda=5.0;
            d.sigma=3.5;
            d.innerDis=5;
            d.outerDis=6;
        end
        
        function imageSize = get.imageSize(d)
            if isempty(d.volumeImage)
                imageSize=[0,0,0];
            else
                imageSize=size(d.volumeImage);
            end
        end
        
        function set.volumeImage(d,volumeImage)
            d.volumeImage=volumeImage;
            d.ResetSegmentationResult();
        end
        
        function SetMultipleProperties(d,varargin)
            argin=varargin;
            while(length(argin)>=2)
                d.(argin{1})=argin{2};
                argin=argin(3:end);
            end
        end
        
        function val=Get2DSlice(d,dataName, sliceIndex)
            switch dataName
                case 'volumeImage'
                    val=d.volumeImage(:,:,sliceIndex);
                case 'probabilityImage'
                    val=d.probabilityImage(:,:,sliceIndex);
                case 'segImage'
                    val=d.segImage(:,:,sliceIndex);
                otherwise
                    error([prop_name,'is not a valid image']);
            end
        end
        
        function OpenImage(d,imgFolderName)
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
            d.volumeImage = volume;
            
        end
        
        function OpenScribbleImage(d,labelFileName)
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
            d.seedImage = ILabel;
            disp('seed image has been loaded successfully');
            
        end
        
        function ResetSegmentationResult(d)
            d.seedImage=uint8(zeros(d.imageSize(1),d.imageSize(2)));
            d.segImage=uint8(zeros(d.imageSize));
            d.probabilityImage=zeros(d.imageSize);
        end
        
        function SaveSegmentationResult(d,segSaveFolder)
            for index=1:d.imageSize(3)
                segFileName=fullfile(segSaveFolder,[num2str(index) '_seg.png']);
                imwrite(d.segImage(:,:,index)*255,segFileName);
            end
            
        end
        
        function StartSliceSegmentation(d)
            if(d.startIndex==0)
                error('slice index should not be 0');
            end
            % segmentation in the start slice
            SeedLabel=d.GetSeedLabelImage();
            currentSeedLabel  = SeedLabel;
            currentTrainLabel = SeedLabel;
            currentSegIndex   = d.startIndex;
            d.Train(currentTrainLabel,SlicSegAlgorithm.GetSliceFeature(d.volumeImage(:,:,currentSegIndex)));
            d.probabilityImage(:,:,currentSegIndex)=d.randomForest.PredictUsingConnectivity(currentSeedLabel,d.volumeImage(:,:,currentSegIndex));
            d.segImage(:,:,currentSegIndex)=SlicSegAlgorithm.GetSingleSliceSegmentation(currentSeedLabel,d.volumeImage(:,:,currentSegIndex),d.probabilityImage(:,:,currentSegIndex),d.lambda,d.sigma);
        end
        
        function SegmentationPropagate(d)
            % propagate to previous slices
            if(d.sliceRange(1)==0 || d.sliceRange(2)==0)
                error('index range should not be 0');
            end
            currentSegIndex=d.startIndex;
            for i=1:d.startIndex-d.sliceRange(1)
                priorSegIndex=currentSegIndex;
                currentSegIndex=currentSegIndex-1;
                d.TrainAndPropagate(d,i>1,currentSegIndex,priorSegIndex);
            end
            
            % propagate to following slices
            currentSegIndex=d.startIndex;
            notify(d,'SegmentationProgress',SegmentationProgressEventDataClass(currentSegIndex));
            for i=d.startIndex:d.sliceRange(2)-1
                priorSegIndex=currentSegIndex;
                currentSegIndex=currentSegIndex+1;
                d.TrainAndPropagate(d,i>d.startIndex,currentSegIndex,priorSegIndex);
            end
        end
        
        function RunSegmention(d)
            d.StartSliceSegmentation();
            d.SegmentationPropagate();
        end
    end
    
    methods (Access=private)
        function TrainAndPropagate(d,train,currentSegIndex,priorSegIndex)
            [currentSeedLabel,currentTrainLabel]=SlicSegAlgorithm.UpdateSeedLabel(d.segImage(:,:,priorSegIndex),d.innerDis,d.outerDis);
            if(train)
                d.randomForest.Train(currentTrainLabel,SlicSegAlgorithm.GetSliceFeature(d.volumeImage(:,:,priorSegIndex)));
            end
            d.probabilityImage(:,:,currentSegIndex)=d.randomForest.PredictUsingPrior(d.volumeImage(:,:,currentSegIndex),d.segImage(:,:,priorSegIndex));
            d.segImage(:,:,currentSegIndex)=SlicSegAlgorithm.GetSingleSliceSegmentation(currentSeedLabel,d.volumeImage(:,:,currentSegIndex),d.probabilityImage(:,:,currentSegIndex),d.lambda,d.sigma);
            notify(d,'SegmentationProgress',SegmentationProgressEventDataClass(currentSegIndex));            
        end
        
        function Label=GetSeedLabelImage(d)
            Label=d.seedImage;
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

