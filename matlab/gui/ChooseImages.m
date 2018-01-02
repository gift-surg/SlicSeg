function [loadedImage, representativeMetadata, sliceLocations] = ChooseImages
    % ChooseImages Prompts the user to select a image (.png, .dicom or .nii). 
    % If a .png or .dcm image is selected, the entire series will be loaded 
    %
    %     Syntax
    %     ------
    %
    %         [loadedImage, representativeMetadata] = ChooseImages
    %
    %     The user selects an image in a folder. If a .png file is
    %     selected, then all .png files in that directory are loaded into an
    %     image volume. If a Dicom image is selected then all Dicom files in 
    %     that directory are loaded and reconstructed into a volume and representative
    %     metadata is also returned. If a .nii image is selected, the
    %     corresponding 3D volume is loaded directly.
    %
    %
    %     Note: This requires the dicomat library
    %     (https://github.com/tomdoel/dicomat) and matlab NifiTi tool
    %     (https://uk.mathworks.com/matlabcentral/fileexchange/8797-tools-for-nifti-and-analyze-image)
    %
    %     Run DMAddPaths to temporarily add the paths required for DicoMat to Matlab
    %    
    %     Author: Tom Doel, 2015.
    %     Translational Imaging Group, UCL  http://cmictig.cs.ucl.ac.uk
    %

    
    % Store the last selcted folder during this session, so that the select folder dialog goes there automatically
    persistent lastLoadedFolder
    lastLoadedFolder
    loadedImage = [];
    representativeMetadata = [];
    sliceLocations = [];
    
    % If this is the first time this is run, then we start at the user's
    % home directory
    if isempty(lastLoadedFolder)
        lastLoadedFolder = CoreDiskUtilities.GetUserDirectory;
    end
    
    % Prompt the user to select a file.
    [FileName, DirName] = uigetfile({'*'},'Select image file',lastLoadedFolder);
    % Prompt the user to select a folder. 
    %importDir = CoreDiskUtilities.ChooseDirectory('Select a directory from which files will be imported', lastLoadedFolder);
    
    % If cancel is selected, return an empty image
    if FileName == 0
        return
    end
    
    % Store the selected folder so 
    lastLoadedFolder = DirName;
    
    % Get a list of png files in this directory and its subdirectories
    if(endsWith(FileName,'.png'))
        pngFileNames = CoreDiskUtilities.GetRecursiveListOfFiles(DirName, '*.png');
        representativeMetadata = [];
        sliceLocations = [];
        
        % Rather than load all png files, we only load them from the first folder we found containing png files
        [pngBaseFolder, ~, ~] = fileparts(pngFileNames{1});
        
        % Now get a list of png filenames in that folder
        pngFileNames = CoreDiskUtilities.GetRecursiveListOfFiles(pngBaseFolder, '*.png');
        
        % Sort the filenames so that the images are in numerical order
        pngFileNames = CoreTextUtilities.SortFilenames(pngFileNames);
        
        % Read the first image
        firstImage = imread(pngFileNames{1});
        
        % Work out the size of the 3D image
        imageSize = [size(firstImage), numel(pngFileNames)];
        
        % Create the empty 3D image
        loadedImage = zeros(imageSize, 'like', firstImage);
        
        % Read in the remaining images
        for imageIndex = 2 : numel(pngFileNames)
            loadedImage(:, :, imageIndex) = imread(pngFileNames{imageIndex});
        end
        
    elseif(endsWith(FileName,'.nii') || endsWith(FileName,'.nii.gz'))
        niiStruct = load_untouch_nii(fullfile(DirName,FileName));
        loadedImage = niiStruct.img;
    else
    % try to load Dicom images from the selected folder and its subdirectories
        
        % Load the largest image series from Dicom files
        try
            [imageWrapper, representativeMetadata, ~, ~, sliceLocations] = DMFindAndLoadMainImageFromDicomFiles(DirName);
        catch ex
            return
        end
        
        % Extract the image from its wrapper
        loadedImage = imageWrapper.RawImage;
    end
end