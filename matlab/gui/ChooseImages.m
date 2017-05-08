function [loadedImage, representativeMetadata, sliceLocations] = ChooseImages
    % ChooseImages Prompts the user to select a directory and loads Dicom or png files
    %
    %     Syntax
    %     ------
    %
    %         [loadedImage, representativeMetadata] = ChooseImages
    %
    %     The user selects a folder. This function will search all
    %     subdirectories looking for .png files. If any .png files are
    %     found, then all png files in that directory are loaded into an
    %     image volume. Otehrwise if Dicom images are found then they are
    %     loaded and reconstructed into a volume and representative
    %     metadata is also returned.
    %
    %
    %     Note: This requires the dicomat library (https://github.com/tomdoel/dicomat)
    %
    %     Run DMAddPaths to temporarily add the paths required for DicoMat to Matlab
    %    
    %     Author: Tom Doel, 2015.
    %     Translational Imaging Group, UCL  http://cmictig.cs.ucl.ac.uk
    %

    
    % Store the last selcted folder during this session, so that the select folder dialog goes there automatically
    persistent lastLoadedFolder
    
    loadedImage = [];
    representativeMetadata = [];
    sliceLocations = [];
    
    % If this is the first time this is run, then we start at the user's
    % home directory
    if isempty(lastLoadedFolder)
        lastLoadedFolder = CoreDiskUtilities.GetUserDirectory;
    end
    
    % Prompt the user to select a folder. 
    importDir = CoreDiskUtilities.ChooseDirectory('Select a directory from which files will be imported', lastLoadedFolder);
    
    % If cancel is selected, return an empty image
    if isempty(importDir)
        return
    end
    
    % Store the selected folder so 
    lastLoadedFolder = importDir;
    
    % Get a list of png files in this directory and its subdirectories
	pngFileNames = CoreDiskUtilities.GetRecursiveListOfFiles(importDir, '*.png');
            
    % If there are no png files then we try to load Dicom images from the
    % selected folder and its subdirectories
    if isempty(pngFileNames)
        
        % Load the largest image series from Dicom files
        try
            [imageWrapper, representativeMetadata, ~, ~, sliceLocations] = DMFindAndLoadMainImageFromDicomFiles(importDir);
        catch ex
            return
        end
        
        % Extract the image from its wrapper
        loadedImage = imageWrapper.RawImage;
%         minValue = min(loadedImage(:));
%         maxValue = 800;%max(loadedImage(:));
%         loadedImage = round(255*(loadedImage - minValue)/(maxValue - minValue));
%         loadedImage = min(255, loadedImage);
%         loadedImage = max(0, loadedImage);
%         loadedImage = uint8(loadedImage);
        
    else
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
    end
end