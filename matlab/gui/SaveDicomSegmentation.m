function SaveDicomSegmentation(segImage, baseFilename, metaData, sliceLocations, reporting)
    % SaveDicomSegmentation Saves segmentation as a series of Dicom files
    %
    %     Syntax
    %     ------
    %
    %         SaveDicomSegmentation(segImage, baseFilename, metaData, sliceLocations, reporting)
    %
    %           segImage - the image matrix, which will be divided into
    %             slices along the third dimension
    %           baseFilename - filename, which will have index numbers
    %             appended to the main part
    %           metaData - representative metata for the image to be saved
    %           sliceLocations - a matrix containing Dicom patient position values
    %              for each slice (this should normally be derived from the original image)
    %           reporting - CoreReporting object for progress
    %
    %
    %     Note: This requires the dicomat library
    %     (https://github.com/tomdoel/dicomat) and the Matlab Image
    %     Processing Toolbox
    %
    %     Author: Tom Doel, 2015.
    %     Translational Imaging Group, UCL  http://cmictig.cs.ucl.ac.uk
    %
    
    softwareInfo = struct;
    softwareInfo.SecondaryCaptureDeviceManufacturer = 'SlicSeg';
    softwareInfo.DicomManufacturer = 'TIG';
    softwareInfo.DicomName = 'Slic-Seg';
    softwareInfo.DicomVersion = '1';

    metaDataToSave = metaData;
    metaDataToSave.SeriesDescription = 'Slic-Seg segmentation';

    DMSaveDicomSeries(baseFilename, segImage, sliceLocations, metaDataToSave, DMImageType.RGBLabel, softwareInfo, reporting);
end