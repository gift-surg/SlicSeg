classdef ImageWrapper < handle
    properties
        rawImage
    end
    
    properties (Dependent)
        ImageSize % Returns the image size
    end    
    
    methods
        function obj = ImageWrapper(imageData)
            if nargin > 0
                obj.rawImage = imageData;
            end
        end
        
        function imageSize = getImageSize(obj)
             % Returns a 3x1 matrix, size of image (width, height, slices)
            if isempty(obj.rawImage)
                imageSize = [0,0,0];
            else
                imageSize = size(obj.rawImage);
            end
        end        
        
        function slice = get2DSlice(obj, sliceNumber, dimension)
            % Returns a 2D slice from the image in the specified direction
            
           switch dimension
               case 1
                   slice = squeeze(obj.rawImage(sliceNumber, :, :));
               case 2
                   slice = squeeze(obj.rawImage(:, sliceNumber, :));
               case 3
                   slice = squeeze(obj.rawImage(:, :, sliceNumber));
               otherwise
                   error('Unsupported dimension');
           end
        end
        
        function sliceSize = get2DSliceSize(obj, dimension)
            % Returns a 2D slice from the image in the specified direction
            
            imageSize = obj.getImageSize;
            switch dimension
                case 1
                    sliceSize = [imageSize(2), imageSize(3)];
                case 2
                    sliceSize = [imageSize(1), imageSize(3)];
                case 3
                    sliceSize = [imageSize(1), imageSize(2)];
                otherwise
                    error('Unsupported dimension');
            end
        end
        
        function replaceImageSlice(obj, newSlice, sliceIndex, direction)
            % Modifies the specified 2D slice of the image
            
            switch direction
                case 1
                    obj.rawImage(sliceIndex, :, :) = newSlice;
                case 2
                    obj.rawImage(:, sliceIndex, :) = newSlice;
                case 3
                    obj.rawImage(:, :, sliceIndex) = newSlice;
            end
        end
    end
    
end

