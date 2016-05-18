classdef (ConstructOnLoad) SegmentationProgressEventDataClass < event.EventData
    % SegmentationProgressEventDataClass Used to notify listeners of a
    % change in the currently processed slice segmentation
    %
    % Author: Guotai Wang
    % Copyright (c) 2014-2016 University College London, United Kingdom. All rights reserved.
    % http://cmictig.cs.ucl.ac.uk
    %
    % Distributed under the BSD-3 licence. Please see the file licence.txt 
    % This software is not certified for clinical use.
    
   properties
      OrgValue = 0;
   end
   methods
      function eventData = SegmentationProgressEventDataClass(value)
         eventData.OrgValue = value;
      end
   end
end