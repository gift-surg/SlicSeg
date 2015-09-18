classdef (ConstructOnLoad) SegmentationProgressEventDataClass < event.EventData
   properties
      OrgValue = 0;
   end
   methods
      function eventData = SegmentationProgressEventDataClass(value)
         eventData.OrgValue = value;
      end
   end
end