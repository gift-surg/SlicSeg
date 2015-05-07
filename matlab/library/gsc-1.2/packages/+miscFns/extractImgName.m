function imgName=extractImgName(fullImgName)
  fullImgName(fullImgName=='\')='/';
  [slashes]=regexp(fullImgName,'/');
  if(isempty(slashes)) lastSlash=0;
  else lastSlash=slashes(end);
  end;

  imgName=fullImgName(lastSlash+1:end);
  imgName=imgName(1:end-4); % Removing the extension

end
