function folder=getSaveFolderName(handles)

global startImageIndex;
global imgFolderName;
username=get(handles.edit_username,'String');
if(length(username)>1)
    user_image_folder=[imgFolderName(1:length(imgFolderName)-4) username '/' num2str(startImageIndex)];
    numberfile=[user_image_folder '/number.txt'];
    fid=fopen(numberfile);
    ntrial=0;
    if(fid==-1)
        if(~exist(user_image_folder,'dir'))
            mkdir(user_image_folder);
        end
    else
        ntrial= fscanf(fid,'%u');
        fclose(fid);
    end
    ntrial=ntrial+1;
    fid=fopen(numberfile,'wt+');
    fprintf(fid,'%u',ntrial);
    fclose(fid);
    folder=[user_image_folder '/trial' num2str(ntrial)];
    mkdir(folder);
else
    folder='n';
end
