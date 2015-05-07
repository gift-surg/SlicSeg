function posteriorImage=getPosteriorImage(features,labelImg,segOpts)
% Function to compute posteriors, by learning color models
% from the user strokes in labelImg
% 
% features is a D x N array , which contains the features at each
% pixel (of the h x w x nFrames video)
% labelImg is uint8 h x w x nFrames array
% segOpts is a segmentation options structure documeted
% in videoOptions.m

switch(segOpts.posteriorMethod)
  case 'gmm_bs'
    posteriorImage=do_gmmBS_posteriorImage(features,labelImg,segOpts);
  case 'gmm_bs_mixtured'
    posteriorImage=do_gmmBS_posteriorImage_mixtured(features,labelImg,segOpts);
  case 'none'
    posteriorImage=0.5*ones(size(labelImg));
  otherwise
    error('Unsupported posteriorMethod: %s\n',segOpts.posteriorMethod);
end

function posteriors=do_gmmBS_posteriorImage_mixtured(features,labelImg,segOpts)
import bj.gmm.*;
% --- learn the gmm first --------
fgFeatures=features(:,labelImg(:)==1);
fgGmm=init_gmmBS(fgFeatures,segOpts.gmmNmix_fg);

bgFeatures=features(:,labelImg(:)==2);
bgGmm=init_gmmBS(bgFeatures,segOpts.gmmNmix_bg);

% --- Now compute posteriors ------
posteriors=compute_gmmPosteriors_mixtured(features,fgGmm,bgGmm,segOpts.gmmLikeli_gamma,segOpts.gmmUni_value);
posteriors=reshape(posteriors,size(labelImg));

function posteriors=do_gmmBS_posteriorImage(features,labelImg,segOpts)

import bj.gmm.*;
% --- learn the gmm first --------
fgFeatures=features(:,labelImg(:)==1);
fgGmm=init_gmmBS(fgFeatures,segOpts.gmmNmix_fg);

bgFeatures=features(:,labelImg(:)==2);
bgGmm=init_gmmBS(bgFeatures,segOpts.gmmNmix_bg);

% --- Now compute posteriors ------
posteriors=compute_gmmPosteriors(features,fgGmm,bgGmm);
posteriors=reshape(posteriors,size(labelImg));
