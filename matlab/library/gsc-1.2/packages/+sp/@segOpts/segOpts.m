% Copyright (C) 2009-10 Varun Gulshan
% This class defines the options for Shortest path segmentation
classdef segOpts
  properties (SetAccess=public, GetAccess=public)
    spImg % 'imgSmoothed' or 'likelihoodImg'

    % if spImg==likelihoodImg then following options apply
    posteriorMethod % gmm_bs, gmm_bs_mixtured, none
    gmmNmix_fg % number of gmm mixtures for fg, usually 2-3
    gmmNmix_bg % number of gmm mixtures for bg, usually 2-3
    gmmUni_value % uniform likelihood mixing, see constructor for default
    gmmLikeli_gamma % strength of uniform likliehood mixing
    featureSpace % rgb or rgb_xy

    % if spImg==imgSmoothed, following options apply
    sigma % the sigma for smoothing
  end

  methods
    function obj=segOpts()
      % Set the default options
      obj.spImg='likelihoodImg';

      obj.gmmNmix_fg=5;
      obj.gmmNmix_bg=5;
      obj.gmmUni_value=1; % assuming features in [0,1]
      obj.gmmLikeli_gamma=0.05;
      obj.posteriorMethod='gmm_bs_mixtured';
      obj.featureSpace='rgb';

      obj.sigma=0;
    end
  end

end % of classdef
