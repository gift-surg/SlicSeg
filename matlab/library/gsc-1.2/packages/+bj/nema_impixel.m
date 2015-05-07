function P = nema_impixel( I, cidx, ridx )

% NEMA_IMPIXEL A function that extracts pixels at integer locations 
% from an image in a similar fashion to IMPIXEL.
%
%    P = NEMA_IMPIXEL( I, C, R ) returns the N pixels located at
%    columns C and and rows R in I.  P is an NxD matrix containing
%    the N pixels from I.  C and R must be integers.
%
%    P = NEMA_IMPIXEL( I, IND ) returns the N pixels located at the
%    indices IND.  IND must be an integer.
%
%    P = NEMA_IMPIXEL( I ) returns all pixels in I.
%
% See also IMPIXEL.

% Author: Nicholas Apostoloff <nema@robots.ox.ac.uk>
% Date: 25 May 05
[ nrows, ncols, ndims ] = size( I );
I = reshape( I, [], ndims );
if nargin == 1
  P = I;
else
  if nargin == 2
    idx = cidx;
  elseif nargin == 3
    idx = sub2ind( [ nrows, ncols ], ridx, cidx );
  else
    error( 'Incorrect number of input arguments' );
  end
  P = I( idx, : );
end

% Calling impixel is very very slow because it in turn calls interp2.

% if nargin == 1
%   [ nrows, ncols, ndims ] = size( I );
%   [ ridx, cidx ] = ind2sub( [nrows, ncols], 1:nrows*ncols );
% elseif nargin == 2
%   [ nrows, ncols, ndims ] = size( I );
%   [ ridx, cidx ] = ind2sub( [nrows, ncols], cidx );
% elseif nargin == 3
% else
%   error( 'Incorrect number of input arguments' );
% end
% 
% if size( I, 3 ) == 3
%   P = impixel( I, cidx, ridx );
% else
%   for i = 1:size( I, 3 )
%     t1 = impixel( I( :, :, i ), cidx, ridx );
%     P( :, i ) = t1( :, 1 );
%   end
% end
  
