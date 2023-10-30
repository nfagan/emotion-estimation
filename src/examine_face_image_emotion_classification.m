data_root = 'D:\data\changlab\ilker_collab';

im_p = fullfile( data_root, 'flickr-faces', 'full' );

cp_p = fullfile( data_root, 'ed_eval', 'checkpoint.mat' );
cp = load( cp_p );

preds = string( cp.label );
preds = preds(cp.prediction + 1);
pred_p = arrayfun( @(x) cp.prediction_ps(x, cp.prediction(x)+1), 1:numel(cp.prediction) );
pred_p = pred_p(:);

im_ps = fullfile( im_p, compose("%s.png", string(cp.identifier)) );

%%  show examples

p_thresh = 0.99;
neg_ind = find( preds == "negative" & pred_p > p_thresh );
pos_ind = find( preds == "positive" & pred_p > p_thresh );

% neg_im_index = neg_ind(1);
neg_im_index = neg_ind(randi(numel(neg_ind)));
% pos_im_index = pos_ind(1);
pos_im_index = pos_ind(randi(numel(pos_ind)));

neg_im = imread( im_ps(neg_im_index) );
pos_im = imread( im_ps(pos_im_index) );

figure(1); clf;
subplot( 1, 2, 1 );
imshow( neg_im );
title( "negative" );

subplot( 1, 2, 2 );
imshow( pos_im );
title( "positive" );