balance_expressions = true;

data_root = 'D:\data\changlab\ilker_collab';
src_im_p = fullfile( data_root, 'flickr-faces', 'full' );

sub_im_dir = 'subset';
if ( balance_expressions )
  sub_im_dir = sprintf( '%s-emotion-balanced', sub_im_dir );
end
dst_im_p = fullfile( fileparts(src_im_p), sub_im_dir );

ims = dir( src_im_p );
ims = ims(~[ims.isdir]);

% only keep images that had 1 face that could be detected reliably
detect_txts = fullfile( src_im_p, 'detections', strrep({ims.name}, '.png', '.txt') );
exist_txt = cellfun( @(x) exist(x, 'file') == 2, detect_txts );
ims = ims(exist_txt);

im_net_splits = string(h5read(fullfile(...
  data_root, 'activations/image_net/train/fc1.h5'), '/splits'));
num_train = sum( contains(im_net_splits, 'train') );
num_valid = sum( contains(im_net_splits, 'valid') );

%%

pred_p_thresh = 0.95;
include_valid = true;

if ( balance_expressions )
  pred_p = fullfile( data_root, 'ed_eval', 'checkpoint.mat' );
  pred_cp = load( pred_p );
  
  pred_ps = arrayfun( @(x) pred_cp.prediction_ps(x, pred_cp.prediction(x)+1) ...
    , 1:numel(pred_cp.prediction) );
  ok_pred = find( pred_ps(:) > pred_p_thresh );
  I = findeachv( pred_cp.prediction, ok_pred );
  assert( mod(num_train, numel(I)) == 0, 'expected evenly divisible training set.' );
  nt_per_cond = num_train / numel( I );
  train_ind = cate1( cellfun(@(x) x(1:nt_per_cond), I, 'un', 0) );
  
  targ_ident = string( pred_cp.identifier );
  target_names = compose( "%s.png", targ_ident(train_ind) );
  [~, train_ind] = ismember( target_names, string({ims.name}) );
  
  include_valid = false;
else
  train_ind = 1:num_train;
  valid_ind = num_train+1:num_train+num_valid;
end

train_ims = ims(train_ind);

if ( include_valid )
  valid_ims = ims(valid_ind);
end

% error( 'xx' );

do_copy( train_ims, fullfile(dst_im_p, 'train') );

if ( include_valid )
  do_copy( valid_ims, fullfile(dst_im_p, 'valid') );
end

%%

function do_copy(files, dst_p)

if ( exist(dst_p, 'dir') == 0 )
  mkdir( dst_p );
end

detects_p = fullfile( dst_p, 'detections' );
if ( exist(detects_p, 'dir') == 0 )
  mkdir( detects_p );
end

for i = 1:numel(files)
  fprintf( '\n %d of %d', i, numel(files) );
  copyfile( fullfile(files(i).folder, files(i).name) ...
    , fullfile(dst_p, files(i).name) );
  
  detects_name = strrep( files(i).name, '.png', '.txt' );
  copyfile( fullfile(files(i).folder, 'detections', detects_name) ...
    , fullfile(detects_p, detects_name) );
end

end