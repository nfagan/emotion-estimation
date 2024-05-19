data_root = 'D:\data\changlab\ilker_collab';

% fs = shared_utils.io.find( fullfile(data_root, 'activations/d3dfr', ["train", "valid"]), '.h5' );
% 
% fs = shared_utils.io.find( fullfile(data_root, 'activations/d3dfr', ["train_expression_balanced"]), '.h5' );
% src_p = fullfile( data_root, 'activations/d3dfr/train_var_subsample' );

fs = shared_utils.io.find( fullfile(data_root, 'activations/resnet_image_embedding' ...
  , ["train_expression_balanced", "valid_expression_balanced_phase_scrambled"]), '.h5' );
src_p = [];

if ( 0 )
  fs = fs(contains(fs, 'layer1'));
end

allow_overwrite = false;
targ_dim = 1000;
order_by_var = true;

for i = 1:numel(fs)
  fprintf( '\n %d of %d', i, numel(fs) );
  
  src_f = fs{i};
  
  split_p = strsplit( src_f, filesep );
  is_train = contains( split_p{end-1}, 'train' );
%   if ( ~is_train ), assert( strcmp(split_p{end-1}, 'valid') ); end
  
  if ( order_by_var )
    split_dir = sprintf( '%s_var_subsample_%d', split_p{end-1}, targ_dim );
  else
    split_dir = sprintf( '%s_subsample', split_p{end-1} );
  end
  
  split_p{end-1} = split_dir;
  
  dst_p = fullfile( fileparts(fileparts(src_f)), split_p{end-1:end} );
  shared_utils.io.require_dir( fileparts(dst_p) );
  
  if ( ~allow_overwrite && exist(dst_p) > 0 )
    fprintf( '\n Skipping %s because it already exists.', dst_p );
    continue
  end
  
  act_info = h5info( src_f, '/activations' );
  act_size = act_info.Dataspace.Size;
  
  if ( ~is_train )
    % use same index for validation as for training
    fprintf( '\n Using cached index for "%s"', dst_p );
    index_p = fullfile( strrep(fileparts(dst_p), 'valid', 'train') ...
      , sprintf('%s_index.mat', shared_utils.io.filenames(dst_p)) );
    index_p = strrep( index_p, '_phase_scrambled', '' );
    select_idx = shared_utils.io.fload( index_p );
  elseif ( ~isempty(src_p) )
    index_p = fullfile( src_p, sprintf('%s_index.mat', shared_utils.io.filenames(dst_p)) );
    fprintf( '\n\t Using "%s".', index_p );
    select_idx = shared_utils.io.fload( index_p );
  else
    if ( act_size(1) > targ_dim )
      if ( order_by_var )
        feat_vars = feature_variances( src_f );
        [~, select_idx] = sort( feat_vars, 'descend' );
        select_idx = select_idx(1:targ_dim);
  %       select_idx = sort( randperm(act_size(1), targ_dim) );
      else
        select_idx = sort( randperm(act_size(1), targ_dim) );
      end
    else
      select_idx = 1:act_size(1);
    end
  end
  
  subsampled_acts = act_subsample( src_f, select_idx );
  
  ds_info = h5info( src_f );
  cp_names = setdiff( {ds_info.Datasets.Name}, 'activations' );
  
  dst_id = H5F.create( dst_p, 'H5F_ACC_TRUNC', 'H5P_DEFAULT', 'H5P_DEFAULT' );
  
  for j = 1:numel(cp_names)
    err = [];
    src_id = H5F.open( src_f );
    try
      H5O.copy( src_id, cp_names{j}, dst_id, cp_names{j}, 'H5P_DEFAULT', 'H5P_DEFAULT' );
    catch err
    end
    H5F.close( src_id );
    if ( ~isempty(err) ), rethrow( err ); end
  end
  
  H5F.close( dst_id );
  
  hdf5write ( dst_p, '/activations', subsampled_acts', 'writemode', 'append' );
  fname = sprintf( '%s_index.mat', shared_utils.io.filenames(dst_p) );
  save( fullfile(fileparts(dst_p), fname), 'select_idx' );
end

%%

function vars = feature_variances(src)

act_info = h5info( src, '/activations' );
act_size = act_info.Dataspace.Size;

chunk_size = min( 5e3, act_size(1) );
num_chunks = ceil( act_size(1) / chunk_size );

% vars = nan( 1, act_size(1) );
chunk_vars = cell( num_chunks, 1 );
parfor c = 1:num_chunks
  fprintf( '\n\t %d of %d', c, num_chunks );
  i0 = (c-1) * chunk_size;
  i1 = min( i0 + chunk_size, act_size(1) );
  n = i1 - i0;  
  act = h5read( src, '/activations', [i0+1, 1], [n, inf] )';
  chunk_vars{c} = var( act, 1 );
end

vars = horzcat( chunk_vars{:} );

end

function act = read_act(f, j, n)

if ( nargin < 3 ), n = 1; end
an = '/activations';
act = h5read( f, an, [1, j], [inf, n] )';

end

function ds = act_subsample(src, idx)

act_info = h5info( src, '/activations' );
act_size = act_info.Dataspace.Size;

if ( strcmp(act_info.Datatype.Class, 'H5T_FLOAT') )
  ds_type = 'single';
else
  assert( strcmp(act_info.Datatype.Class, 'H5T_DOUBLE') )
  ds_type = 'double';
end

ds = zeros( act_size(2), numel(idx), ds_type );
chunk_size = min( 100, act_size(2) );
num_chunks = ceil( act_size(2) / chunk_size );

for c = 1:num_chunks
  i0 = (c-1) * chunk_size;
  i1 = min( i0 + chunk_size, act_size(2) );
  n = i1 - i0;
  acts = read_act( src, i0+1, n );
  ds(i0+1:i0+n, :) = acts(:, idx);
end

end