data_root = 'D:\data\changlab\ilker_collab';

fs = shared_utils.io.find( fullfile(data_root, 'activations/resnet_image_embedding'), '.h5', true );
fs = fs(contains(fs, 'valid_expression_balanced_phase_scrambled_var_subsample_1000'));

dst_p = fullfile( data_root, 'distinctiveness' );

use_custom = false;
custom_mask = 1:144;  % (R^80) = identity + beta (R^64) = expression
custom_dst_layer = 'ReconNetWrapper_output_identity_expression';

for i = 1:numel(fs)
  fprintf( '\n %d of %d', i, numel(fs) );
  
  full_dst_p = fullfile( dst_p, strrep(fs{i}, fullfile(data_root, 'activations'), '') );
  full_dst_p = strrep( full_dst_p, '.h5', '.mat' );
  
  if ( use_custom )
    [p, name] = fileparts( full_dst_p );
    full_dst_p = fullfile( p, sprintf('%s.mat', custom_dst_layer) );
  end
  
  if ( exist(full_dst_p, 'file') )
    fprintf( '\n Skipping: %s', full_dst_p );
    continue
  end
  
  an = '/activations';
  valid_f = fs{i};
  act_info = h5info( valid_f, an );
  act_size = act_info.Dataspace.Size;
  
  train_f = strrep( fs{i}, 'valid', 'train' );
  train_f = strrep( train_f, '_phase_scrambled', '' );
  
  train_info = h5info( train_f, an );
  train_size = train_info.Dataspace.Size;
  
  valid_ds = nan( act_size(2), 1 );
  parfor j = 1:act_size(2)
%     fprintf( '\n %d of %d', j, act_size(2) );
    act = read_act( valid_f, j );
    ds = act_dist( act, valid_f, custom_mask );
    valid_ds(j) = min( ds(ds > 0) );
  end
  
  train_ds = nan( size(valid_ds) );
  parfor j = 1:act_size(2)
%     fprintf( '\n %d of %d', j, act_size(2) );
    act = read_act( valid_f, j );
    ds = act_dist( act, train_f, custom_mask );
    train_ds(j) = min( ds(ds > 0) );
  end
  
  distinct = min( valid_ds, train_ds );
  
  shared_utils.io.require_dir( fileparts(full_dst_p) );
  save( full_dst_p, 'distinct' );    
end

%%

function act = read_act(f, j, n)

if ( nargin < 3 ), n = 1; end
an = '/activations';
act = h5read( f, an, [1, j], [inf, n] )';

end

function ds = act_dist(act, src, mask)

if ( ~isempty(mask) )
  act = act(:, mask);
end

an = '/activations';
act_info = h5info( src, an );
act_size = act_info.Dataspace.Size;

ds = nan( act_size(2), 1 );
chunk_size = min( 100, act_size(2) );
num_chunks = ceil( act_size(2) / chunk_size );

for c = 1:num_chunks
  i0 = (c-1) * chunk_size;
  i1 = min( i0 + chunk_size, act_size(2) );
  n = i1 - i0;
  acts = read_act( src, i0+1, n );
  if ( ~isempty(mask) )
    acts = acts(:, mask);
  end
  chunk_ds = vecnorm( act - acts, 2, 2 );
  ds(i0+1:i0+n) = chunk_ds;
end

end