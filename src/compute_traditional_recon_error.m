%%  full images

dr = 'D:\data\changlab\ilker_collab';

is_masked = true;

src_p = fullfile( dr, 'd3dfr_face_recon', 'results/images' );
dst_p = fullfile( dr, 'd3dfr_face_recon', 'recon_error' );
dst_name = 'goldenberg_faces';
if ( is_masked ), dst_name = sprintf('%s_masked', dst_name); end

src_im_ps = shared_utils.io.find( src_p, '.png', true );

recon_err = nan( numel(src_im_ps), 1 );
for i = 1:numel(src_im_ps)
  [~, im_name] = fileparts( src_im_ps{i} );
  pred_mask = load( fullfile(fileparts(src_im_ps{i}), sprintf('%s_pred_mask.mat', im_name)) );
  pred_mask = repmat( pred_mask.pred_mask, 1, 1, 3 );
  
  src_im = imread( src_im_ps{i} );
  src_im = double( src_im ) ./ 255;
  assert( size(src_im, 1) == 224 && size(src_im, 2) == 672 );
  src = src_im(:, 1:224, :);
  recon = src_im(:, 225:224*2, :);
  
  if ( is_masked )
    src = src(pred_mask > 0);
    recon = recon(pred_mask > 0);
  end
  
  recon_err(i) = norm( src(:) - recon(:) );
end

idents = shared_utils.io.filenames( src_im_ps );
err_tbl = table( recon_err, string(idents(:)), 'va', {'recon_error', 'identifier'} );

if ( 1 )
  shared_utils.io.require_dir( dst_p );
  save( fullfile(dst_p, sprintf('%s.mat', dst_name)), 'err_tbl' );
end

%%  resnet activations

dr = 'D:\data\changlab\ilker_collab';

embed_name = 'resnet_image_embedding';
act_p = fullfile( dr, 'activations', embed_name, 'goldenberg_faces' );
attend_h5s = shared_utils.io.find( fullfile(act_p, 'attended_images'), '.h5' );
recon_h5s = shared_utils.io.find( fullfile(act_p, 'reconstructed_images'), '.h5' );
attend_layer_names = shared_utils.io.filenames( attend_h5s );
recon_layer_names = shared_utils.io.filenames( recon_h5s );

dst_p = fullfile( dr, 'd3dfr_face_recon', 'recon_error' );
dst_name = sprintf( 'goldenberg_faces_%s', embed_name );

err_tbls = cell( numel(attend_layer_names), 1 );
for i = 1:numel(attend_layer_names)
  ln = attend_layer_names{i};
  ri = find( strcmp(recon_layer_names, ln) );
  assert( numel(ri) == 1 );
  
  attend_ident = deblank( string(h5read(attend_h5s{i}, '/identifiers')) );
  recon_ident = deblank( string(h5read(recon_h5s{i}, '/identifiers')) );
  assert( isequal(attend_ident, recon_ident) );
  
  attend_acts = h5read( attend_h5s{i}, '/activations' )';
  recon_acts = h5read( recon_h5s{ri}, '/activations' )';
  recon_err = vecnorm( attend_acts - recon_acts, 2, 2 );  
  
  err_name = sprintf( 'recon_error_%s_%s', embed_name, ln );
  err_tbl = table( recon_err(:), 'va', {err_name} );
  
  if ( i == 1 )
    search_ord = attend_ident(:);
  else
    [~, lb] = ismember( attend_ident(:), search_ord );
    err_tbl = err_tbl(lb, :);
  end
  
  err_tbls{i} = err_tbl;
end

err_tbl = horzcat( err_tbls{:} );
err_tbl.identifier = search_ord;

if ( 1 )
  shared_utils.io.require_dir( dst_p );
  save( fullfile(dst_p, sprintf('%s.mat', dst_name)), 'err_tbl' );
end