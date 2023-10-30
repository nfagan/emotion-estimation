dr = 'D:\data\changlab\ilker_collab';
src_p = fullfile( dr, 'd3dfr_face_recon', 'results/images' );
dst_p = fullfile( dr, 'goldenberg_faces' );
src_im_ps = shared_utils.io.find( src_p, '.png', true );

for i = 1:numel(src_im_ps)
  [~, im_name] = fileparts( src_im_ps{i} );
  pred_mask = load( ...
    fullfile(fileparts(src_im_ps{i}), sprintf('%s_pred_mask.mat', im_name)) );
  pred_mask = repmat( pred_mask.pred_mask, 1, 1, 3 );
  
  src_im = imread( src_im_ps{i} );
  assert( size(src_im, 1) == 224 && size(src_im, 2) == 672 );
  src = src_im(:, 1:224, :);
  recon = src_im(:, 225:224*2, :);
  
  src = src .* uint8( pred_mask > 0 );
  recon = recon .* uint8( pred_mask > 0 );
  
  if ( 1 )
    recon_p = fullfile( dst_p, 'reconstructed_images' );
    shared_utils.io.require_dir( recon_p );
    imwrite( recon, fullfile(recon_p, sprintf('%s.png', im_name)) );
  
    attended_p = fullfile( dst_p, 'attended_images' );
    shared_utils.io.require_dir( attended_p );
    imwrite( src, fullfile(attended_p, sprintf('%s.png', im_name)) );
  end
end