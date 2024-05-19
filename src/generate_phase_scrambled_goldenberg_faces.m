dr = 'D:\data\changlab\ilker_collab';
dst_p = fullfile( dr, 'goldenberg_faces/phase_scrambled_images' );
src_p = fullfile( dr, 'goldenberg_faces/images' );
src_im_ps = shared_utils.io.find( src_p, '.jpg' );

for i = 1:numel(src_im_ps)
  [~, im_name] = fileparts( src_im_ps{i} );
  
  src_im = imread( src_im_ps{i} );
  src_im = double( src_im ) ./ 255;
  dst_im = phase_scramble( src_im );
  
  if ( 0 )
    figure(1); clf; imshow( dst_im ); pause( 2 );
  end
  
  dst_im_p = fullfile( dst_p, sprintf('%s.png', im_name) );
  shared_utils.io.require_dir( dst_p );
  imwrite( dst_im, dst_im_p );
end