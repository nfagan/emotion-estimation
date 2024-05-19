ds_p = 'C:\Users\nick\source\changlab\ilker_collab\Deep3DFaceRecon_pytorch\datasets\examples';
im_name = '000002.jpg';

ds_p = 'D:\data\changlab\ilker_collab\goldenberg_faces\images';
im_name = 'A62.jpg';

ds_p = 'D:\data\changlab\ilker_collab\flickr-faces-subset\train';
im_name = '00018.png';

ds_p = 'D:\data\changlab\ilker_collab\dc-face-stimuli';
im_name = 'am46nes.png';

ds_p = 'D:\data\changlab\ilker_collab\abscreen_to_run';
im_name = 'am57nes.png';

detect_p = fullfile( ds_p, 'detections', sprintf('%s.txt', file_name(im_name)) );
im_p = fullfile( ds_p, sprintf('%s', im_name) );

im = imread( im_p );
detects = dlmread( detect_p );

figure(1); clf; imshow( im ); hold on;
scatter( detects(:, 1), detects(:, 2) );

%%

function name = file_name(name)
[~, name] = fileparts( name );
end