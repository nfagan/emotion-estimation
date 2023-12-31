ROOT = 'C:\Users\nick\source\changlab\ilker_collab\EIG-faces';

baselpath = strcat(ROOT, '/bfm09-generator', '/bfm_utils/PublicMM1');
baselmatlabpath = strcat(baselpath, '/matlab');
addpath(baselpath);
addpath(baselmatlabpath);

% load('../neural_analysis/mask_faces.mat');
load(fullfile(ROOT, 'neural_analysis/mask_faces.mat'))
mask = tr;

rng shuffle;

[model, msz] = load_model();

out_p = fullfile(ROOT, 'infer_render_using_eig/output');

OUTDIR = fullfile( out_p, 'renderings' );
inputfile = fullfile( out_p, 'infer_output.hdf5' );

network_features = h5read(inputfile, '/latents');
network_features = transpose(network_features);
network_features = network_features/10;
filenames = h5read(inputfile, '/filenames');

attendedfaces = h5read(inputfile, '/Att');
attendedfaces = transpose(attendedfaces);

N = size(network_features);
N = N(1);

mkdir(OUTDIR);
DIM = 50;

rp     = defrp;
rp.dir_light.dir = [0;1;1];
rp.dir_light.intens = .6.*ones(3,1);
rp.mode_az = 0;
rp.mode_el = 0;
rp.phi = 0;
rp.elevation = 0;
rp.width = 300; rp.height=300;

for i = 1:N
    input = network_features(i, :);

    shape = input(1:200);
    texture = input(201:400);
    alpha = reshape(shape, [DIM,4]);
    beta = reshape(texture, [DIM,4]);
    [shape, tex, tl] = coeffs_to_head(alpha, beta, DIM, 1);

    handle = display_intrinsics(shape, tex, tl, rp, rp.mode_az, rp.mode_el, 1, mask, 1);
    %handle = display_face(shape, tex, model.tl, rp, rp.mode_az, rp.mode_el, []);
%     img = hardcopy(handle, '-dopengl', '-r300');
%     img = getframe;
    img = print('-RGBImage'); % if you are using matlab versions newer than 2016b.
    img = imresize(img, [300, 300]);
    imwrite(img, strcat(OUTDIR, filenames{i}, '_render', '.png'), 'png');

    attend = attendedfaces(i, :);
    attend = reshape(attend, 227, 227, 3);
    attend = uint8(attend);
    attend = permute(attend, [2, 1, 3]);
    imwrite(attend, strcat(OUTDIR, filenames{i}, '_attended', '.png'), 'png');

end;
