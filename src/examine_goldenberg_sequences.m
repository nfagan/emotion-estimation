% `face_number` - sequence length
% `face_mean` - true mean rating
% `estimation` - estimated mean
% `x1-x12` - true emotion ratings of faces

data_root = 'D:\data\changlab\ilker_collab\goldenberg_faces';
seq_csv = 'sequential_study1_Jan2020_just_amplification_different_sequences.csv';
qualtrics_csv = 'study1a_qualtrics.csv';

% im_p = 'C:\Users\nick\source\changlab\ilker_collab\task-sequential-faces-emotion-estimation\img';
im_p = fullfile( data_root, 'images' );

seq_tbl = readtable( fullfile(data_root, seq_csv), 'readva', true );
qt_tbl = readtable( fullfile(data_root, qualtrics_csv), 'readva', true );

clean_vars = arrayfun( @(x) sprintf('x%d', x), 1:12, 'un', 0 );
for i = 1:numel(clean_vars)
  cv = clean_vars{i};
  if ( ~isnumeric(seq_tbl.(cv)) )
    seq_tbl.(cv) = str2double( seq_tbl.(cv) );
  end
end

% remove outliers
% see sequential_analysis_1aJan.Rmd
[I, ids] = findeach( seq_tbl, 'participant_id' );
mean_est = cellfun( @(x) mean(seq_tbl.estimation(x)), I );
rem_ids = ids.participant_id(mean_est < 10 | mean_est > 40);
seq_tbl(ismember(seq_tbl.participant_id, rem_ids), :) = [];

true_seq_ratings = seq_tbl{:, clean_vars};
seq_tbl.sequence = true_seq_ratings;
% confirm matches `face_mean` column
true_mean_ratings = nanmean( true_seq_ratings, 2 );
true_var_ratings = nanvar( true_seq_ratings, [], 2 );
true_seq_lens = seq_tbl.face_number;

est_diff = seq_tbl.estimation - true_mean_ratings;

seq_identifiers = [ seq_tbl.sequence, double(char(seq_tbl.faceIdentity)) ];
seq_identifiers(isnan(seq_identifiers)) = 0;
[~, ~, seq_ids] = unique( seq_identifiers, 'rows' );

seq_tbl.sequence_id = seq_ids;

%%  match up images

% //randomly choose from negative and face_tive emotion
% positive = 50, negative = 100
% base_emotion = getRandomElement([50, 100]); 

% //singleFace = getRandomInt(1, 50);
% 

imgs = shared_utils.io.find( im_p, '.jpg' );
img_names = shared_utils.io.filenames( imgs );
img_nums = cellfun( @(x) str2double(x(2:end)), img_names );

prefixes = { 'A', 'B', 'C', 'D' };
matches = cellfun( @(x) startsWith(img_names, x), prefixes, 'un', 0 );
ok_ims = or_many( matches{:} );

face_idents = seq_tbl.faceIdentity;
seq_ratings = seq_tbl.sequence;

valence_offsets = ones( size(seq_ratings) ) * 100;
valence_offsets(strcmp(seq_tbl.valence, 'positive'), :) = 50;

expect_nums = seq_ratings + valence_offsets;

im_files = strings( size(seq_ratings) );
for i = 1:size(seq_tbl, 1)
  ok_seq = ~isnan( expect_nums(i, :) );
  im_files(i, ok_seq) = arrayfun( ...
    @(x) sprintf('%s%d', face_idents{i}, x), expect_nums(i, ok_seq), 'un', 0 );
end

non_matched = ~ismember( im_files, img_names );
assert( isequal(non_matched, im_files == "") );

seq_tbl.image_files = im_files;

valences = repmat( string(seq_tbl.valence), 1, size(im_files, 2) );
im_info = table( im_files(:), seq_ratings(:), valences(:) ...
  , 'va', {'identifier', 'rating', 'valence'} );
im_info(isnan(im_info.rating), :) = [];
im_info = unique( im_info );
im_info.subject = extractBefore( im_info.identifier, 2 );
assert( numel(unique(im_info.identifier)) == numel(im_info.identifier) );

im_info_struct = struct();
for i = 1:numel(im_info.Properties.VariableNames)
  vn = im_info.Properties.VariableNames{i};
  tbl_var = im_info.(vn);
  if ( ~isnumeric(tbl_var) && ~islogical(tbl_var) )
    tbl_var = cellstr( tbl_var );
  end
  im_info_struct.(vn) = tbl_var;
end

%%  save output

if ( 1 )
  save( fullfile(data_root, 'study1_sequences.mat'), 'seq_tbl' );
  save( fullfile(data_root, 'image_info.mat'), 'im_info' );
  save( fullfile(data_root, 'image_info_struct.mat'), 'im_info_struct' );
end

%%  histogram of sequence means

figure(1); clf;
hist( true_mean_ratings, 1000 );

%%  fig 1c

do_save = true;
% prctile_mean_rating = 50;
prctile_mean_rating = [];
is_lt = false;

if ( 0 )
  % first average trials within subject
  [I, C] = findeach( seq_tbl, {'participant_id', 'valence', 'face_number'} );
  C.mean_diffs = cellfun( @(x) mean(est_diff(x)), I );
else
  % matching the analysis script, use N as trial
  C = seq_tbl;
  C.mean_diffs = est_diff;
end

mask = rowmask( C );
if ( ~isempty(prctile_mean_rating) )
  op = ternary( is_lt, @lt, @gt );
  mask = op( true_mean_ratings, prctile(true_mean_ratings, prctile_mean_rating) );
end

%  p(see image with rating above x)
thresh = prctile( true_mean_ratings, 90 );
above_thresh = seq_tbl.sequence >= thresh;
ps = sum( above_thresh, 2 ) ./ seq_tbl.face_number;

if ( 0 )
  plt = ps;
  plt = true_mean_ratings;
else
  plt = C.mean_diffs;
end

[I, id, L] = rowsets( 3, C, {}, {'face_number'}, {'valence'} ...
  , 'to_string', true, 'mask', mask );
ord = plots.orderby( L, arrayfun(@num2str, 1:12, 'un', 0) );
ord = plots.orderby( L, 'positive', ord );
[I, id, L] = rowref_many( ord, I, id, L );

figure(2); clf;
[axs, hs] = plots.simplest_barsets( plt, I, id, L ...
  , 'error_func', @plotlabeled.nansem ...
  , 'as_line_plot', true ...
);
ylabel( axs(1), 'Estimated emotion rating - actual emotion' );
title( axs(1), 'Human subject behavior' );
xlabel( axs(1), 'Face number' );

for i = 1:numel(hs)
  set( hs{i}, 'linewidth', 2 );
end

if ( isempty(prctile_mean_rating) )
  ylim( axs(1), [-6, 6] );
else
  ylim( axs(1), [-24, 24] );
end

if ( 1 )
  assert( isnumeric(mask) );
  mask = intersect( mask, find(strcmp(C.valence, 'negative')) );
  mean_est = mean( plt(mask) );  
  hold( axs, 'on' );
  h1 = shared_utils.plot.add_horizontal_lines( axs, 0 );
  h2 = shared_utils.plot.add_horizontal_lines( axs, mean_est );
end

set( gcf, 'Renderer', 'painters' );
style_line_plots( axs );

if ( do_save )
  save_p = fullfile( fileparts(data_root), 'plots', dsp3.datedir, 'behavior' );
  fname = 'sequence_estimate';
  if ( ~isempty(prctile_mean_rating) )
    lt_str = ternary( is_lt, 'less_than', 'greater_than' );
    fname = sprintf( '%s_%s_%d_prctile', fname, lt_str, prctile_mean_rating );
  end
  shared_utils.io.require_dir( save_p );
  shared_utils.plot.save_fig( gcf, fullfile(save_p, fname), {'png', 'fig', 'epsc'}, false );
  exportgraphics( gcf, fullfile(save_p, sprintf('%s.eps', fname)), 'ContentType', 'vector' );
end

%%  slopes and means as f(prctile)

do_save = true;

C = seq_tbl;
C.mean_diffs = est_diff;

prctiles = [ 50, 75, 90, 95, 100 ];
ops = { @lt, @gt };

all_mdls = table();
for idx = 1:numel(ops)
  for i = 1:numel(prctiles)
    prctile_mean_rating = prctiles(i);
    op = ops{idx};
    mask = op( true_mean_ratings, prctile(true_mean_ratings, prctile_mean_rating) );
    if ( sum(mask) == 0 ), continue; end    
    mdl_each = { 'valence' };
    [mdl_I, mdls] = findeach( seq_tbl, mdl_each, mask );
    xs = cellfun( @(x) C.face_number(x), mdl_I, 'un', 0 );
    ys = cellfun( @(x) C.mean_diffs(x), mdl_I, 'un', 0 );
    lms = cellfun( @(x, y) fitlm(x, y), xs, ys, 'un', 0 );
    betas = cellfun( @(x) x.Coefficients.Estimate(2), lms );
    [r, p] = cellfun( @(x, y) corr(x, y, 'rows', 'complete', 'type', 'Spearman'), xs, ys );
    mus = cellfun( @nanmean, ys );
    mdls.beta = betas(:);
    mdls.mean = mus(:);
    [mdls.corr_r, mdls.corr_p] = deal( r(:), p(:) );
    mdls.op = repmat( string(func2str(op)), size(mdls, 1), 1 );
    mdls.prctile = repmat( prctile_mean_rating, numel(betas), 1 );
    all_mdls = [ all_mdls; mdls ];
  end
end

mdls = all_mdls;

figure(1); clf;
var_name = 'mean';
[I, id, C] = rowsets( 3, mdls, 'op', 'prctile', 'valence', 'to_string', 1 );
[~, ord] = sort( double(string(C(:, 2))) );
[I, id, C] = rowref_many( ord, I, id, C );
axs = plots.simplest_barsets( mdls.(var_name), I, id, C );
shared_utils.plot.match_ylims( axs );
ylabel( axs(1), var_name );

if ( contains(var_name, 'corr_r') )
  ylim( axs, [-0.6, 0.6] );
end

if ( do_save )
  save_p = fullfile( fileparts(data_root), 'plots', dsp3.datedir, 'behavior' );
  fname = sprintf( 'sequence_metric_%s_as_f_seq_mean', var_name );
  shared_utils.io.require_dir( save_p );
  shared_utils.plot.save_fig( gcf, fullfile(save_p, fname), {'png', 'fig'}, false );
end

%%

thresh = prctile( true_mean_ratings, 90 );
[I, C] = findeach( seq_tbl, {'valence', 'face_number'} );
mus = cellfun( @(x) mean((...
  sum(seq_tbl.sequence(x, :) >= thresh, 2) ./ seq_tbl.face_number(x)) > 0), I );
C.mus = mus;
[~, ord] = sort( C.face_number );
C = C(ord, :);

figure(1); clf;
[I, id, L] = rowsets( 3, C, {}, {'face_number'}, {'valence'}, 'to_string', true );
ord = plots.orderby( L, arrayfun(@num2str, 1:12, 'un', 0) );
ord = plots.orderby( L, 'positive', ord );
[I, id, L] = rowref_many( ord, I, id, L );
axs = plots.simplest_barsets( C.mus, I, id, L ...
  , 'error_func', @plotlabeled.nansem ...
  , 'as_line_plot', true ...
);

%%  show images

rating_thresh = 45;
[posi, posj] = find( seq_ratings > rating_thresh & string(seq_tbl.valence) == 'positive' );
[negi, negj] = find( seq_ratings > rating_thresh & string(seq_tbl.valence) == 'negative' );

pos_ind = randi( numel(posi) );
[posi, posj] = deal( posi(pos_ind), posj(pos_ind) );

neg_ind = randi( numel(negi) );
[negi, negj] = deal( negi(neg_ind), negj(neg_ind) );

figure(1); clf;
subplot( 1, 2, 1 );
imshow( imread(fullfile(im_p, sprintf('%s.jpg', im_files(posi, posj)))) );
title( 'positive' );

subplot( 1, 2, 2 );
imshow( imread(fullfile(im_p, sprintf('%s.jpg', im_files(negi, negj)))) );
title( 'negative' );

