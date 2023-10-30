data_root = 'D:\data\changlab\ilker_collab';
[metrics, train_metrics, seq_tbl, im_info] = load_data( data_root );

%%  training trajectory

mask = train_metrics.model == 'sc_eval';
[I, C] = findeach( train_metrics, {'layer', 'model'}, mask );
batch_errs = cate1( eachcell(@(x) train_metrics.data{x}.batch_error, I) );

figure(1); clf;
[I, id, C] = rowsets( 2, C, 'model', 'layer', 'to_string', true );
C = plots.strip_underscore( C );
[PI, PL] = plots.nest2( id, I, C );
[axs, hs] = plots.simplest_linesets( 1:size(batch_errs, 2), batch_errs, PI, PL ...
  , 'error_func', @(x) nan(size(x)));

xlabel( axs, 'iteration' );
ylabel( axs, 'batch loss (mean reconstruction error)' );

shared_utils.plot.match_ylims( axs );

%%  valid error

num_mean = 20;
target_model = 'sc_eval';

% num_mean = 0;
% target_model = 'pca_eval';

src_tbl = train_metrics(train_metrics.model == target_model, :);
% src_tbl = train_metrics;
train_err = cellfun( @(x) mean(double(x.batch_error(end-num_mean:end))), src_tbl.data );
valid_err = cellfun( @(x) mean(double(x.batch_error)), src_tbl.valid_data );

names = repelem( ["train"; "valid"], rows(src_tbl) );
err_tbl = [ table([train_err; valid_err], 'va', {'error'}), [src_tbl; src_tbl] ];
err_tbl.split = names;

figure(1); clf;
[I, id, C] = rowsets( 3, err_tbl, 'model', 'layer', 'split', 'to_string', true );
C = plots.strip_underscore( C );
axs = plots.simplest_barsets( err_tbl.error, I, id, C );
set( axs, 'xticklabelrotation', 10 );
ylabel( axs(1), 'reconstruction error' );

%%  correlation b/w error and emotionality

per_subj = false;
do_save = false;
rem_outliers = false;
match_lims = false;

% err_metric_name = 'distinctiveness';
% err_metric_name = 'error';
err_metric_name = 'image_recon_error_resnet_image_embedding_layer4';
err_metric = metrics.(err_metric_name);

corr_each = { 'layer', 'valence', 'model' };
if ( per_subj ), corr_each{end+1} = 'subject'; end

I = findeach( metrics, corr_each );
lt_thresh = false( size(err_metric) );
for i = 1:numel(I)
  lt_thresh(I{i}) = err_metric(I{i}) < prctile(err_metric(I{i}), 95);
end

fig_mask = ternary( rem_outliers, lt_thresh, true(size(err_metric)) );

if ( contains(err_metric_name, 'image_recon_error') )
  % only one "model" and "layer" for recon error
  un_I = findeach( metrics, {'model', 'layer'} );
  fig_mask = intersect( find(fig_mask), un_I{1} );
end

fig_each = { 'layer' };
if ( per_subj ), fig_each{end+1} = 'model'; end
[f_I, f_C] = findeach( metrics, fig_each, fig_mask );

[lims_I, lims_C] = findeach( metrics, 'model', fig_mask );
[mins, maxs] = cellfun( @(x) deal(min(err_metric(x)), max(err_metric(x))), lims_I );

for idx = 1:numel(f_I)

mask = f_I{idx};

[I, C] = findeach( metrics, corr_each, mask );
[C.r, C.p] = cellfun( ...
  @(x) corr(err_metric(x), metrics.rating(x), 'type', 'Spearman'), I );

figure(1); clf;
axs = plots.panels( numel(I) );
for i = 1:numel(axs)
  ind = I{i};
  lm = fitlm( metrics(ind, :), sprintf('rating ~ %s', err_metric_name) );
  plot( axs(i), lm );
  
  if ( 1 )
    hold( axs(i), 'on' );
    gscatter( axs(i), err_metric(ind), metrics.rating(ind), metrics.subject(ind) )
  end

  title_str = plots.strip_underscore(strjoin(C{i, corr_each}, " | "));
  title_str = compose( "%s (r = %0.3f, p = %0.3f)", title_str, C.r(i), C.p(i) );
  title( axs(i), title_str );
end

if ( 1 )
  if ( match_lims )
    [~, lims_ind] = ismember( C.model, lims_C.model );
    for i = 1:numel(lims_ind)
      li = lims_ind(i);
      xlim( axs(i), [mins(li), maxs(li)] );
    end
  else
    mi = findeach( C, 'model' );
    for i = 1:numel(mi)
      shared_utils.plot.match_xlims( axs(mi{i}) );
    end
  end
  ylim( axs, [-10, 60] );
%   shared_utils.plot.match_ylims( axs );
end

if ( do_save )
  shared_utils.plot.fullscreen( gcf );
  subdir = 'corr_emotionality';
  subdir = sprintf( '%s%s', subdir, ternary(per_subj, '_per_subj', '') );
  subdir = sprintf( '%s%s', subdir, ternary(rem_outliers, '_rem_outliers', '') );
  save_p = fullfile( data_root, 'plots', dsp3.datedir, subdir, err_metric_name );
  shared_utils.io.require_dir( save_p );
  fname = char( plots.cellstr_join({f_C(idx, :)}) );
  fname = strrep( fname, ' | ', '_' );
  shared_utils.plot.save_fig( gcf, fullfile(save_p, fname), {'png', 'fig'}, false );
end

end

%%  lms

per_subj = false;
rem_outliers = true;

err_metric_name = 'distinctiveness';
% err_metric_name = 'error';
err_metric = metrics.(err_metric_name);

corr_each = { 'layer', 'valence', 'model' };
if ( per_subj ), corr_each{end+1} = 'subject'; end

I = findeach( metrics, corr_each );

lt_thresh = false( size(err_metric) );
for i = 1:numel(I)
  lt_thresh(I{i}) = err_metric(I{i}) < prctile(err_metric(I{i}), 95);
end

lm_mask = ternary( rem_outliers, lt_thresh, true(size(err_metric)) );
[I, C] = findeach( metrics, corr_each, lm_mask );
lms = eachcell( @(x) fitlm(metrics(x, :), sprintf('rating ~ %s', err_metric_name)), I );
C.slope = cellfun( @(x) x.Coefficients.Estimate(2), lms );
C.r2 = cellfun( @(x) x.Rsquared.Ordinary, lms );
C.sig = cellfun( @(x) x.Coefficients.pValue(2) < 0.05, lms );

summary_var = 'slope';
[I, id, L] = rowsets( 3, C, {'model'}, {'layer'}, {'valence'}, 'to_string', true );
L = plots.strip_underscore( L );
figure(1); clf;
[axs, hs, xs] = plots.simplest_barsets( double(C.(summary_var)), I, id, L );
% shared_utils.plot.match_ylims( axs );
set( axs, 'xticklabelrotation', 12 );
ylabel( axs(1), summary_var );

[PI, PL] = plots.nest3( id, I, L );
for i = 1:numel(PI)
  lims = get( axs(i), 'ylim' );
  is_sig = cellfun( @(x) C.sig(x), PI{i} );
  mu = cellfun( @(x) C.(summary_var)(x), PI{i} );
  scatter( axs(i), xs{i}(is_sig), mu(is_sig) + diff(lims) * 0.025, 'k*' );
end

%%  seqs

% err_var_name = 'distinctiveness';
err_var_name = 'error';
err_var_name = 'image_recon_error';
% err_var_name = 'image_recon_error_resnet_image_embedding_layer2';

mask = rowmask( metrics );
if ( contains(err_var_name, 'image_recon') )
  % only one "layer" / "model" per image
  mask = ref( findeach(metrics, {'layer', 'model'}), '{}', 1 );
end

seq_err_tbl = sequence_estimation( seq_tbl, im_info, metrics(mask, :), err_var_name );

%%  seqs, multiple metrics

err_var_names = {...
    'distinctiveness', 'error', 'image_recon_error' ...
  , 'image_recon_error_resnet_image_embedding_layer2' ...
  , 'image_recon_error_resnet_image_embedding_layer3' ...
  , 'image_recon_error_resnet_image_embedding_layer4' ...
};

seq_err_tbls = table();
for i = 1:numel(err_var_names)
  fprintf( '\n %d of %d', i, numel(err_var_names) );
  mask = rowmask( metrics );
  is_image_metric = false;
  if ( contains(err_var_names{i}, 'image_recon') )
    % only one "layer" / "model" per image
    mask = ref( findeach(metrics, {'layer', 'model'}), '{}', 1 );
    is_image_metric = true;
  end
  err_tbl = sequence_estimation( seq_tbl, im_info, metrics(mask, :), err_var_names{i} );
  err_tbl.error_metric = repmat( string(err_var_names{i}), rows(err_tbl), 1 );
  if ( is_image_metric )
    err_tbl.model(:) = "image";
    err_tbl.layer(:) = "image";
  end
  seq_err_tbls = [ seq_err_tbls; err_tbl ];
end

%%  corr estimates trial by trial

[I, C] = findeach( seq_err_tbl, {'participant_id', 'model', 'layer', 'valence'} );
[C.r, C.p] = cellfun( ...
  @(x) corr(seq_err_tbl.model_estimate_diff(x), seq_err_tbl.estimation_diff(x), 'type', 'Spearman'), I );
[I, L] = findeach( C, {'model', 'layer', 'valence'} );
p_sig = cellfun( @(x) pnz(C.p(x) < 0.05), I );

figure(1); clf;

if ( 1 )
  [I, id, C] = rowsets( 3, L, 'model', 'layer', 'valence', 'to_string', true );
  C = plots.strip_underscore( C );
  axs = plots.simplest_barsets( p_sig, I, id, C );
  ylim( axs, [0, 0.14] );
else
  axs = plots.panels( numel(I) );
  for i = 1:numel(axs)
    ind = I{i};
    hist( axs(i), C.p(ind) );
    title( axs(i), plots.strip_underscore(strjoin(L{i, :}), ' | ') );
  end
end

%%  scatter corr estimates trial by trial

do_save = false;

x_var = 'model_estimate_diff';
y_var = 'estimation_diff';

mask = seq_err_tbl.model == 'sc_eval';

[fig_I, f_C] = findeach( seq_err_tbl, 'layer', mask );

figure(1); clf; 
for idx = 1:numel(fig_I)
  
mask = fig_I{idx};

[I, C] = findeach( seq_err_tbl, {'model', 'layer', 'valence'}, mask );
axs = plots.panels( numel(I) );
for i = 1:numel(axs)
  err_tbl = seq_err_tbl(I{i}, :);
  [x, y] = deal( err_tbl.(x_var), err_tbl.(y_var) );
  [r, p] = corr( x, y, 'rows', 'complete', 'type', 'spearman' );
  lm = fitlm( x, y );
  plot( axs(i), lm );
%   hold( axs(i), 'on' );
%   if ( 0 ), gscatter( axs(i), x, y, err_tbl.faceIdentity ); end
  title_str = plots.strip_underscore( strjoin(C{i, :}, ' | ') );
  title( axs(i), sprintf( '(%s) r = %0.3f, p = %0.3f', title_str, r, p) );
  xlabel( axs(i), x_var );
  ylabel( axs(i), sprintf( '%s (%s)', y_var, err_var_name) );
end

shared_utils.plot.match_xlims( axs );
shared_utils.plot.match_ylims( axs );

if ( do_save )
  shared_utils.plot.fullscreen( gcf );
  subdir = 'corr_trial_by_trial_estimation_diff';
  save_p = fullfile( data_root, 'plots', dsp3.datedir, subdir, err_var_name );
  shared_utils.io.require_dir( save_p );
  fname = strrep( char(plots.cellstr_join({f_C(idx, :)})), ' | ', '_' );
  shared_utils.plot.save_fig( gcf, fullfile(save_p, fname), {'png', 'fig'}, false );
end

end

%%  slopes of 1c

slope_each = { 'model', 'layer', 'valence' };
[I, C] = findeach( seq_err_tbl, slope_each );
xs = eachcell( @(x) seq_err_tbl.face_number(x), I );
ys = eachcell( @(x) seq_err_tbl.model_estimate_diff(x), I );
lms = eachcell( @(x, y) fitlm(x, y), xs, ys );
C.slope = cellfun( @(x) x.Coefficients.Estimate(2), lms );
C.p = cellfun( @(x) x.Coefficients.pValue(2), lms );
C.r2 = cellfun( @(x) x.Rsquared.Ordinary, lms );
C.sig = C.p < 0.05;

var_name = 'slope';
[I, id, L] = rowsets( 3, C, {'model'}, {'layer'}, {'valence'}, 'to_string', true );
L = plots.strip_underscore( L );
figure(1); clf;
axs = plots.simplest_barsets( C.(var_name), I, id, L );
% shared_utils.plot.match_ylims( axs );
set( axs, 'xticklabelrotation', 12 );
ylabel( axs(1), sprintf('%s of (face number vs. estimate diff)', var_name) );

%%  plot fig 1c

err_var_name = 'error';
seq_err_tbl = seq_err_tbls(seq_err_tbls.error_metric == err_var_name, :);

do_save = true;
match_lims = false;

if ( contains(err_var_name, 'image_recon') )
  % only one "layer" / "model" per image
  I = findeach( seq_err_tbl, {'model', 'layer'} );
  mask = I{1};
else
  mask = seq_err_tbl.layer == 'resnet_output';
  mask = seq_err_tbl.model == 'pca_eval';
end

% error( 'xx' );
seq_mus = nanmean( seq_err_tbl.sequence, 2 );
if ( 0 )
  mask = intersect( mask, find(seq_mus < prctile(seq_mus, 100)) );
end

[I, id, L] = rowsets( 3, seq_err_tbl ...
  , {'model', 'layer'}, {'face_number'}, {'valence'} ...
  , 'to_string', true, 'mask', mask );
L = plots.strip_underscore( L );
ord = plots.orderby( L, {'layer1', 'layer2', 'layer3', 'layer4', 'output'} );
ord = plots.orderby( L(:, 2), arrayfun(@num2str, 1:12, 'un', 0), ord );
ord = plots.orderby( L, 'positive', ord );
[I, id, L] = rowref_many( ord, I, id, L );

figure(1); clf;
axs = plots.simplest_barsets( seq_err_tbl.model_estimate_diff, I, id, L ...
  , 'error_func', @plotlabeled.nansem ...
  , 'as_line_plot', true ...
);

if ( match_lims )
  shared_utils.plot.match_ylims( axs );
end

xlabel( axs, 'Face number' );
ylabel( axs, sprintf('Delta %s', err_var_name) );

if ( 0 )
  max_lim = max( get(axs(1), 'ylim') ) * 1.125;
  ylim( axs(1), [-max_lim, max_lim] );
end

if ( 0 )
  ylim( axs(1), [-0.65, 0.65] );
end

if ( do_save )
  fname = 'sequence';
  save_p = fullfile( data_root, 'plots', dsp3.datedir, 'sequence', err_var_name );
  shared_utils.io.require_dir( save_p );
  shared_utils.plot.fullscreen( gcf );
  shared_utils.plot.save_fig( gcf, fullfile(save_p, fname), {'png', 'fig'}, false );
end

%%  slopes and means as f(prctile)

prctiles = [ 50, 75, 90, 95, 100 ];
[I, C] = findeach( seq_err_tbls, {'model', 'layer', 'error_metric', 'valence'} );
all_mdls = table();
for i = 1:numel(I)
  mdls = fit_sequence_lms_by_prctile( seq_err_tbls(I{i}, :), prctiles );
  mdls = [ mdls, repmat(C(i, :), rows(mdls), 1) ];
  all_mdls = [ all_mdls; mdls ];
end
mdls = all_mdls;

%%  slopes and means

[I, C] = findeach( seq_err_tbls, {'model', 'layer', 'error_metric', 'valence'} );
all_mdls = table();
has_behav_pos = false;
has_behav_neg = false;

for i = 1:numel(I)
  st = seq_err_tbls(I{i}, :);
  xs = st.face_number;
  ys = st.model_estimate_diff;
  mdls = fit_sequence_lm( xs, ys );
  mdls = [ mdls, repmat(C(i, :), rows(mdls), 1) ];
  all_mdls = [ all_mdls; mdls ];
  
  is_pos = false;
  if ( string(C.valence{i}) == "positive" )
    need_fit_behav = ~has_behav_pos;
    is_pos = true;
  else
    need_fit_behav = ~has_behav_neg;
  end
  
  if ( need_fit_behav )
    % add behavior
    mdls = fit_sequence_lm( xs, st.estimation_diff );
    mdls = [ mdls, repmat(C(i, :), rows(mdls), 1) ];
    [mdls.model, mdls.layer, mdls.error_metric] = deal( "behavior" );
    all_mdls = [ all_mdls; mdls ];
    if ( is_pos ), has_behav_pos = true; end
    if ( ~is_pos), has_behav_neg = true; end
  end
end
mdls = all_mdls;

%%  plot slopes and means

targ_layers = ["resnet_output", "resnet_layer2", "ReconNetWrapper_output"];
mask = ...
  (mdls.model == 'sc_eval' & ismember(mdls.layer, targ_layers)) | ...
  (mdls.model == 'behavior') | ...
  mdls.model == 'image';

[I, id, C] = rowsets( 3, mdls ...
  , {'model'}, {'valence'}, {'layer', 'error_metric'} ...
  , 'mask', mask, 'to_string', true ...
);
C = plots.strip_underscore( C );
var_name = 'corr_r';
figure(1); clf;
axs = plots.simplest_barsets( mdls.(var_name), I, id, C );
shared_utils.plot.match_ylims( axs );
set( axs, 'xticklabelrotation', 15 );
ylabel( axs(1), var_name );

%%  plot slopes and means as f(prctile)

do_save = true;
[fig_I, fig_C] = findeach( mdls, {'model', 'layer', 'error_metric'} );

for i = 1:numel(fig_I)
fi = fig_I{i};
figure(1); clf;
var_name = 'corr_r';
[I, id, C] = rowsets( 3, mdls ...
  , {'op', 'model', 'layer', 'error_metric'}, 'prctile', 'valence' ...
  , 'to_string', 1, 'mask', fi );
C = plots.strip_underscore( C );
[~, ord] = sort( double(string(C(:, 2))) );
[I, id, C] = rowref_many( ord, I, id, C );
axs = plots.simplest_barsets( mdls.(var_name), I, id, C );
shared_utils.plot.match_ylims( axs );
ylabel( axs(1), var_name );

if ( contains(var_name, 'corr_r') )
  ylim( axs, [-0.6, 0.6] );
end

if ( do_save )
  subdir = strjoin( fig_C{i, {'error_metric'}}, '__' );
  mdl_info = strjoin( fig_C{i, {'model', 'layer'}}, '_' );
  save_p = fullfile( data_root, 'plots', dsp3.datedir, 'sequence', char(subdir), var_name );
  fname = sprintf( '%s_sequence_metric_as_f_seq_mean', mdl_info );
  shared_utils.io.require_dir( save_p );
  shared_utils.plot.save_fig( gcf, fullfile(save_p, fname), {'png', 'fig'}, false );
end

end

%%

function mdl = fit_sequence_lm(xs, ys)

lm = fitlm( xs, ys );
beta = lm.Coefficients.Estimate(2);
mu = nanmean( ys );
[r, p] = corr( xs(:), ys(:), 'rows', 'complete', 'type', 'Spearman' );

mdl = table();
mdl.mdl = { lm };
mdl.beta = beta;
mdl.mean = mu;
mdl.r2 = lm.Rsquared.Ordinary;
[mdl.corr_r, mdl.corr_p] = deal( r, p );

end

function mdls = fit_sequence_lms_by_prctile(seq_err_tbl, prctiles)

true_mean_ratings = nanmean( seq_err_tbl.sequence, 2 );

ops = { @lt, @gt };

all_mdls = table();
for idx = 1:numel(ops)
  for i = 1:numel(prctiles)
    prctile_mean_rating = prctiles(i);
    op = ops{idx};
    mask = op( true_mean_ratings, prctile(true_mean_ratings, prctile_mean_rating) );
    if ( sum(mask) == 0 ), continue; end
    
    xs = seq_err_tbl.face_number(mask);
    ys = seq_err_tbl.model_estimate_diff(mask);
    mdls = fit_sequence_lm( xs, ys );
    
    mdls.op = string( func2str(op) );
    mdls.prctile = prctile_mean_rating;
    
    all_mdls = [ all_mdls; mdls ];
  end
end

mdls = all_mdls;

end

function seq_err_tbl = sequence_estimation(seq_tbl, im_info, metrics, err_var_name)

err_var = metrics.(err_var_name);

% find the face with the mean emotionality of the sequence
est_rating = max( 1, min(max(im_info.rating), round(nanmean(seq_tbl.sequence, 2))) );
% est_rating = max( 1, min(max(im_info.rating), floor(nanmean(seq_tbl.sequence, 2)) - 1) );
est_ident = strings( size(est_rating) );

for i = 1:numel(est_rating)
  match_im = im_info.rating == est_rating(i) & ...
    im_info.valence == string(seq_tbl.valence(i)) & ...
    im_info.subject == string(seq_tbl.faceIdentity(i));
  assert( sum(match_im) == 1 );
  est_ident(i) = im_info.identifier(match_im);
end

search = unique( metrics(:, {'model', 'layer'}) );

seq_err_tbl = cell( rows(search), 1 );
parfor i = 1:rows(search)
  fprintf( '\n %d of %d', i, rows(search) );
  
  do_search = @(ident) find( ...
      metrics.identifier == ident & ...
      metrics.model == search.model(i) & ...
      metrics.layer == search.layer(i) ...
    );
  
  metric_diffs = nan( size(est_ident) );
  for j = 1:numel(est_ident)
    ims = seq_tbl.image_files(j, :);
    % find error of "true" mean emotion
    match_est = do_search( est_ident(j) );
    assert( numel(match_est) == 1 );
    % find mean error of sequence of presented images
    match_others = arrayfun( do_search, ims(ims ~= "") );
    match_errs = err_var(match_others);
    % difference of mean error from "true"
    metric_diffs(j) = nanmean( match_errs ) - err_var(match_est);
  end
  
  rep_t = repmat( search(i, :), rows(est_ident), 1 );
  t = [ table(metric_diffs, 'va', {'model_estimate_diff'}), rep_t, seq_tbl ];
  seq_err_tbl{i} = t;
end

seq_err_tbl = vertcat( seq_err_tbl{:} );

end

function va = setdiff_var_names(a, b)
va = setdiff( a.Properties.VariableNames, b.Properties.VariableNames );
end

function [metrics, train_metrics, seq_tbl, im_info] = load_data(data_root)

seq_tbl = shared_utils.io.fload( fullfile(data_root, 'goldenberg_faces/study1_sequences.mat') );
im_info = shared_utils.io.fload( fullfile(data_root, 'goldenberg_faces/image_info.mat') );

im_recon_err = shared_utils.io.fload( ...
  fullfile(data_root, 'd3dfr_face_recon/recon_error/goldenberg_faces_masked.mat') );
im_embed_recon_err = shared_utils.io.fload( ...
  fullfile(data_root, 'd3dfr_face_recon/recon_error/goldenberg_faces_resnet_image_embedding.mat') );
[~, ord] = ismember( im_embed_recon_err.identifier, im_recon_err.identifier );
keep_va = setdiff_var_names( im_embed_recon_err, im_recon_err );
im_recon_err = [ im_recon_err, im_embed_recon_err(ord, keep_va) ];
im_recon_err.Properties.VariableNames = strrep(...
  im_recon_err.Properties.VariableNames, 'recon_error', 'image_recon_error' );

layer_names = ["ReconNetWrapper_output", "resnet_output", compose("resnet_layer%d", 2:4)];
ds_name = "d3dfr/valid_var_subsample";  % original

ds_name = "d3dfr/valid_expression_balanced_var_subsample_1000";

% ds_name = "d3dfr/valid";
% layer_names = ["ReconNetWrapper_output", "resnet_output"];

eval_dir_names = ["pca_eval", "sc_eval"];

all_metrics = table();
all_train_metrics = table();

for i = 1:numel(eval_dir_names)

eval_dirs = fullfile( data_root, eval_dir_names(i), ds_name, layer_names );
act_files = fullfile( data_root, "activations", ds_name, compose("%s.h5", layer_names) );
distinct_files = fullfile( data_root, "distinctiveness", ds_name, compose("%s.mat", layer_names) );

eval_cps = arrayfun( @load, fullfile(eval_dirs, "cp.mat"), 'un', 0 );
eval_errs = cate1( cellfun(@(x) x.error(:), eval_cps, 'un', 0) );

to_str = @(x) columnize(deblank(string(x)));
idents = cate1(arrayfun(@(x) to_str(h5read(char(x), '/identifiers')), act_files, 'un', 0));
layers = arrayfun( @(x) to_str(h5read(char(x), '/layers')), act_files, 'un', 0 );
layers = cate1( arrayfun(@(i) repmat(layer_names(i), size(layers{i})), 1:numel(layers), 'un', 0) );

distincts = cate1( cellfun(@shared_utils.io.fload, distinct_files, 'un', 0) );

metrics = table( ...
  eval_errs, distincts, idents, layers, 'va' ...
  , {'error', 'distinctiveness', 'identifier', 'layer'} );
% only keep images that were shown during task
metrics(~ismember(metrics.identifier, im_info.identifier), :) = [];

% add image info to metrics
keep_vars = setdiff( im_info.Properties.VariableNames, metrics.Properties.VariableNames );
[~, ind] = ismember( metrics.identifier, im_info.identifier );
metrics = [ metrics, im_info(ind, keep_vars) ];
metrics.model = repmat( eval_dir_names(i), rows(metrics), 1 );

% add recon error to metrics
[~, ind] = ismember( metrics.identifier, im_recon_err.identifier );
keep_va = setdiff( im_recon_err.Properties.VariableNames, 'identifier' );
metrics = [ metrics, im_recon_err(ind, keep_va) ];

all_metrics = [ all_metrics; metrics ];

% training metrics
train_cp_dirs = strrep( strrep(eval_dirs, 'valid', 'train'), '_eval', '_checkpoints' );
train_cps = arrayfun( @load, fullfile(train_cp_dirs, "cp.mat"), 'un', 0 );
[~, layers] = arrayfun( @fileparts, train_cp_dirs );

train_metrics = table( ...
    train_cps(:) ...
  , eval_cps(:) ...
  , repmat(eval_dir_names(i), numel(train_cps), 1) ...
  , layers(:), 'va', {'data', 'valid_data', 'model', 'layer'} ...
);

all_train_metrics = [ all_train_metrics; train_metrics ];

end

metrics = all_metrics;
train_metrics = all_train_metrics;
train_metrics.batch_error = cellfun( @(x) x.batch_error(:)', train_metrics.data, 'un', 0 );

end
