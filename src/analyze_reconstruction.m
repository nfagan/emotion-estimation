data_root = 'D:\data\changlab\ilker_collab';
[metrics, train_metrics, seq_tbl, im_info] = load_data( data_root );
rb_ensemble_code_est = load( fullfile(data_root, 'goldenberg_faces/study1_robinson_brady_ensemble_code.mat') );
rb_ensemble_code_est = rb_ensemble_code_est.tcc_ests;

%%  check for correlation b/w phase-scrambled recon error and true image error, by valence

do_save = false;

[I, C] = findeach( metrics, {'layer', 'valence', 'src_model', 'model'} ...
  , metrics.src_model == 'arcface_recog' | ...
  metrics.src_model == 'resnet_image_embedding' ...
);
scram_ind = find( contains(metrics.dataset_name, 'valid_expression_balanced_phase_scrambled_var_subsample_1000') );
ctrl_ind = find( contains(metrics.dataset_name, 'valid_expression_balanced_var_subsample_1000') );

scram_metrics = { 'error' };
% ctrl_metrics = { 'error', 'rating' };
ctrl_metrics = { 'rating' };

CC = table();
for ii = 1:numel(scram_metrics)
  for jj = 1:numel(ctrl_metrics)
    
    scram_metric = scram_metrics{ii};
    ctrl_metric = ctrl_metrics{jj};
    
    C.r = nan( rows(C), 1 );
    C.p = nan( rows(C), 1 );
    for i = 1:numel(I)
      iscram = intersect( I{i}, scram_ind );
      ictrl = intersect( I{i}, ctrl_ind );
      if ( isempty(iscram) || isempty(ictrl) )
        continue;
      end

      [~, ord] = ismember( metrics.identifier(iscram), metrics.identifier(ictrl) );
      assert( isequal(ord(:)', 1:numel(iscram)) );
      [C.r(i), C.p(i)] = corr( metrics.(scram_metric)(iscram), metrics.(ctrl_metric)(ictrl) );
      C.scram_metric = scram_metric + strings( rows(C), 1 );
      C.ctrl_metric = ctrl_metric + strings( rows(C), 1 );
    end
    
    CC = [CC; C];
  end
end

% add correlation between original recon error and rating
if ( 0 )
for i = 1:numel(I)
  ictrl = intersect( I{i}, ctrl_ind );
  if ( isempty(ictrl) )
    continue;
  end
  [C.r(i), C.p(i)] = corr( metrics.error(ictrl), metrics.rating(ictrl) );
  C.scram_metric = strings(rows(C), 1) + "none";
  C.ctrl_metric = strings(rows(C), 1) + "real_rating";
end
CC = [CC; C];
end

C = CC;

[I, id, L] = rowsets( 3, C ...
  , {'model', 'ctrl_metric'}, {'layer'}, {'valence', 'src_model'} ...
  , 'to_string', 1 );
ord = plots.orderby( L, {'positive'} );
ord = plots.orderby( L, {'arcface', 'resnet_image'}, ord );
L = strrep( L, 'error', 'reconstruction error' );
L = strrep( L, 'rating', 'emotion rating' );
[I, id, L] = rowref_many( ord, I, id, L );
L = plots.strip_underscore( L );
L = rename_models_for_plotting( L );
figure(1); clf;
% [axs, hs, xs, ip] = plots.simplest_barsets( C.r, I, id, L, 'panel_shape', [2, 2] );
[axs, hs, xs, ip] = plots.simplest_barsets( C.r, I, id, L );
shared_utils.plot.match_ylims( axs );
% ylabel( axs(1), sprintf('Correlation between\nreal %s and phase-scrambled %s', ctrl_metric, scram_metric) );
ylabel( axs(1), sprintf('R (correlation between\nreal and phase-scrambled metric)') );
set( axs, 'xticklabelrotation', 10 );
for i = 1:numel(ip)
  isig = C.p(cellfun(@identity, ip{i})) < 0.05;
  xsig = xs{i}(isig);
  ysig = C.r(cellfun(@identity, ip{i}));
  ysig = ysig(isig);
  scatter( axs(i), xsig, ysig + sign(ysig)*0.05, 'k*' );
end

if ( strcmp(ctrl_metric, 'rating') )
  ylim( axs, [-0.5, 0.5]*0.5 );
else
  ylim( axs, [-0.6, 0.6] );
end

% ylim( axs, [-0.6, 0.6] );
ylim( axs, [-1, 1] );

plots.onelegend( gcf );
style_bar_plots( axs, 'no_error_lines', 1 );

if ( do_save )
  fname = 'error_corr';
  save_p = fullfile( data_root, 'plots', dsp3.datedir, 'phase_scramble' );
  shared_utils.io.require_dir( save_p );
  shared_utils.plot.save_fig( gcf, fullfile(save_p, fname), {'png', 'fig'}, false );
  exportgraphics( gcf, fullfile(save_p, sprintf('%s.eps', fname)), 'ContentType', 'vector' );
end

%%  add image pca

im_ps = fullfile( string(data_root), "goldenberg_faces/images" ...
  , compose("%s.jpg", im_info.identifier) );
ims = arrayfun( @(x) double(imread(x))/255, im_ps, 'un', 0 );
im_vecs = cate1( cellfun(@(x) x(:)', ims, 'un', 0) );

for n_pcs = [7, 80]
  
[coeff, score] = pca( im_vecs, 'NumComponents', n_pcs );
recon = (coeff * score')' + mean( im_vecs, 1 );
recon_ims = arrayfun( @(i) reshape(recon(i, :), size(ims{i})), 1:size(recon, 1), 'un', 0 );
recon_errs = vecnorm( im_vecs - recon, 2, 2 );

[~, lb] = ismember( metrics.identifier, im_info.identifier );
metrics.(sprintf('image_recon_error_pca_n_pc_%d', n_pcs)) = recon_errs(lb);

% %%
% 
% imi = 32;
% subplot( 1, 2, 1 ); imshow( ims{imi} ); subplot( 1, 2, 2 ); imshow( recon_ims{imi} );
end


%%  training trajectory

mask = train_metrics.model == 'sc_eval';
[I, C] = findeach( train_metrics, {'layer', 'model', 'src_model'}, mask );
batch_errs = cate1( eachcell(@(x) train_metrics.data{x}.batch_error, I) );

figure(1); clf;
[I, id, C] = rowsets( 2, C, {'model', 'src_model'}, 'layer', 'to_string', true );
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
[I, id, C] = rowsets( 3, err_tbl, {'model', 'src_model'}, 'layer', 'split', 'to_string', true );
C = plots.strip_underscore( C );
axs = plots.simplest_barsets( err_tbl.error, I, id, C );
set( axs, 'xticklabelrotation', 10 );
ylabel( axs(1), 'reconstruction error' );

%%  correlation b/w error and emotionality

per_subj = false;
do_save = false;
rem_outliers = true;
match_lims = false;

% err_metric_name = 'distinctiveness';
err_metric_name = 'error';
% err_metric_name = 'image_recon_error_resnet_image_embedding_layer4';
err_metric = metrics.(err_metric_name);

corr_each = { 'layer', 'model', 'src_model' };
if ( per_subj ), corr_each{end+1} = 'subject'; end

I = findeach( metrics, corr_each );
lt_thresh = false( size(err_metric) );
for i = 1:numel(I)
  lt_thresh(I{i}) = err_metric(I{i}) < prctile(err_metric(I{i}), 95);
end

fig_mask = ternary( rem_outliers, lt_thresh, true(size(err_metric)) );

if ( contains(err_metric_name, 'image_recon_error') )
  % only one "model" and "layer" for recon error
  un_I = findeach( metrics, {'model', 'layer', 'src_model'} );
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

%%  correlation b/w error and emotionality

do_save = false;

err_metric_names = {'error', 'distinctiveness'};
per_valence = false;

corr_stats = table();
for i = 1:numel(err_metric_names)
  err_metric = metrics.(err_metric_names{i});
  corr_each = {'layer', 'model', 'src_model', 'dataset_name'};
  if ( per_valence ), corr_each{end+1} = 'valence'; end
  [I, stats] = findeach( metrics, corr_each );
  [r, p] = cellfun(...
    @(x) corr(err_metric(x), metrics.rating(x), 'type', 'spearman'), I);
  [stats.err_metrics, stats.ratings] = cellfun( @(x) deal(err_metric(x), metrics.rating(x)), I, 'un', 0 );
  stats.r = r(:);
  stats.p = p(:);
  stats.error_metric = repmat( string(err_metric_names{i}), numel(I), 1 );
  corr_stats = [ corr_stats; stats ];
end

as_violin = false;

mask = contains( corr_stats.layer, 'resnet' );
mask = mask & corr_stats.model == 'sc_eval';
mask = mask & corr_stats.src_model ~= 'd3dfr';
mask = mask & ~contains( corr_stats.dataset_name, 'phase_scrambled' );

if ( as_violin )
  [I, id, L] = rowsets( 2, corr_stats ...
    , {'src_model', 'model', 'layer'}, {'error_metric'} ...
    , 'to_string', true, 'mask', mask );
else
  [pcats, gcats, xcats] = deal( { 'src_model', 'model', 'valence' }, {'layer'}, {'error_metric'} );
  [pcats, gcats, xcats] = deal( {'model', 'valence'}, {'layer'}, {'error_metric', 'src_model'} );
%   [pcats, gcats, xcats] = deal( { 'src_model', 'model'}, {'layer'}, {'error_metric', 'valence'} );
  
  if ( ~per_valence )
    [pcats, gcats, xcats] = deal( ...
        setdiff(pcats, 'valence') ...
      , setdiff(gcats, 'valence') ...
      , setdiff(xcats, 'valence') );
  end
  
  [I, id, L] = rowsets( 3, corr_stats ...
    , pcats, gcats, xcats ...
    , 'to_string', true, 'mask', mask );
end

ord = plots.orderby( L, {'positive', 'negative'} );
ord = plots.orderby( L, {'error', 'distinct'}, ord );
ord = plots.orderby( L, {'arcface_recog'}, ord );

[I, id, L] = rowref_many( ord, I, id, L );

L = plots.strip_underscore( L );
L = strrep( L, 'd3dfr', 'EIG' );
L = strrep( L, 'arcface recog', 'FaceDeepNet' );
L = strrep( L, 'resnet image embedding', 'ImageNet' );
L = strrep( L, 'sc eval', 'Sparse Coding' );
L = strrep( L, 'error', 'reconstruction error' );
L = strrep( L, 'negative', 'Negative Valence' );
L = strrep( L, 'positive', 'Positive Valence' );
figure(1); clf;

if ( as_violin )
  axs = plots.violins( corr_stats.r, I, id, L );  
  hold( axs, 'on' );
  shared_utils.plot.add_horizontal_lines( axs, 0 );
else
  [axs, hs, xs, ip] = plots.simplest_barsets( corr_stats.r, I, id, L ...
    , 'error_func', @plotlabeled.nansem ...
    , 'as_line_plot', false ...
  );
  for i = 1:numel(ip)
    lims = get( axs(i), 'ylim' );
    ps = cellfun( @(x) corr_stats.p(x), ip{i} );
    y = cellfun( @(x) mean(corr_stats.r(x)), ip{i} );
    y = y + diff( lims ) * 0.1 * sign( y );
    pm = ps < 0.05;
    scatter( axs(i), xs{i}(pm), y(pm), 'k*' );
  end
end

set( axs, 'xticklabelrotation', 0 );
shared_utils.plot.match_ylims( axs );
plots.onelegend( gcf, 'last' );
ylabel( axs(1), 'R value of correlation' );
ylim( axs, [-0.9, 0.9] );

style_bar_plots( axs, 'legend_location', 'southeast', 'no_error_lines', 1, 'prefer_valence_coloring', 1 );

if ( do_save )
  fname = 'bar';
  save_p = fullfile( data_root, 'plots', dsp3.datedir, 'corr_emotion' );
  shared_utils.io.require_dir( save_p );
  shared_utils.plot.save_fig( gcf, fullfile(save_p, fname), {'png', 'fig'}, false );
  exportgraphics( gcf, fullfile(save_p, sprintf('%s.eps', fname)), 'ContentType', 'vector' );
end

%%

sel = ...
  stats.layer == 'resnet_layer3' & ...
  stats.model == 'sc_eval' & ...
  stats.src_model == 'arcface_recog' & ...
  stats.valence == 'negative';
assert( sum(sel) == 1 );

figure(1); clf;
scatter( stats.err_metrics{sel}, stats.ratings{sel} );

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
  , 'image_recon_error_resnet_image_embedding_layer1' ...
  , 'image_recon_error_resnet_image_embedding_layer2' ...
  , 'image_recon_error_resnet_image_embedding_layer3' ...
  , 'image_recon_error_resnet_image_embedding_layer4' ...
  , 'image_recon_error_pca_n_pc_7' ...
  , 'image_recon_error_pca_n_pc_80' ...
};

% err_var_names = {...
%     'image_recon_error_pca_n_pc_7' ...
%   , 'image_recon_error_pca_n_pc_80' ...
% };

err_var_names = err_var_names(1:2);
% err_var_names = err_var_names(2);

seq_err_tbls = table();
for i = 1:numel(err_var_names)
  fprintf( '\n %d of %d', i, numel(err_var_names) );
  mask = rowmask( metrics );
%   mask = metrics.model == 'sc_n_pc_40_eval';
  is_image_metric = false;
  if ( contains(err_var_names{i}, 'image_recon') )
    % only one "layer" / "model" per image
    mask = ref( findeach(metrics, {'layer', 'model', 'src_model', 'dataset_name'}), '{}', 1 );
    is_image_metric = true;
  end
  err_tbl = sequence_estimation( seq_tbl, im_info, metrics(mask, :), err_var_names{i} );
  err_tbl.error_metric = repmat( string(err_var_names{i}), rows(err_tbl), 1 );
  if ( is_image_metric )
    err_tbl.model(:) = "image";
    err_tbl.layer(:) = "image";
    err_tbl.src_model(:) = "image";
  end
  seq_err_tbls = [ seq_err_tbls; err_tbl ];
end

% add tcc model metrics
mask = ref( findeach(seq_err_tbls, {'error_metric', 'layer', 'model', 'src_model', 'dataset_name'}), '{}', 1 );
assert( numel(mask) == numel(rb_ensemble_code_est) );
err_tbl = seq_err_tbls(mask, :);
err_tbl.z_model_estimate(:) = nan;
err_tbl.z_model_estimate_diff(:) = nan;
err_tbl.model_estimate = rb_ensemble_code_est;
err_tbl.model_estimate_diff = rb_ensemble_code_est - err_tbl.face_mean;
[err_tbl.model(:), err_tbl.layer(:), err_tbl.src_model(:)] = deal( "robinson_brady_ensemble_code" );
seq_err_tbls = [ seq_err_tbls; err_tbl ];

%%  corr estimates trial by trial

behav_var = 'estimation_diff';
model_var = 'model_estimate_diff';

% behav_var = 'estimation';
% model_var = 'model_estimate';

% model_name = 'sc_eval';
% model_name = 'pca_nc_80_eval';
model_name = 'sc_n_pc_80_eval';

err_tbl = seq_err_tbls;

bootstrap_n = 1000;
bootstrap_frac = 0.125 * 0.5;
do_bootstrap = true;
residualize_distinctiveness = true;
per_valence = false;

if ( 0 )  % zscore
  [~, ~, ic] = unique( err_tbl(:, {'src_model', 'model', 'layer', 'error_metric'}) );
  ic = groupi( ic );
  for i = 1:numel(ic)
    err_tbl.(model_var)(ic{i}) = zscore( err_tbl.(model_var)(ic{i}) );
  end
end

if ( 0 )
  % average within sequence id
  seq_vars = {'src_model', 'model', 'layer', 'error_metric', 'sequence_id', 'face_number'};
  [C, ~, ic] = unique( err_tbl(:, seq_vars) );
  I = groupi( ic );
  vars = { behav_var, model_var };
  for i = 1:numel(vars)
    C.(vars{i}) = rowifun( @mean, I, err_tbl.(vars{i}) );
  end
  err_tbl = C;
end

corr_each = {'model', 'layer', 'src_model', 'error_metric'};
if ( per_valence ), corr_each{end+1} = 'valence'; end
[I, C] = findeach( err_tbl, corr_each );

if ( residualize_distinctiveness )  
  % residualize distinctiveness
  rs = nan( numel(I), 1 );
  ps = nan( size(rs) );
  
  tot_rs = cell( size(I) );
  tot_ps = cell( size(I) );
  
  distinct_rs = nan( size(rs) );
  distinct_ps = nan( size(rs) );
  
  parfor i = 1:numel(I)
    fprintf( '\n %d of %d', i, numel(I) );
    search = C(i, :);
    search.error_metric = 'distinctiveness';
    xi = I{i};
    zi = find( ismember(err_tbl(:, C.Properties.VariableNames), search) );
    
    tmp_rs = nan( bootstrap_n, 1 );
    tmp_ps = nan( size(tmp_rs) );
    if ( numel(zi) == numel(xi) )
      % residualize-out distinctiveness
      if ( do_bootstrap )
        for it = 1:bootstrap_n
          ind = randsample(numel(xi), ceil(bootstrap_frac * numel(xi)), true);
          tmp_xi = xi(ind);
          tmp_zi = zi(ind);
          [tmp_rs(it), tmp_ps(it)] = partialcorr( ...
              err_tbl.(model_var)(tmp_xi) ...
            , err_tbl.(behav_var)(tmp_xi), err_tbl.(model_var)(tmp_zi) );
        end
      else
        [rs(i), ps(i)] = partialcorr( ...
            err_tbl.(model_var)(xi) ...
          , err_tbl.(behav_var)(xi), err_tbl.(model_var)(zi) );

        % compute correlation with distinctiveness
        [distinct_rs(i), distinct_ps(i)] = corr( ....
          err_tbl.(behav_var)(xi), err_tbl.(model_var)(zi), 'type', 'Spearman' );
      end
    end
    
    tot_rs{i} = tmp_rs;
    tot_ps{i} = tmp_ps;
  end
  
  C.r = rs;
  C.p = ps;
  C.rs = tot_rs;
  C.ps = tot_ps;
  
  [C.distinct_r, C.distinct_p] = deal( distinct_rs, distinct_ps );
  
elseif ( do_bootstrap )
  
  tot_rs = cell( size(I) );
  tot_ps = cell( size(I) );
  parfor i = 1:numel(I)
    fprintf( '\n %d of %d', i, numel(I) );
    rs = nan( bootstrap_n, 1 );
    ps = nan( size(rs) );
    for it = 1:bootstrap_n
      ind = I{i};
      ind = ind(randsample(numel(ind), ceil(bootstrap_frac * numel(ind)), true));
      [rs(it), ps(it)] = corr( ...
          err_tbl.(model_var)(ind) ...
        , err_tbl.(behav_var)(ind), 'type', 'Spearman' );
    end
    
    tot_rs{i} = rs;
    tot_ps{i} = ps;
  end
  
  C.rs = tot_rs;
  C.ps = tot_ps;
else
  [C.r, C.p] = cellfun( ...
    @(x) corr(err_tbl.(model_var)(x) ...
    , err_tbl.(behav_var)(x), 'type', 'Spearman'), I );
end

C = relabel_image_reconstruction_error_metrics( C );

distincti = find( C.error_metric == 'distinctiveness' );
keep_di = intersect( distincti, find(C.model == model_name) );
del_di = setdiff( distincti, keep_di );
% C.model(keep_di) = 'image';
C(del_di, :) = [];
C.error_metric(C.error_metric == 'error') = 'recon_error';

%%  plot corr estimates trial by trial

mask = ...
  (C.model == model_name | C.model == 'image');
mask = mask & (...
  C.error_metric ~= 'distinctiveness' | (...
  C.error_metric == 'distinctiveness' & (...
    C.src_model == 'd3dfr' | C.src_model == 'arcface_recog' | C.src_model == 'resnet_image_embedding')))
% mask = mask & (...
%   C.error_metric ~= 'image_recon_error' | (...
%   C.error_metric == 'image_recon_error' & contains(C.layer, 'image_recon_error_pca_')));

mask = mask & C.error_metric ~= 'distinctiveness';
% mask = mask & C.error_metric ~= 'error';
mask = mask & C.error_metric ~= 'image_recon_error';

mask = mask & ~contains(C.layer, "ReconNetWrapper_output");

match_lims = true;
fixed_lims = [-0.1, 0.1];
% fixed_lims = [];

if ( 1 )
  if ( per_valence )
    [pcats, xcats, gcats] = deal(...
      {'src_model', 'error_metric', 'model', }, {'layer'}, {'valence'});
  else  
    [pcats, xcats, gcats] = deal(...
      {'src_model', 'error_metric', 'model', }, {'layer'}, {});
  end
  [I, id, L] = rowsets( 3, C, pcats, xcats, gcats ...
      , 'mask', mask, 'to_string', true ...
    );

  L = plots.strip_underscore( L );
  ord = plots.orderby( L, 'd3dfr' );
  ord = plots.orderby( L, {'recon error', 'distinct', 'image recon error'}, ord );
%   ord = plots.orderby( L, 'distinct', ord );
  ord = plots.orderby( L, {'positive', 'negative'}, ord );
  [I, id, L] = rowref_many( ord, I, id, L );
  
  L = strrep( L, 'd3dfr', 'EIG' );
  L = strrep( L, 'arcface recog', 'Face ID' );
  L = strrep( L, 'resnet image embedding', 'ImageNet' );
  L = strrep( L, 'pca nc 80 eval', 'PCA' );
  L = strrep( L, 'sc eval', 'Sparse coding' );
  
  figure(1); clf;
  [axs, hs, xs] = plots.simplest_barsets( C.r, I, id, L ...
    , 'error_func', @plotlabeled.nansem, 'panel_shape', [] );
  if ( match_lims ), shared_utils.plot.match_ylims( axs ); end
  if ( ~isempty(fixed_lims) ), shared_utils.plot.set_ylims(axs, fixed_lims); end
  set( axs, 'xticklabelrotation', 15 );
  ylabel( axs, 'corr_r' );
  if ( 1 )
    PI = plots.nest3( id, I, L );
    for i = 1:numel(PI)
      hold( axs(i), 'on' );
      for j = 1:numel(PI{i})
        if ( isempty(PI{i}{j}) ), continue; end
        ind = PI{i}{j};
        p = C.p(ind);
        if ( p < 0.05 )
          scatter( axs(i), xs{i}(j), C.r(ind)+ sign(C.r(ind)) * 0.01, 'k*' );
        end
      end
    end
  end  
else
  [I, L] = findeach( C, {'model', 'layer', 'src_model'} );
  p_sig = cellfun( @(x) pnz(C.p(x) < 0.05), I );

  figure(1); clf;

  if ( 0 )
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
end

plots.onelegend( gcf );

%%  

test_pairs = true;
fixed_lims = [0, 0.5];

% plt_model_name = 'sc_eval';
% plt_model_name = 'pca_nc_80_eval';
plt_model_name = 'sc_n_pc_80_eval';

mask = (C.model == plt_model_name | C.model == 'image');
mask = mask & C.error_metric == 'recon_error';
mask = mask & ~contains(C.layer, "ReconNetWrapper_output");
% mask = mask & ...
%   (C.src_model ~= 'arcface_recog' | C.src_model == 'arcface_recog' & C.layer == 'resnet_layer2') & ...
%   (C.src_model ~= 'd3dfr' | C.src_model == 'd3dfr' & C.layer == 'resnet_layer2') & ...
%   (C.src_model ~= 'resnet_image_embedding' | C.src_model == 'resnet_image_embedding' & C.layer == 'resnet_layer4');
mask = mask & contains(C.layer, 'resnet');
mask = mask & strcmp(C.valence, 'negative');

base = C(mask, :);
inds = repelem( 1:rows(base), numel(base.rs{1}) );
unpacked = base(inds, :);
unpacked.r = cate1( base.rs );
unpacked.p = cate1( base.ps );

if ( 1 )
  unpacked.r = abs( unpacked.r );
end

[pcats, gcats] = deal( {'error_metric', 'valence', 'layer'}, {'src_model'} );
[I, id, L] = rowsets( 2, unpacked, pcats, gcats ...
  , 'to_string', false ...
);
ord = plots.orderby( plots.cellstr_join(L), {'layer1', 'layer2', 'layer3', 'layer4'} );
[I, id, L] = rowref_many( ord, I, id, L );

tbls = cate1( arrayfun(@(x) horzcat(L{x, :}), 1:rows(L), 'un', 0) );
PI = findeach( tbls, pcats );

if ( test_pairs )
  
comps = cell( size(PI) );
for i = 1:numel(PI)
  inds = PI{i};
  xs = nchoosek( 1:numel(inds), 2 );
  pairs = inds(xs);
  ps = nan( size(pairs, 1), 1 );
  for j = 1:size(pairs, 1)
    a = unpacked.r(I{pairs(j, 1)});
    b = unpacked.r(I{pairs(j, 2)});
%     error( 'xx' );
    [h, ps(j), stats] = ttest2( a, b );
  end
  comps{i} = struct( 'x', xs, 'stat', ps );
end
end

figure(1); clf; 

L = plots.strip_underscore( plots.cellstr_join(L) );
L = strrep( L, 'd3dfr', 'EIG' );
L = strrep( L, 'arcface recog', 'Face ID' );
L = strrep( L, 'resnet image embedding', 'ImageNet' );

axs = plots.violins( unpacked.r, I, id, L ...
  , 'panel_shape', [2, 2] ...
);
shared_utils.plot.match_ylims( axs );
if ( ~isempty(fixed_lims) ), shared_utils.plot.set_ylims(axs, fixed_lims); end
hold( axs, 'on' );
ylabel( axs(1), 'Bootstrapped correlation coefficient' );
% [PI, PL] = plots.nest2( id, I, plots.cellstr_join(L) );
% plots.simple_boxsets( plots.panels(numel(PI)), unpacked.r, PI, PL );

if ( test_pairs )
  for i = 1:numel(comps)
    cs = comps{i};
    lims = get( axs(1), 'ylim' );
    
    for j = 1:size(cs.x, 1)
      x0 = cs.x(j, 1);
      x1 = cs.x(j, 2);
      y = lims(2) - (j * diff(lims) * 0.025);
      plot( axs(i), [x0, x1], [y, y] );
      ind = [];
      threshs = [0.05, 0.01, 0.001];
      stars = { '*', '**', '***' };
      for k = 1:numel(threshs)
        if ( cs.stat(j) < threshs(k) )
          ind = k;
        end
      end
      
      if ( ~isempty(ind) )
        text( axs(i), mean([x0, x1]), y + diff(lims) * 0.005, stars{ind} );
      end
    end
  end
end

%%

d_sig = C.distinct_p < 0.05 & mask & C.error_metric ~= 'distinctiveness';
C(d_sig, :)

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

%%

face_len1 = seq_err_tbl.face_number == 1;

[~, ~, expr_ids] = unique( seq_err_tbl.faceIdentity );
[~, ~, val_ids] = unique( seq_err_tbl.valence );
[~, ~, subj_ids] = unique( seq_err_tbl.participant_id );
seq_ids = [ expr_ids, val_ids, seq_err_tbl.sequence(:, 1) ];

un_subjs = findeachv( subj_ids );
ids_per_subj = cell( size(un_subjs) );
for i = 1:numel(un_subjs)
  this_subj = intersect( un_subjs{i}, find(face_len1) );
  [image_I, ids] = findeach( seq_ids, 1:size(seq_ids, 2), this_subj );
  ids_per_subj{i} = ids;
end

error( 'xx' );

[I, C] = findeach(...
  seq_err_tbl, {'layer', 'src_model', 'model', 'valence', 'error_metric'} );

behav_est = seq_err_tbl.estimation_diff;
model_est = seq_err_tbl.model_estimate_diff;

for i = 1:numel(I)
  match_ims = cellfun( @(x) intersect(I{i}, x), image_I, 'un', 0 );
  error( 'xx' );
end

%%

[I, C] = findeach( seq_err_tbl, {'layer', 'src_model', 'model', 'valence', 'error_metric'} );
a = seq_err_tbl.model_estimate_diff;
b = seq_err_tbl.estimation_diff;
[r, p] = cellfun( @(x) corr(a(x), b(x), 'type', 'spearman'), I );

corr_stats = C;
corr_stats.r = r;
corr_stats.p = p;

mask = contains( corr_stats.layer, 'resnet' ) & corr_stats.model == 'sc_eval';
mask = mask & corr_stats.src_model ~= 'd3dfr';

pcats = { 'src_model', 'model' };
if ( per_valence ), pcats{end+1} = 'valence'; end
[I, id, L] = rowsets( 3, corr_stats ...
  , pcats, {'layer'}, {'error_metric'} ...
  , 'to_string', true, 'mask', mask );

ord = plots.orderby( L, {'error', 'distinct'} );

[I, id, L] = rowref_many( ord, I, id, L );

L = plots.strip_underscore( L );
L = strrep( L, 'd3dfr', 'EIG' );
L = strrep( L, 'arcface recog', 'FaceDeepNet' );
L = strrep( L, 'resnet image embedding', 'ImageNet' );
L = strrep( L, 'sc eval', 'Sparse Coding' );
L = strrep( L, 'error', 'reconstruction error' );

figure(1); clf;
axs = plots.simplest_barsets( corr_stats.r, I, id, L );
shared_utils.plot.match_ylims( axs );

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

seq_err_tbl = relabel_image_reconstruction_error_metrics( seq_err_tbls );

% err_var_name = 'distinctiveness';
err_var_name = 'error';
% err_var_name = 'image_recon_error';
seq_err_tbl = seq_err_tbl(seq_err_tbl.error_metric == err_var_name, :);

do_save = true;
include_behav = true;
per_model = true;
selected_src_model_names = {'arcface_recog', 'resnet_image_embedding', 'behavior'};
% selected_src_model_names = { 'arcface_recog', 'resnet_image_embedding' };
selected_src_model_names = {'arcface_recog', 'behavior'};
only_best_layers = true;
% selected_src_model_name = 'resnet_image_embedding';
% selected_src_model_name = 'd3dfr';
do_z = true;
do_z_post = true;
match_lims = true;
zero_centered_lims = true;
per_valence = true;
% model_name = 'pca_nc_80_eval';
% model_name = 'sc_n_pc_80_eval';
model_name = 'sc_eval';
fixed_lims = [-0.8, 0.6];
fixed_lims = [-0.8, 0.8];
dup_behav = false;
panel_shape = [2, 4];
panel_shape = [];

% fixed_lims = [-0.8, 0.4];

model_var = 'model_estimate_diff';
if ( do_z && ~do_z_post )
  model_var = 'z_model_estimate_diff';

elseif ( do_z && do_z_post )
  [~, ~, ic] = unique( seq_err_tbl(:, {'model', 'layer', 'src_model', 'error_metric', 'dataset_name'}) );
  ic = groupi( ic );
  for i = 1:numel(ic)
    mv = seq_err_tbl.(model_var)(ic{i});
    seq_err_tbl.(model_var)(ic{i}) = zscore( mv );
  end
end

% include behavior
if ( include_behav )
  [~, ~, ic] = unique( seq_err_tbl(:, {'model', 'layer', 'src_model', 'dataset_name'}) );
  ic = groupi( ic );
  if ( ~dup_behav ), ic = ic(1); end
  for i = 1:numel(ic)
    behav_tbl = seq_err_tbl(ic{i}, :);
    behav_est = behav_tbl.estimation_diff;
    if ( do_z ) behav_est = zscore( behav_est ); end
    behav_tbl.(model_var) = behav_est;
    behav_tbl.model(:) = model_name;
    behav_tbl.src_model(:) = 'behavior';
    if ( ~dup_behav ), behav_tbl.layer(:) = 'behavior'; end
    seq_err_tbl = [ seq_err_tbl; behav_tbl ];
  end
end

mask = true( size(seq_err_tbl, 1), 1 );
%   mask = seq_err_tbl.layer == 'resnet_output';

if ( ~strcmp(err_var_name, 'image_recon_error') )
  mask = seq_err_tbl.src_model == 'behavior' | (seq_err_tbl.model == model_name & contains(seq_err_tbl.layer, 'resnet'));
end

if ( only_best_layers )
  mask = mask & (...
    (seq_err_tbl.src_model == 'arcface_recog' & seq_err_tbl.layer == 'resnet_layer2') | ...
    (seq_err_tbl.src_model == 'resnet_image_embedding' & seq_err_tbl.error_metric == 'error' & seq_err_tbl.layer == 'resnet_layer1') | ...
    (seq_err_tbl.src_model == 'resnet_image_embedding' & seq_err_tbl.error_metric == 'distinctiveness' & seq_err_tbl.layer == 'resnet_layer2') | ...
    (seq_err_tbl.layer == 'behavior'));  
end

mask = mask & ~contains( seq_err_tbl.dataset_name, 'phase_scrambled' );

%   mask = seq_err_tbl.model == 'sc_eval';
%   mask = mask & seq_err_tbl.src_model == 'resnet_image_embedding';

% mask = mask & ...
%   (seq_err_tbl.layer == 'resnet_layer4') | ...
%   (seq_err_tbl.layer == 'behavior');

if ( per_model )
  mask = mask & ismember(seq_err_tbl.src_model, selected_src_model_names);
%   mask = mask & seq_err_tbl.src_model == 'resnet_image_embedding';
%   mask = mask & seq_err_tbl.src_model == 'd3dfr';
  
elseif ( ~dup_behav )
%   mask = mask & (...
%     (seq_err_tbl.src_model == 'd3dfr' & (...
%     seq_err_tbl.layer == 'ReconNetWrapper_output_identity_expression' | ...
%     seq_err_tbl.layer == 'ReconNetWrapper_output')) | ...
%     (seq_err_tbl.layer == 'behavior'));  
  
  mask = mask & (...
    (seq_err_tbl.src_model == 'd3dfr' & seq_err_tbl.layer == 'resnet_layer2') | ...
    (seq_err_tbl.src_model == 'arcface_recog' & seq_err_tbl.layer == 'resnet_layer2') | ...
    (seq_err_tbl.src_model == 'resnet_image_embedding' & seq_err_tbl.layer == 'resnet_layer4') | ...
    (seq_err_tbl.layer == 'behavior'));
%     (contains(seq_err_tbl.src_model, 'image'));
end

if ( 0 )  % for distinctiveness
  mask = mask & (...
    seq_err_tbl.src_model == 'd3dfr' | ...
    seq_err_tbl.src_model == 'behavior');
end

seq_mus = nanmean( seq_err_tbl.sequence, 2 );
if ( 0 )
  mask = intersect( mask, find(seq_mus < prctile(seq_mus, 100)) );
end

plt_var = seq_err_tbl.(model_var);

if ( per_valence )
  [I, id, L] = rowsets( 3, seq_err_tbl ...
    , {'src_model', 'layer'}, {'face_number'}, {'valence'} ...
    , 'to_string', true, 'mask', mask );
else
  if ( ~dup_behav )
    [pcats, xcats, gcats] = deal(...
      {'src_model'}, {'face_number'}, {'model', 'layer', 'error_metric', 'valence'});  
  else
    [pcats, xcats, gcats] = deal(...
      {'model', 'error_metric', 'layer'}, {'face_number'}, {'src_model'});  
  end
  [I, id, L] = rowsets( 3, seq_err_tbl, pcats, xcats, gcats ...
    , 'to_string', true, 'mask', mask );
end
L = plots.strip_underscore( L );
ord = plots.orderby( L, {'layer1', 'layer2', 'layer3', 'layer4', 'output'} );
ord = plots.orderby( L(:, 2), arrayfun(@num2str, 1:12, 'un', 0), ord );
ord = plots.orderby( L, 'positive', ord );
ord = plots.orderby( L, {'behavior', 'd3dfr', 'arcface'}, ord );
[I, id, L] = rowref_many( ord, I, id, L );

L = strrep( L, 'd3dfr', 'EIG' );
L = strrep( L, 'arcface recog', 'FaceDeepNet' );
L = strrep( L, 'resnet image embedding', 'ImageNet' );

figure(1); clf;
[axs, hs] = plots.simplest_barsets( plt_var, I, id, L ...
  , 'error_func', @plotlabeled.nansem ...
  , 'as_line_plot', true ...
  , 'panel_shape', panel_shape ...
);

plots.onelegend( gcf );

if ( 1 )
  for i = 1:numel(hs)
    if ( isempty(hs{i}) ), continue; end
    [hs{i}.LineWidth] = deal( 2 );
  end
end

if ( 1 )
  for i = 1:numel(hs)
    if ( isempty(hs{i}) ), continue; end
    names = arrayfun( @(x) string(x.DisplayName), hs{i} );
    ind = find( contains(names, 'behavior') );
    [hs{i}(ind).LineWidth] = deal( 3 );
  end
end

if ( zero_centered_lims )
  for i = 1:numel(axs)
    lim = max( abs(get(axs(i), 'ylim')) );
    ylim( axs(i), [-lim, lim] );
  end
end

if ( match_lims )
  shared_utils.plot.match_ylims( axs );
end

if ( ~isempty(fixed_lims) ), shared_utils.plot.set_ylims( axs, fixed_lims ); end

ylab = sprintf('Difference in %s', err_var_name);
if ( strcmp(err_var_name, 'error') )
  ylab = 'Difference in reconstruction error';
end

if ( do_z ), ylab = sprintf( '%s (zscored)', ylab ); end
xlabel( axs, 'Face number' );

if ( include_behav && ismember('behavior', selected_src_model_names) )
  ylabel( axs(1), 'Emotion estimate difference (zscored)' );
  ylabel( axs(2), ylab );
else
  ylabel( axs(1), ylab );
end

if ( 0 )
  max_lim = max( get(axs(1), 'ylim') ) * 1.125;
  ylim( axs(1), [-max_lim, max_lim] );
end

strs = arrayfun( @(x) string(get(get(x, 'title'), 'string')), axs );
strs(strs == "behavior | behavior") = "Behavior";
for i = 1:numel(strs), set(get(axs(i), 'title'), 'string', strs(i)), end

style_line_plots( axs );

if ( do_save )
  fname = 'sequence';
  if ( only_best_layers ), fname = sprintf( '%s_best', fname ); end
  save_p = fullfile( data_root, 'plots', dsp3.datedir, 'sequence', err_var_name );
  shared_utils.io.require_dir( save_p );
%   shared_utils.plot.fullscreen( gcf );
  shared_utils.plot.save_fig( gcf, fullfile(save_p, fname), {'png', 'fig'}, false );
  exportgraphics( gcf, fullfile(save_p, sprintf('%s.eps', fname)), 'ContentType', 'vector' );
end

%%  stats on mean sequence traces from 1c

%%% stat
model_name = 'sc_eval';
model_var = 'model_estimate_diff';
do_bootstrap = false;
% bootstrap_n = 1000;
bootstrap_n = 100;
bootstrap_frac = 0.75;

mask_fn = @(t) ...
  t.src_model == 'behavior' | (t.layer == 'resnet_layer2' & t.error_metric == 'error' & t.src_model == 'arcface_recog');
mask_fn = @(t) true(rows(t), 1);

%%% what column to bootstrap and compare between (p(a < b))?
% comp_col = 'error_metric';
% search_a = @(col) col == 'error';
% search_b = @(col) col == 'distinctiveness';

comp_col = 'dataset_name';
search_a = @(col) contains(col, 'valid_expression_balanced_var_subsample_1000');
search_b = @(col) contains(col, 'valid_expression_balanced_phase_scrambled_var_subsample_1000');
%%%

select_valences = { 'positive', 'negative' };

% for examining relationship between phase-scrambled and non-phase
% scrambled images
include_phase_scrambled = true;
if ( include_phase_scrambled )
  select_valences = { 'negative' };
end

seq_err_tbl = relabel_image_reconstruction_error_metrics( seq_err_tbls );

[~, ~, ic] = unique( seq_err_tbl(:, {'model', 'layer', 'src_model', 'error_metric', 'dataset_name'}) );
ic = groupi( ic );
for i = 1:numel(ic)
  mv = seq_err_tbl.(model_var)(ic{i});
  seq_err_tbl.(model_var)(ic{i}) = zscore( mv );
end

is_val = ismember( seq_err_tbl.valence, select_valences );
seq_err_tbl.(model_var)(~is_val) = nan;

% include behavior
[~, ~, ic] = unique( seq_err_tbl(:, {'model', 'layer', 'src_model', 'error_metric', 'dataset_name'}) );
ic = groupi( ic );
ic = ic(1);
for i = 1:numel(ic)
  behav_tbl = seq_err_tbl(ic{i}, :);
  behav_est = behav_tbl.estimation_diff;
  behav_tbl.(model_var) = behav_est;
  behav_tbl.model(:) = model_name;
  behav_tbl.src_model(:) = 'behavior';
  behav_tbl.layer(:) = 'behavior';
  seq_err_tbl = [ seq_err_tbl; behav_tbl ];
end

plt_var = seq_err_tbl.(model_var);

[I, L] = findeach( seq_err_tbl, {'src_model', 'layer', 'model', 'error_metric', 'dataset_name'} );

L.t_p = nan( rows(L), 1 );
mu_tbls = table();
for i = 1:numel(I)
  pos_ind = intersect( find(strcmp(seq_err_tbl.valence, 'positive')), I{i} );
  neg_ind = intersect( find(strcmp(seq_err_tbl.valence, 'negative')), I{i} );
  pos_I = findeach( seq_err_tbl, 'face_number', pos_ind );
  neg_I = findeach( seq_err_tbl, 'face_number', neg_ind );
  pos_mu = cellfun( @(x) mean(plt_var(x)), pos_I ); 
  neg_mu = cellfun( @(x) mean(plt_var(x)), neg_I );
  [~, L.t_p(i)] = ttest( neg_mu, pos_mu, 'tail', 'right' );
  mu_tbl = [ L(i, :); L(i, :) ];
  mu_tbl.valence = ["positive"; "negative"];
  mu_tbl = mu_tbl(repelem(1:2, numel(pos_mu), 1), :);
  mu_tbl.means = [pos_mu; neg_mu];
  mu_tbls = [ mu_tbls; mu_tbl ];
end
stat_test = L;

if ( ~do_bootstrap ), bootstrap_n = 1; end

tot_rest_seqs = cell( bootstrap_n, 1 );

while ( true )

parfor idx = 1:bootstrap_n
  
fprintf( '\n %d of %d', idx, bootstrap_n );

[I, L] = findeach( seq_err_tbl ...
  , {'src_model', 'layer', 'face_number', 'valence', 'model', 'error_metric', 'dataset_name'} ...
  , mask_fn(seq_err_tbl) ...
);

if ( do_bootstrap )
  vi = findeach( L, {'valence', 'face_number'} );
  for j = 1:numel(vi)
    ii = vi{j}(1);
    sub_ind = sort( randsample(numel(I{ii}), ceil(bootstrap_frac * numel(I{ii})), true) );
    I(vi{j}) = cellfun( @(x) x(sub_ind), I(vi{j}), 'un', 0 );
  end
end

ord = unique( L.face_number );
si = findeach( L, setdiff(L.Properties.VariableNames, {'valence', 'face_number'}) );

L.I = I;

seqs = table();
for i = 1:numel(si)
  pos_ind = intersect( find(strcmp(L.valence, 'positive')), si{i} );
  neg_ind = intersect( find(strcmp(L.valence, 'negative')), si{i} );
  L_pos = L(pos_ind, :);
  L_neg = L(neg_ind, :);
  [~, ord_pos] = ismember( ord, L_pos.face_number );
  L_pos = L_pos(ord_pos, :);
  [~, ord_neg] = ismember( ord, L_neg.face_number );
  L_neg = L_neg(ord_neg, :);
  L_both = [ L_pos; L_neg ];
  vec = cellfun( @(x) mean(plt_var(x)), L_both.I );
  un_each = setdiff( L_both.Properties.VariableNames, {'I', 'face_number', 'valence'} );
  seq = unique( L_both(:, un_each) );
  seq.estimate = vec(:)';
  seqs = [ seqs; seq ];
end
behav_ind = seqs.src_model == 'behavior';
assert( sum(behav_ind) == 1 );
ref_est = seqs.estimate(behav_ind, :);
rest_seqs = seqs(~behav_ind, :);
[rest_seqs.r, rest_seqs.p] = arrayfun( ...
  @(x) corr(seqs.estimate(x, :)', ref_est(:), 'rows', 'complete'), (1:rows(rest_seqs))' );
% error( 'xx' );
tot_rest_seqs{idx} = rest_seqs;

end

rest_seqs = vertcat( tot_rest_seqs{:} );

if ( do_bootstrap )  
  [ti, stats] = findeach( rest_seqs, setdiff(...
    rest_seqs.Properties.VariableNames, {'estimate', 'r', 'p', comp_col}) );
  
  has_pairs = cellfun( @numel, ti ) == bootstrap_n * 2;
  ti = ti(has_pairs);
  stats = stats(has_pairs, :);
  
  stats.p = nan( numel(ti), 1 );
  for i = 1:numel(ti)
    recon_ind = intersect( ti{i}, find(search_a(rest_seqs.(comp_col))) );
    distinct_ind = intersect( ti{i}, find(search_b(rest_seqs.(comp_col))) );
    assert( numel(recon_ind) == numel(distinct_ind) );
    num_gt = sum( rest_seqs.r(recon_ind) > rest_seqs.r(distinct_ind) );
    stats.p(i) = 1 - num_gt / numel( recon_ind );
  end
  
%   if ( stats.p(8) <= 0.05 )
%     break
%   end
end

if ( 1 )
  break
end

end

%%  plot mean sequence traces from 1c

as_violin = false;
do_save = false;

mask = contains( rest_seqs.layer, 'resnet' );
mask = mask & rest_seqs.model == 'sc_eval';
mask = mask & rest_seqs.src_model ~= 'd3dfr';
if ( include_phase_scrambled )
%   mask = mask & rest_seqs.error_metric ~= "distinctiveness";
end
mask = mask | rest_seqs.layer == 'robinson_brady_ensemble_code';

if ( ~include_phase_scrambled )
  mask = mask & ~contains( rest_seqs.dataset_name, 'phase_scrambled' );
end

fixed_lims = [-1, 1];
gcats = { 'error_metric', 'src_model' };

if ( include_phase_scrambled ), gcats{end+1} = 'dataset_name'; end

if ( as_violin )
  [I, id, L] = rowsets( 2, rest_seqs ...
    , {'src_model', 'model', 'layer'}, gcats ...
    , 'to_string', true, 'mask', mask );
else
  [I, id, L] = rowsets( 3, rest_seqs ...
    , {'model'}, {'layer'}, gcats ...
    , 'to_string', true, 'mask', mask );
end

L = plots.strip_underscore( L );

L = strrep( L, 'd3dfr', 'EIG' );
L = strrep( L, 'arcface recog', 'FaceDeepNet' );
L = strrep( L, 'resnet image embedding', 'ImageNet' );
L = strrep( L, 'sc eval', 'Sparse Coding' );
L = strrep( L, 'error', 'reconstruction error' );
L = strrep( L, '\valid expression balanced phase scrambled var subsample 1000', ' phase-scrambled' );
L = strrep( L, '\valid expression balanced var subsample 1000', ' original' );
L = strrep( L, 'EIG original', 'original' );
L = strrep( L, 'reconstruction error | original', 'original' );

ord = plots.orderby( L, {'error', 'distinct'} );
ord = plots.orderby( L, {'FaceDeepNet'}, ord );
ord = plots.orderby( L, {'original'}, ord );
[I, id, L] = rowref_many( ord, I, id, L );

figure(1); clf;

if ( as_violin )
  axs = plots.violins( rest_seqs.r, I, id, L );  
  hold( axs, 'on' );
  shared_utils.plot.add_horizontal_lines( axs, 0 );
else
[axs, hs] = plots.simplest_barsets( rest_seqs.r, I, id, L ...
  , 'error_func', @plotlabeled.nansem ...
  , 'as_line_plot', false ...
);
end
shared_utils.plot.match_ylims( axs );
set( axs, 'xticklabelrotation', 10 );
ylabel( axs(1), 'R (correlation of sequence to behavior)' );
% delete( findobj(gcf, 'tag', 'legend') );
% plots.onelegend( gcf );
shared_utils.plot.set_ylims( axs, fixed_lims );

if ( as_violin )
  [PI, PL] = plots.nest2( id, I, L );
  for i = 1:numel(PI)
    lims = get( axs(i), 'ylim' );
    i0 = PI{i}{1};
    i1 = PI{i}{2};
    p = 1 - sum( rest_seqs.r(i0) > rest_seqs.r(i1) ) / numel( i0 );
    threshs = [ 0.1, 0.05, 0.01 ];
    thresh_txt = { '#', '*', '**' };
    thresh_ind = [];
    for j = 1:numel(threshs)
      if ( p < threshs(j) ), thresh_ind = j; end
    end
    if ( ~isempty(thresh_ind) )
      if ( thresh_ind == 1 )
        txt = sprintf( '# (p = %0.3f)', p );
        x_off = -0.25;
      else
        txt = thresh_txt{thresh_ind};
        x_off = 0;
      end
      text( axs(i), 1.5 + x_off, max(lims) - diff(lims) * 0.1, txt );
    end
  end
end

plots.onelegend( gcf );
style_bar_plots( axs, 'no_error_lines', 1, 'distinctiveness_opacity', 0.25 );

if ( do_save )
  fname = 'corr_behavior';
  if ( include_phase_scrambled ), fname = sprintf('%s_phase_scrambled', fname); end
  subdir = 'sequence';
  if ( include_phase_scrambled ), subdir = 'phase_scramble'; end
  save_p = fullfile( data_root, 'plots', dsp3.datedir, subdir );
  shared_utils.io.require_dir( save_p );
  shared_utils.plot.save_fig( gcf, fullfile(save_p, fname), {'png', 'fig'}, false );
  exportgraphics( gcf, fullfile(save_p, sprintf('%s.eps', fname)), 'ContentType', 'vector' );
end

%{
 main effect of sequence and valence
sequence correlation - bootstrap
%}

%%  valence effect on SAE

do_save = true;
plt_behav = false;
combine_err_metrics = true;
rescale_err = true;

plt_tbl = mu_tbls;
plt_var = plt_tbl.means;

plt_tbl = seq_err_tbls;
plt_var = plt_tbl.model_estimate_diff;

if ( rescale_err )
  rescale_mask = plt_tbl.error_metric == 'error';
  plt_var(rescale_mask) = plt_var(rescale_mask) * 100;
end

if ( plt_behav )
  % plot behavior as reference
  [~, ~, ic] = unique( plt_tbl(:, {'model', 'layer', 'src_model', 'error_metric', 'dataset_name'}) );
  ic = ref( groupi(ic), '{}', 1 );
  plt_tbl = plt_tbl(ic, :);
  plt_var = plt_tbl.estimation_diff;
end

if ( 0 )
  [~, ~, ic] = unique( plt_tbl(:, {'model', 'layer', 'src_model', 'error_metric'}) );
  ic = groupi( ic );
  for i = 1:numel(ic)
    mv = plt_var(ic{i});
    plt_var(ic{i}) = zscore( mv );
  end
end

mask = contains( plt_tbl.layer, 'resnet' );
mask = mask & plt_tbl.model == 'sc_eval';
mask = mask & plt_tbl.src_model ~= 'd3dfr';
% mask = mask & plt_tbl.error_metric == 'error';
mask = mask & ~contains( plt_tbl.dataset_name, 'phase_scrambled' );

if ( plt_behav )
  mask(:) = true;
end

if ( combine_err_metrics )
  [pcats, gcats, xcats] = deal( {'src_model'}, {'layer'}, {'valence', 'error_metric'} );
else
  [pcats, gcats, xcats] = deal( {'src_model', 'error_metric'}, {'layer'}, {'valence'} );
end
[I, id, L] = rowsets( 3, plt_tbl ...
  , pcats, gcats, xcats ...
  , 'to_string', true, 'mask', mask );
L = plots.strip_underscore( L );
ord = plots.orderby( L, {'positive', 'negative'} );
ord = plots.orderby( L, {'error'}, ord );
[I, id, L] = rowref_many( ord, I, id, L );
L = rename_models_for_plotting( L );
L = strrep( L, 'error', 'reconstruction error' );
  
figure(1); clf;
[axs, hs, xs, ip] = plots.simplest_barsets( plt_var, I, id, L ...
  , 'error_func', @plotlabeled.nansem );
set( axs, 'xticklabelrotation', 0 );

if ( ~plt_behav )
  shared_utils.plot.match_ylims( axs(1:2) );
  if ( ~combine_err_metrics )
    shared_utils.plot.match_ylims( axs(3:4) );
  end
end

ylabel( axs(1), 'Mean estimate difference' );

if ( ~combine_err_metrics )
for i = 1:numel(ip)
  ips = ip{i};
  lims = get( axs(i), 'ylim' );
  lim = max( lims );
  for j = 1:size(ips, 1)
    a = ips{j, 1};
    b = ips{j, 2}; 
    [h, p] = ttest2( plt_var(b), plt_var(a), 'tail', 'right' );
    if ( h )
      plot( axs(i), mean(xs{i}(j, :)), lim - diff(lims) * 0.1, 'k*' );
    end
  end
end
end

plots.onelegend( gcf, 'last' );
style_bar_plots( axs, 'legend_location', 'southeast', 'prefer_valence_coloring', 1 );

if ( do_save )
  fname = 'valence_effect';
  if ( plt_behav ), fname = sprintf( 'behav_%s', fname ); end
  save_p = fullfile( data_root, 'plots', dsp3.datedir, 'sequence' );
  shared_utils.io.require_dir( save_p );
  shared_utils.plot.save_fig( gcf, fullfile(save_p, fname), {'png', 'fig'}, false );
  exportgraphics( gcf, fullfile(save_p, sprintf('%s.eps', fname)), 'ContentType', 'vector' );
end

%%  

do_transpose = false;

mask = contains( stat_test.layer, 'resnet' );
mask = mask & stat_test.model == 'sc_eval';
mask = mask & stat_test.src_model ~= 'd3dfr';
xs = compose( "resnet_layer%d", 1:4 );
[rest_I, C] = findeach( stat_test ...
  , setdiff(stat_test.Properties.VariableNames, {'t_p', 'layer', 'model'}), mask );
p_mat = zeros( numel(rest_I), numel(xs) );
for i = 1:numel(rest_I)
  ri = rest_I{i};
  assert( isequal(stat_test.layer(ri), xs(:)) );
  p_mat(i, :) = stat_test.t_p(ri) < 0.05;
end

C = arrayfun( @(x) C(x, :), (1:rows(C))', 'un', 0 );
L = plots.cellstr_join( C );
L = plots.strip_underscore( L );
L = rename_models_for_plotting( L );
L = strrep( L, 'error', 'reconstruction error' );

figure(2); clf;
ax = gca;
imagesc( ax, p_mat );

if ( do_transpose )
  [xt, xtl] = deal( 'ytick', 'yticklabel' );
  [yt, ytl] = deal( 'xtick', 'xticklabel' );
else
  [xt, xtl] = deal( 'xtick', 'xticklabel' );
  [yt, ytl] = deal( 'ytick', 'yticklabel' );
end

% set( ax, 'XTickLabelRotation', 10 );
set( ax, 'YTickLabelRotation', 45 );
set( ax, xt, 1:numel(xs) ); set( ax, yt, 1:numel(rest_I) );
set( ax, xtl, plots.strip_underscore(xs) ); set( ax, ytl, L );
title( 'Significance of comparison for valence' );
colormap( 'gray' );

%%  slopes and means as f(prctile)

prctiles = [ 50, 75, 90, 95, 100 ];
[I, C] = findeach( seq_err_tbls, {'src_model', 'model', 'layer', 'error_metric', 'valence'} );
all_mdls = table();
for i = 1:numel(I)
  mdls = fit_sequence_lms_by_prctile( seq_err_tbls(I{i}, :), prctiles );
  mdls = [ mdls, repmat(C(i, :), rows(mdls), 1) ];
  all_mdls = [ all_mdls; mdls ];
end
mdls = all_mdls;

%%  slopes and means

err_tbl = seq_err_tbls;
always_fit_behav = true;

if ( 0 )
  [C, ~, ic] = unique( err_tbl(:, {'src_model', 'model', 'layer', 'error_metric', 'sequence_id', 'face_number'}) );
  I = groupi( ic );
  C.model_estimate_diff = rowifun( @mean, I, err_tbl.model_estimate_diff );
  C.estimation_diff = rowifun( @mean, I, err_tbl.estimation_diff );
  err_tbl = C;
end

fit_each = {'dataset_name', 'src_model', 'model', 'layer', 'error_metric', 'valence'};
if ( 0 ), fit_each(end+1) = { 'valence' }; end
[I, C] = findeach( err_tbl, fit_each );
all_mdls = table();
has_behav_pos = false;
has_behav_neg = false;
transform_ys = @zscore;

for i = 1:numel(I)
  fprintf( '\n %d of %d', i, numel(I) );
  st = err_tbl(I{i}, :);
  xs = st.face_number;
  ys = st.model_estimate_diff;
  mdls = fit_sequence_lm( xs, transform_ys(ys) );
  mdls = [ mdls, repmat(C(i, :), rows(mdls), 1) ];
  all_mdls = [ all_mdls; mdls ];
  
  is_pos = false;
  if ( ismember('valence', fit_each) )
    if ( string(C.valence{i}) == "positive" )
      need_fit_behav = ~has_behav_pos;
      is_pos = true;
    else
      need_fit_behav = ~has_behav_neg;
    end
  else
    need_fit_behav = ~has_behav_neg;
  end
  
  if ( always_fit_behav || need_fit_behav )
    % add behavior
    mdls = fit_sequence_lm( xs, transform_ys(st.estimation_diff) );
    mdls = [ mdls, repmat(C(i, :), rows(mdls), 1) ];
    if ( always_fit_behav )
      [mdls.src_model] = deal( "behavior" );
    else
      [mdls.model, mdls.layer, mdls.error_metric, mdls.src_model] = deal( "behavior" );
    end
    all_mdls = [ all_mdls; mdls ];
    if ( is_pos ), has_behav_pos = true; end
    if ( ~is_pos), has_behav_neg = true; end
  end
end

mdls = all_mdls;
mdls = relabel_image_reconstruction_error_metrics( mdls );

%%  perm test slopes and means

err_tbl = seq_err_tbls;

fit_each = {'src_model', 'model', 'layer', 'error_metric'};

[I, C] = findeach( err_tbl, fit_each );
perm_mdls = table();
transform_ys = @zscore;
perm_its = 100;

for i = 1:numel(I)
  fprintf( '\n %d of %d', i, numel(I) );
  
  pos_ind = intersect( I{i}, find(strcmp(err_tbl.valence, 'positive')) );
  neg_ind = intersect( I{i}, find(strcmp(err_tbl.valence, 'negative')) );
  
  inds = { pos_ind, neg_ind };
  tmp_mdls = cell( size(inds) );
  betas = nan( size(tmp_mdls) );
  for j = 1:numel(inds)
    st = err_tbl(inds{j}, :);
    xs = st.face_number;
    ys = st.model_estimate_diff;

    tmp_mdls{j} = fit_sequence_lm( xs, transform_ys(ys) );
    tmp_mdls{j} = [ tmp_mdls{j}, repmat(C(i, :), rows(tmp_mdls{j}), 1) ];    
    betas(j) = tmp_mdls{j}.mdl{1}.Coefficients.Estimate(2);
  end
  
  true_diff = abs( diff(betas) );
  
  null_diffs = nan( perm_its, 1 );
  for it = 1:numel(null_diffs)
    [~, ~, ic] = unique( err_tbl.face_number );
    ic = groupi( ic );
    
    inds = { [], [] };
    for j = 1:numel(ic)
      posi = intersect( ic{j}, pos_ind );
      negi = intersect( ic{j}, neg_ind );
      num_pos = numel( posi );
      shuff_ind = randperm( numel(posi) + numel(negi) );
      toti = [ posi; negi ];
      inds{1} = [ inds{1}; toti(shuff_ind(1:num_pos)) ];
      inds{2} = [ inds{2}; toti(shuff_ind(num_pos+1:end)) ];
    end
    
    tmp_mdls = cell( size(inds) );
    betas = nan( size(tmp_mdls) );
    for j = 1:numel(inds)
      st = err_tbl(inds{j}, :);
      xs = st.face_number;
      ys = st.model_estimate_diff;

      tmp_mdls{j} = fit_sequence_lm( xs, transform_ys(ys) );
      tmp_mdls{j} = [ tmp_mdls{j}, repmat(C(i, :), rows(tmp_mdls{j}), 1) ];    
      betas(j) = tmp_mdls{j}.mdl{1}.Coefficients.Estimate(2);
    end
    
    null_diffs(it) = abs( diff(betas) );
  end
  
  if ( true_diff >= 0 )
    p_diff = sum( null_diffs > true_diff ) / numel( null_diffs );
  else
    p_diff = sum( null_diffs < true_diff ) / numel( null_diffs );
  end
  
  fprintf( '\n\t p_diff: %0.3f', p_diff );
  
  perm_tbl = C(i, :);
  perm_tbl.p_diff = p_diff;
  perm_mdls = [ perm_mdls; perm_tbl ];
end

perm_mdls = relabel_image_reconstruction_error_metrics( perm_mdls );

%%  plot slopes and means

var_name = 'beta';
% match_lims = false;
match_lims = ismember( var_name, ["r2", "corr_r", "beta"] );
equal_lims = true;
custom_lims = [-0.09, 0.09];
do_save = true;
include_behavior = false;

mask = ...
  (mdls.model == 'sc_eval') | ...
  (mdls.model == 'pca_nc_80_eval') | ...
  (mdls.model == 'sc_n_pc_80_eval') | ...
  (mdls.src_model == 'behavior') | ...
  (mdls.model == 'image');

mask = mask & ~contains(mdls.dataset_name, 'phase_');

% mask = mask & mdls.error_metric ~= 'distinctiveness';
mask = mask & mdls.layer ~= "resnet_output";

% mask = mask & mdls.model == 'sc_n_pc_80_eval';
% mask = mask & mdls.model == 'pca_nc_80_eval';
mask = mask & mdls.model == 'sc_eval';
% mask = mask & (...
%   mdls.error_metric ~= 'distinctiveness' | (...
%   mdls.error_metric == 'distinctiveness' & mdls.src_model == 'd3dfr'));
% mask = mask & mdls.error_metric ~= 'distinctiveness';

% mask = mask & mdls.error_metric ~= 'distinctiveness';
mask = mask & contains(mdls.layer, 'resnet');
mask = mask & mdls.src_model ~= 'd3dfr';

% mask = mask & (...
%   mdls.src_model ~= 'd3dfr' | (...
%   mdls.src_model == 'd3dfr' & mdls.layer == 'resnet_layer2'));
% 
% mask = mask & (...
%   mdls.src_model ~= 'arcface_recog' | (...
%   mdls.src_model == 'arcface_recog' & mdls.layer == 'resnet_layer2'));
% 
% mask = mask & (...
%   mdls.src_model ~= 'resnet_image_embedding' | (...
%   mdls.src_model == 'resnet_image_embedding' & mdls.layer == 'resnet_layer4'));
% 
mask = mask & (...
  mdls.src_model ~= 'behavior' | (...
  mdls.src_model == 'behavior' & mdls.layer == 'resnet_layer1' & mdls.error_metric == 'distinctiveness'));

if ( ~include_behavior )
  mask = mask & mdls.src_model ~= 'behavior';
end

if ( ismember('valence', mdls.Properties.VariableNames) )
%   [pcats, gcats, xcats] = deal( {'src_model', 'model', 'error_metric'}, {'layer'}, {'valence'} );
  [pcats, gcats, xcats] = deal( {'src_model', 'model'}, {'layer'}, {'error_metric', 'valence'} );
  [I, id, C] = rowsets( 3, mdls ...
    , pcats, gcats, xcats ...
    , 'mask', mask, 'to_string', true ...
  );
else
  [I, id, C] = rowsets( 3, mdls ...
    , {'model'}, {'layer'}, {'error_metric', 'src_model'} ...
    , 'mask', mask, 'to_string', true ...
  );
end
C = plots.strip_underscore( C );
ord = plots.orderby( C, {'behavior', 'd3dfr'} );
ord = plots.orderby( C, {'positive', 'negative'}, ord );
ord = plots.orderby( C, compose("layer%d", 1:4), ord );
ord = plots.orderby( C, 'error', ord );
[I, id, C] = rowref_many( ord, I, id, C );

C = strrep( C, 'd3dfr', 'EIG' );
C = strrep( C, 'arcface recog', 'FaceDeepNet' );
C = strrep( C, 'resnet image embedding', 'ImageNet' );
C = strrep( C, 'sc eval', 'Sparse Coding' );
C = strrep( C, 'error', 'reconstruction error' );

figure(1); clf;
[axs, hs, xs] = plots.simplest_barsets( mdls.(var_name), I, id, C );
if ( match_lims ), shared_utils.plot.match_ylims( axs ); end
set( axs, 'xticklabelrotation', 15 );
ylabel( axs(1), var_name );

if ( equal_lims )
  lim = max( abs(get(axs(1), 'ylim')) );
  shared_utils.plot.set_ylims( axs, [-lim, lim] );
end
if ( ~isempty(custom_lims) )
  shared_utils.plot.set_ylims( axs, custom_lims );
end

plots.onelegend( gcf, 'last' );

pvals = cellfun( @(x) x.Coefficients.pValue(2), mdls.mdl );

if ( 1 )
  PI = plots.nest3( id, I, C );
  for i = 1:numel(PI)
    hold( axs(i), 'on' );
    for j = 1:numel(PI{i})
      if ( isempty(PI{i}{j}) ), continue; end
      ind = PI{i}{j};
      p = pvals(ind);
      if ( p < 0.01 )
        x = repmat( xs{i}(j), numel(ind), 1 );
%           scatter( axs(i), x, mdls.(var_name)(ind) + sign(mdls.(var_name)(ind)) * 0.005, 'k*' );
%         text( axs(i), x-0.125, mdls.(var_name)(ind) + sign(mdls.(var_name)(ind)) * 0.005, '***' );
        text( axs(i), x-0.125*0.25, mdls.(var_name)(ind) + sign(mdls.(var_name)(ind)) * 0.005, '*' );
      end
    end
  end
end  
  
set( gcf, 'renderer', 'painters' );
style_bar_plots( axs, 'no_error_lines', 1, 'prefer_valence_coloring', 1 );
% set(get(hA, 'XAxis'), 'Visible', 'off');
% set(get(hA, 'YAxis'), 'Visible', 'off');

if ( do_save )
  fname = sprintf( 'bar_%s', var_name );
  if ( include_behavior ), fname = sprintf( '%s_behav', fname ); end
  save_p = fullfile( data_root, 'plots', dsp3.datedir, 'sequence' );
  shared_utils.io.require_dir( save_p );
%   shared_utils.plot.fullscreen( gcf );
  shared_utils.plot.save_fig( gcf, fullfile(save_p, fname), {'png', 'fig'}, false );
  exportgraphics( gcf, fullfile(save_p, sprintf('%s.eps', fname)), 'ContentType', 'vector' );
  
  print( '-depsc', fullfile(save_p, sprintf('%s.eps', fname)) );
end

%%  plot (notional) lines

var_name = 'beta';
% match_lims = false;
match_lims = ismember( var_name, ["r2", "corr_r", "beta"] );
equal_lims = true;
custom_lims = [-0.09, 0.09];
do_save = false;
include_behavior = true;

mask = ...
  (mdls.model == 'sc_eval') | ...
  (mdls.model == 'pca_nc_80_eval') | ...
  (mdls.model == 'sc_n_pc_80_eval') | ...
  (mdls.src_model == 'behavior') | ...
  (mdls.model == 'image');

mask = mask & ~contains(mdls.dataset_name, 'phase_');
mask = mask & mdls.layer ~= "resnet_output";
mask = mask & mdls.model == 'sc_eval';
mask = mask & contains(mdls.layer, 'resnet');
mask = mask & mdls.src_model ~= 'd3dfr';
mask = mask & (...
  mdls.src_model ~= 'behavior' | (...
  mdls.src_model == 'behavior' & mdls.layer == 'resnet_layer1' & mdls.error_metric == 'distinctiveness'));

if ( 1 )  % highlight best model/layer combos ("best") based on correlation w/ sequences
  mask = mask & (...
    (mdls.error_metric == 'error' & mdls.src_model == 'arcface_recog' & mdls.layer == 'resnet_layer2') | ...
    (mdls.error_metric == 'distinctiveness' & mdls.src_model == 'arcface_recog' & mdls.layer == 'resnet_layer2') | ...
    (mdls.error_metric == 'error' & mdls.src_model == 'resnet_image_embedding' & mdls.layer == 'resnet_layer1') | ...
    (mdls.error_metric == 'distinctiveness' & mdls.src_model == 'resnet_image_embedding' & mdls.layer == 'resnet_layer2') | ...
    (mdls.src_model == 'behavior') ...
  );
end

if ( ~include_behavior ), mask = mask & mdls.src_model ~= 'behavior'; end

line_betas = mdls.beta(mask);
line_intercepts = cellfun( @(x) x.Coefficients.Estimate(1), mdls.mdl(mask) );
line_mat = [1, 12] .* line_betas + line_intercepts;

figure(1); clf;

[I, id, C] = rowsets( 2, mdls(mask, :) ...
  , {'src_model', 'model', 'error_metric', 'layer'}, {'valence'} ...
  , 'to_string', true ...
);
C = plots.strip_underscore( C );
C = rename_models_for_plotting( C );

[PI, PL] = plots.nest2( id, I, C );
[axs, hs] = plots.simplest_linesets( [1, 12], line_mat, PI, PL ...
  , 'error_func', @(x) nan(1, size(x, 2)) );
cellfun( @(x) set(x, 'linewidth', 2), hs );
shared_utils.plot.match_ylims( axs );
xlim( axs, [0, 13] );
style_line_plots( axs );

%%  plot slopes and means as f(prctile)

do_save = true;
[fig_I, fig_C] = findeach( mdls, {'src_model', 'model', 'layer', 'error_metric'} );

for i = 1:numel(fig_I)
fi = fig_I{i};
figure(1); clf;
var_name = 'mean';
[I, id, C] = rowsets( 3, mdls ...
  , {'op', 'model', 'src_model', 'layer', 'error_metric'}, 'prctile', 'valence' ...
  , 'to_string', 1, 'mask', fi );
C = plots.strip_underscore( C );
[~, ord] = sort( double(string(C(:, 2))) );
[I, id, C] = rowref_many( ord, I, id, C );
axs = plots.simplest_barsets( mdls.(var_name), I, id, C );
shared_utils.plot.match_ylims( axs );
ylabel( axs(1), var_name );

if ( 1 )
  max_lim = max( arrayfun(@(x) max(abs(get(x, 'ylim'))), axs) );
  ylim( axs, [-max_lim, max_lim] );
end

if ( contains(var_name, 'corr_r') )
  ylim( axs, [-0.6, 0.6] );
end

% error( 'xx' );

if ( do_save )
  subdir = strjoin( fig_C{i, {'error_metric'}}, '__' );
  mdl_info = strjoin( fig_C{i, {'src_model', 'model', 'layer'}}, '_' );
  save_p = fullfile( data_root, 'plots', dsp3.datedir, 'sequence', char(subdir), var_name );
  fname = sprintf( '%s_sequence_metric_as_f_seq_mean', mdl_info );
  shared_utils.io.require_dir( save_p );
  shared_utils.plot.fullscreen( gcf );
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

[search, ~, ic] = unique( metrics(:, {'model', 'layer', 'src_model', 'dataset_name'}) );
ic = groupi( ic );
z_err_var = err_var;
for i = 1:numel(ic)
  z_err_var(ic{i}) = zscore( z_err_var(ic{i}) );
end

seq_err_tbl = cell( rows(search), 1 );
parfor i = 1:rows(search)
  fprintf( '\n %d of %d', i, rows(search) );
  
  do_search = @(ident) find( ...
      metrics.identifier == ident & ...
      metrics.model == search.model(i) & ...
      metrics.layer == search.layer(i) & ...
      metrics.src_model == search.src_model(i) & ...
      metrics.dataset_name == search.dataset_name(i) ...
    );
  
  metric_diffs = nan( size(est_ident) );
  metric_ests = nan( size(est_ident) );
  
  z_metric_diffs = nan( size(est_ident) );
  z_metric_ests = nan( size(est_ident) );
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
    metric_ests(j) = nanmean( match_errs );
    
    z_match_errs = z_err_var(match_others);
    z_metric_diffs(j) = nanmean( z_match_errs ) - z_err_var(match_est);
    z_metric_ests(j) = nanmean( z_match_errs );
  end
  
  rep_t = repmat( search(i, :), rows(est_ident), 1 );
  va = {'model_estimate_diff', 'model_estimate', 'z_model_estimate_diff', 'z_model_estimate'};
  t = [ table(metric_diffs, metric_ests, z_metric_diffs, z_metric_ests ...
    , 'va', va), rep_t, seq_tbl ];
  seq_err_tbl{i} = t;
end

seq_err_tbl = vertcat( seq_err_tbl{:} );

end

function va = setdiff_var_names(a, b)
va = setdiff( a.Properties.VariableNames, b.Properties.VariableNames );
end

function layers = clean_layer_names(layers)

rep_ind = ismember( layers, compose("layer%d", 1:4) );
layers(rep_ind) = compose("resnet_%s", layers(rep_ind));

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

src_model_names = ["resnet_image_embedding", "arcface_recog", "d3dfr"];
% "resnet_output"
layer_name_sets = {
  compose("layer%d", 1:4) ... % resnet_image_embedding
  compose("layer%d", 1:4) ... % arcface_recog
  , [compose("resnet_layer%d", 1:4), "ReconNetWrapper_output", "ReconNetWrapper_output_identity_expression"] ... % d3dfr
};

variants = repmat( ...
  "valid_expression_balanced_var_subsample_1000", numel(src_model_names), 1 );

if ( 1 )
variants = [ variants; repmat( ...
  "valid_expression_balanced_phase_scrambled_var_subsample_1000", numel(src_model_names), 1) ];
src_model_names = repmat( src_model_names, 1, 2 );
layer_name_sets = repmat( layer_name_sets, 1, 2 );
end

% layer_names = ;
% ds_name = "d3dfr/valid_var_subsample";  % original
% 
% ds_name = "d3dfr/valid_expression_balanced_var_subsample_1000";

% ds_name = "d3dfr/valid";
% layer_names = ["ReconNetWrapper_output", "resnet_output"];

eval_dir_names = ["sc_n_pc_80_eval", "sc_eval", "pca_nc_80_eval"];
eval_dir_names = ["sc_eval"];

all_metrics = table();
all_train_metrics = table();

for idx = 1:numel(src_model_names)
  ds_name = fullfile( src_model_names(idx), variants(idx) );
  layer_names = layer_name_sets{idx};
  src_model_name = src_model_names(idx);
  
  for i = 1:numel(eval_dir_names)

    lns = layer_names;
    
    eval_dirs = fullfile( data_root, eval_dir_names(i), ds_name, lns );
    miss_layers = arrayfun( @(x) exist(x, 'dir') == 0, eval_dirs );
    eval_dirs(miss_layers) = [];
    lns(miss_layers) = [];
    
    if ( isempty(eval_dirs) ), continue; end
    
    % load activations from source layer
    act_lns = lns;
    [~, lb] = ismember( 'ReconNetWrapper_output_identity_expression', lns );
    for j = 1:numel(lb)
      if ( lb(j) > 0 )
        act_lns(lb(j)) = 'ReconNetWrapper_output';
      end
    end
    
    act_files = fullfile( data_root, "activations", ds_name, compose("%s.h5", act_lns) );
    distinct_files = fullfile( data_root, "distinctiveness", ds_name, compose("%s.mat", lns) );

    eval_cps = arrayfun( @load, fullfile(eval_dirs, "cp.mat"), 'un', 0 );
    eval_errs = cate1( cellfun(@(x) x.error(:), eval_cps, 'un', 0) );

    to_str = @(x) columnize(deblank(string(x)));
    idents = cate1(arrayfun(@(x) to_str(h5read(char(x), '/identifiers')), act_files, 'un', 0));
    layers = arrayfun( @(x) to_str(h5read(char(x), '/layers')), act_files, 'un', 0 );
    layers = cate1( arrayfun(@(i) repmat(lns(i), size(layers{i})), 1:numel(layers), 'un', 0) );
    src_model = repmat( src_model_name, numel(eval_errs), 1 );

    distincts = cate1( cellfun(@shared_utils.io.fload, distinct_files, 'un', 0) );

    ds_names = repmat( string(ds_name), numel(distincts), 1 );
    metrics = table( ...
      eval_errs, distincts, idents, layers, src_model, ds_names, 'va' ...
      , {'error', 'distinctiveness', 'identifier', 'layer', 'src_model', 'dataset_name'} );
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
    train_cp_fs = fullfile( train_cp_dirs, "cp.mat" );
    % @NOTE: Changed 02/02/24
    ok_train_cps = arrayfun( @(x) exist(x, 'file') > 0, train_cp_fs );
    train_cps = arrayfun( @load, train_cp_fs(ok_train_cps), 'un', 0 );
    [~, layers] = arrayfun( @fileparts, train_cp_dirs );

    train_metrics = table( ...
        train_cps(:) ...
      , columnize(eval_cps(ok_train_cps)) ...
      , repmat(eval_dir_names(i), numel(train_cps), 1) ...
      , columnize(layers(ok_train_cps)) ...
      , repmat(src_model_name, numel(train_cps), 1) ...
      , 'va', {'data', 'valid_data', 'model', 'layer', 'src_model'} ...
    );

    all_train_metrics = [ all_train_metrics; train_metrics ];
  end
end

metrics = all_metrics;
train_metrics = all_train_metrics;
train_metrics.batch_error = cellfun( @(x) x.batch_error(:)', train_metrics.data, 'un', 0 );

metrics.layer = clean_layer_names( metrics.layer );
train_metrics.layer = clean_layer_names( train_metrics.layer );

end

function mdls = relabel_image_reconstruction_error_metrics(mdls)

relabeli = ismember(...
    mdls.error_metric ...
  , compose("image_recon_error_resnet_image_embedding_layer%d", 1:4));
relabel_layers = strrep( ...
  mdls.error_metric(relabeli) ...
  , 'image_recon_error_resnet_image_embedding_', 'resnet_' );
mdls.error_metric(relabeli) = 'image_recon_error';
mdls.layer(relabeli) = relabel_layers;

relabeli = contains( mdls.error_metric, 'image_recon_error_pca_n_pc' );
mdls.layer(relabeli) = mdls.error_metric(relabeli);
mdls.error_metric(relabeli) = 'image_recon_error';

end

function L = rename_models_for_plotting(L)

L = strrep( L, 'd3dfr', 'EIG' );
L = strrep( L, 'arcface recog', 'FaceDeepNet' );
L = strrep( L, 'resnet image embedding', 'ImageNet' );
L = strrep( L, 'sc eval', 'Sparse Coding' );

end