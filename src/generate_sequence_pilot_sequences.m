%%  run after loading data from analyze_reconstruction

sel_metrics = ...
  metrics.src_model == 'arcface_recog' & ...
  metrics.layer == 'resnet_layer2' & ...
  metrics.dataset_name == 'arcface_recog\valid_expression_balanced_var_subsample_1000';

subset_metrics = metrics(sel_metrics, :);

assert( size(unique(subset_metrics(:, 'identifier')), 1) == size(subset_metrics, 1) ...
  , 'Expected one unique set of images.' );

[I, C] = findeach( subset_metrics, {'subject', 'valence'} );
C.lo_ind = cell( numel(I), 1 );
C.hi_ind = cell( size(C.lo_ind) );

for i = 1:numel(I)
  err = subset_metrics.error(I{i});
  % 30th and 70th percentile
  quants = prctile( err, [30, 70] );
  lo_si = I{i}(err < quants(1));
  hi_si = I{i}(err >= quants(2));
  assert( numel(lo_si) == numel(hi_si) )
  C.lo_ind{i} = lo_si;
  C.hi_ind{i} = hi_si;
end

C.lo_idents = cellfun( @(x) subset_metrics.identifier(x), C.lo_ind, 'un', 0 );
C.hi_idents = cellfun( @(x) subset_metrics.identifier(x), C.hi_ind, 'un', 0 );

%%

seqs_per_cond = 8;
present_times = [ "short", "medium", "long" ];

make_url = @(im) compose("dist/img_goldenberg_big/%s.jpg", im);

conds = table();

for i = 1:size(C, 1)
  % for each condition ...
  for j = 1:seqs_per_cond
    % for each sequence ...
    ims_per_seq = randi( [2, 12] );
    lo_ims = randsample( C.lo_idents{i}, ims_per_seq );
    hi_ims = randsample( C.hi_idents{i}, ims_per_seq );
    
    for k = 1:numel(present_times)
      make_cond = @(ims, quant, present_time) ...
        [ C(i, :), table({make_url(ims)}, quant, present_time ...
        , 'va', {'sequence', 'error_quantile', 'present_time'}) ];

      lo_cond = make_cond( lo_ims, "low", present_times(k) );
      hi_cond = make_cond( hi_ims, "high", present_times(k) );

      conds = [ conds; lo_cond; hi_cond ];
    end
  end
end

conds = conds(randperm(rows(conds)), :);

%%

conds_js = jsonencode( table2struct(conds) );
clipboard( 'copy', conds_js );

%%

im_info_js = jsonencode( table2struct(im_info) );
clipboard( 'copy', im_info_js );
