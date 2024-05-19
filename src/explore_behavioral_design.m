%%

targ_valence = 'negative';

mdl1 = metrics.layer == 'resnet_layer3' & metrics.model == 'sc_eval' & metrics.src_model == 'arcface_recog';
mdl2 = metrics.layer == 'resnet_layer3' & metrics.model == 'sc_eval' & metrics.src_model == 'resnet_image_embedding';

mdl1 = mdl1 & metrics.valence == targ_valence;
mdl2 = mdl2 & metrics.valence == targ_valence;

tiles = [30, 70];
[im_inds1, set_labels1] = make_metric_grid( metrics.error, metrics.distinctiveness, tiles, mdl1 );
tbl1 = cate1( cellfun(@(x) metrics(x, :), im_inds1, 'un', 0) );
tbl1.set = cate1( set_labels1 );

[im_inds2, set_labels2] = make_metric_grid( metrics.error, metrics.distinctiveness, tiles, mdl2 );
tbl2 = cate1( cellfun(@(x) metrics(x, :), im_inds2, 'un', 0) );
tbl2.set = cate1( set_labels2 );

tbl = [ tbl1; tbl2 ];

%%

rating_tiles = prctile( metrics.rating, [50, 50] );
lo1 = metrics.rating < rating_tiles(1);
hi1 = metrics.rating > rating_tiles(2);

emotion_part_name = { 'lo', 'hi' };
emotion_parts = { lo1, hi1 };
mdl_inds = { mdl1, mdl2 };

tbl = table();
for i = 1:numel(emotion_parts)
  for j = 1:numel(mdl_inds)
    [im_inds, set_labels] = make_metric_grid( ...
      metrics.error, metrics.distinctiveness, [30, 70], emotion_parts{i} & mdl_inds{j} );
    im_inds = cellfun( @find, im_inds, 'un', 0 );
    base_tbl = cate1( cellfun(@(x) metrics(x, :), im_inds, 'un', 0) );
    base_tbl.set = cate1( set_labels );
    base_tbl.emotion_partition(:) = string( emotion_part_name{i} );
    tbl = [ tbl; base_tbl ];
  end
end

%%

pcats = {'valence', 'layer', 'model', 'emotion_partition'};
gcats = { 'src_model' };

[I, id, C] = rowsets( 3, tbl, pcats, {'set'}, gcats, 'to_string', 1 );
C = strrep( C, 'lo1', 'lo_error' );
C = strrep( C, 'lo2', 'lo_distinct' );
C = strrep( C, 'hi1', 'hi_error' );
C = strrep( C, 'hi2', 'hi_distinct' );
C = strrep( C, 'arcface_recog', 'FaceDeepNet' );
C = strrep( C, 'resnet_image_embedding', 'ImageNet' );
C = plots.strip_underscore( C );
figure(1); clf;
axs = plots.simplest_barsets( tbl.rating, I, id, C ...
  , 'as_line_plot', 1, 'error_func', @plotlabeled.nansem, 'summary_func', @nanmean ...
  , 'panel_shape', [2, 1] );
set( axs, 'xticklabelrotation', 15 );
ylabel( axs(1), 'Mean emotion intensity of set' );
shared_utils.plot.match_ylims( axs );

%%

function [im_inds, set_labels] = make_metric_grid(metric1, metric2, tiles, mask)

tiles1 = prctile( metric1(mask), tiles );
tiles2 = prctile( metric2(mask), tiles );

lo1 = mask & metric1 < tiles1(1);
hi1 = mask & metric1 > tiles1(2);
lo2 = mask & metric2 < tiles2(1);
hi2 = mask & metric2 > tiles2(2);

lo_lo = lo1 & lo2;
lo_hi = lo1 & hi2;
hi_lo = hi1 & lo2;
hi_hi = hi1 & hi2;
im_inds = { lo_lo, lo_hi, hi_lo, hi_hi };

set_labels = cell( 1, 4 );
set_names = { 'lo1 | lo2', 'lo1 | hi2', 'hi1 | lo2', 'hi1 | hi2' };
for i = 1:4
  set_labels{i} = repmat( string(set_names{i}), sum(im_inds{i}), 1 );
end

end