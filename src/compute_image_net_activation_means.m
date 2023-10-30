data_root = 'D:\data\changlab\ilker_collab';

act_ps = shared_utils.io.find( fullfile(data_root, 'activations/image_net'), '.h5' );

to_str = @(x) deblank(string(x(:)));

ts = cell( numel(act_ps), 1 );
for i = 1:numel(act_ps)
  p = act_ps{i};
  acts = h5read( p, '/activations' );
  layers = to_str( h5read( p, '/layers') );
  splits = to_str( h5read( p, '/splits') );
  idents = to_str( h5read(p, '/identifiers') );
  ts{i} = table( acts', layers, splits, idents ...
    , 'va', {'activation', 'layer', 'split', 'identifier'} );
end

t = vertcat( ts{:} );
[I, C] = findeach( t, {'layer', 'split'} );
mus = cellfun( @(x) nanmean(abs(columnize(t.activation(x, :)))), I )