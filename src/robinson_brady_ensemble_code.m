function res = robinson_brady_ensemble_code(ims, sequence, sim_func, dp, sig_noise, its)

seq_len = numel( sequence );
num_expressions = numel( ims );

res = nan( its, 1 );

for idx = 1:its

dists = nan( num_expressions, 1 );
for i = 1:num_expressions
  % image with rating i
  ref_im = ims{i};
  
  fi = 0;
  for j = 1:seq_len
    % image in sequence position j with rating sequence(j)
    seq_im = ims{sequence(j)};
    fi = fi + sim_func( ref_im, seq_im ) * dp;
  end
  
  dists(i) = fi + randn * sig_noise;
end

[~, res(idx)] = max( dists );

end

res = mean( res );

end