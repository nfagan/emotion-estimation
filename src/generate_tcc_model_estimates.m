% @NOTE: Run load_data from analyze_reconstruction first

im_ps = compose( "%s/%s.jpg", fullfile(data_root, 'goldenberg_faces/images'), im_info.identifier );
ims = cellfun( @imread, im_ps, 'un', 0 );
ims = cellfun( @(x) double(x) ./ 255, ims, 'un', 0 );

%%

sim_func = @(a,b) dot(a(:)/norm(a(:)),b(:)/norm(b(:)));

tcc_ests = nan( size(seq_tbl, 1), 1 );

parfor seq_ind = 1:size(seq_tbl, 1)
  
fprintf( '\n %d of %d', seq_ind, size(seq_tbl, 1) );
  
seq = seq_tbl.sequence(seq_ind, 1:seq_tbl.face_number(seq_ind));

match_im = find( ...
  im_info.subject == seq_tbl.faceIdentity{seq_ind} & ...
  im_info.valence == seq_tbl.valence{seq_ind} );
[~, ord] = sort( im_info.rating(match_im) );

match_im = match_im(ord);
subset_ims = ims(match_im);

tcc_ests(seq_ind) = robinson_brady_ensemble_code( subset_ims, seq, sim_func, 1, 0.001, 25 );

end