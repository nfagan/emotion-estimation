function [f, ang_shift] = phase_scramble(f, ang_shift)

if ( nargin < 2 || isempty(ang_shift) )
  ang_shift = repmat( (rand(size(f, [1, 2])) * 2 - 1) * pi, [1, 1, 3] );
end
  
for i = 1:size(f, 3)
  ct = fft2( f(:, :, i) );
  mag = abs( ct );
  ang = angle( ct );
  ang = ang + ang_shift(:, :, i);
  ft = abs( ifft2(mag .* exp(1i * ang)) );
  f(:, :, i) = ft;
end

end