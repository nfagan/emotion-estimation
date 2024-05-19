figure(1); clf;

axs = plots.panels( [1, 1] );
hold( axs, 'on' );
hs = [];
hs(end+1) = plot( axs(1), linspace(0.2, 0.3, 3)+0.05, 'DisplayName', 'Low RE, positive' );
hs(end+1) = plot( axs(1), linspace(0.2*1.15, 0.3*1.15, 3)+0.05, 'DisplayName', 'Low RE, negative' );

hs(end+1) = plot( axs(1), [0.3, 0.5, 0.56], 'b', 'DisplayName', 'High RE, positive', 'LineStyle', '--' );
hs(end+1) = plot( axs(1), [0.35, 0.63, 0.65], 'r', 'DisplayName', 'High RE, negative', 'LineStyle', '--' );

for i = 1:numel(axs)
  set( axs(i), 'xtick', 1:3 );
  set( axs(i), 'XTickLabel', {'Short', 'Medium', 'Long'} );
  xlim( axs(i), [0, 4] );
end

ylim( axs, [0, 1] );
set( hs, 'linewidth', 2 );
legend( hs );
ylabel( axs(1), 'Sequence amplification effect (au)' );
title( sprintf([...
  'Hypothesized sequence amplification effect as a function of image presentation time\n' ...
  , ' and recon. error level']) );