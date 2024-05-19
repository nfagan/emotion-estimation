function style_line_plots(axs, varargin)

defaults = struct();
defaults.style = plot_style();
defaults.style.position = [104 227 1421 643];

params = shared_utils.general.parsestruct( defaults, varargin );

s = params.style;

h = [ findobj(axs, 'type', 'errorbar'); findobj(axs, 'type', 'line') ];
disp_names = arrayfun( @(x) string(x.DisplayName), h );
is_neg = contains( disp_names, 'negative' );
is_pos = contains( disp_names, 'positive' );

set( h(is_neg), 'color', s.color_negative );
set( h(is_pos), 'color', s.color_positive );

fig = axs(1).Parent;
set( fig, 'units', 'pixels', 'position', s.position );
set( fig, 'color', s.figure_color );

fn = s.font_name;
set( axs, 'fontname', fn );
set( axs, 'xticklabelrotation', 0 );

legs = findobj( axs(1).Parent, 'type', 'legend' );
for i = 1:numel(legs)
  leg = legs(i);
  leg.EdgeColor = 'none';
  set( leg, 'FontName', fn );
end

arrayfun( @(x) set(get(x, 'XAxis'), 'FontSize', s.axis_font_size), axs );
arrayfun( @(x) set(get(x, 'YAxis'), 'FontSize', s.axis_font_size), axs );

for i = 1:numel(axs)
  txt = get( get(axs(i), 'title'), 'string' );
  txt = strrep( txt, 'resnet layer', 'layer ' );
  set( get(axs(i), 'title'), 'string', txt );  
end

end