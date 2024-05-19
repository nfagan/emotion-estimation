function style_bar_plots(axs, varargin)

%%

defaults = struct();
defaults.no_error_lines = false;
defaults.style = plot_style();
defaults.legend_location = [];
defaults.prefer_valence_coloring = false;
defaults.distinctiveness_opacity = 1;

params = shared_utils.general.parsestruct( defaults, varargin );
s = params.style;

h = findobj( axs, 'type', 'bar' );
disp_names = arrayfun( @(x) string(x.DisplayName), h );

% is_neg = disp_names == 'negative';
% is_pos = disp_names == 'positive';

is_neg = contains( disp_names, 'negative', 'IgnoreCase', 1 );
is_pos = contains( disp_names, 'positive', 'IgnoreCase', 1 );

is_recon = contains( disp_names, "reconstruction error" );
is_distinct = contains( disp_names, "distinctiveness" );

do = params.distinctiveness_opacity;
if ( do < 1 )
  set( h(is_distinct), 'FaceAlpha', do );
end

is_orig = contains( disp_names, "original" );
is_phase_scram = contains( disp_names, "phase-scrambled" );

if ( params.prefer_valence_coloring )
  is_val = is_neg | is_pos;
  is_recon = is_recon & ~is_val;
  is_distinct = is_distinct & ~is_val;
end

set( h(is_neg), 'FaceColor', s.color_negative );
set( h(is_pos), 'FaceColor', s.color_positive );
set( h(is_recon), 'FaceColor', s.color_recon_error );
set( h(is_distinct), 'FaceColor', s.color_distinctiveness );
set( h(is_orig), 'FaceColor', s.color_non_phase_scrambled );
set( h(is_phase_scram), 'FaceColor', s.color_phase_scrambled );
set( h, 'edgecolor', 'none' );

if ( params.no_error_lines )
  hl = findobj( axs, 'type', 'line' );
  delete( hl );
end

for i = 1:numel(h)
  dn = get( h(i), 'displayname' );
  dn = strrep( dn, ' | ImageNet original', ' (original images)' );
  dn = strrep( dn, ' | ImageNet phase-scrambled', ' (phase-scrambled images)' );
  set( h(i), 'displayname', dn );
end

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

  if ( ~isempty(params.legend_location) )
    set( leg, 'location', params.legend_location );
  end
end

arrayfun( @(x) set(get(x, 'XAxis'), 'FontSize', s.axis_font_size), axs );
arrayfun( @(x) set(get(x, 'YAxis'), 'FontSize', s.axis_font_size), axs );

for i = 1:numel(axs)
  txt = get( axs(i), 'xticklabel' );
  txt = cellfun( @(x) strrep(x, 'resnet ', ''), txt, 'un', 0 );
  txt = cellfun( @(x) strrep(x, 'layer', 'layer '), txt, 'un', 0 );
  set( axs(i), 'xticklabel', txt );
end

for i = 1:numel(axs)
  txt = get( get(axs(i), 'title'), 'string' );
  txt = strrep( txt, 'Sparse Coding | ', '' );
  txt = strrep( txt, ' | Sparse Coding', '' );
  set( get(axs(i), 'title'), 'string', txt );  
end

end