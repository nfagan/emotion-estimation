function s = plot_style()

s = struct();
s.color_negative = [0.25, 0.5, 1];
s.color_positive = [0.25, 1, 0.5];
s.color_recon_error = [0.2824    0.6275    0.7569];
s.color_distinctiveness = [0.8745    0.4667    0.3294];
s.color_non_phase_scrambled = [1, 0, 1];
s.color_phase_scrambled = [0, 1, 1];
s.position = [777 264 896 641];
s.figure_color = 'w';
s.font_name = 'arial';
s.axis_font_size = 10;

end