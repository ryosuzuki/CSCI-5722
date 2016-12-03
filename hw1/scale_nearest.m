function [ new_img ] = scale_nearest(current_img, factor)

  [n_x, n_y, n_c] = size(current_img);
  new_img = zeros(factor * n_x, factor * n_y, n_c);

  for x = 1 : factor * n_x
    for y = 1 : factor * n_y
      f_x = round((x-1)*(n_x-1) / (factor*n_x-1) + 1);
      f_y = round((y-1)*(n_y-1) / (factor*n_y-1) + 1);
      new_img(x, y, :) = current_img(f_x, f_y, :);
    end
  end

end