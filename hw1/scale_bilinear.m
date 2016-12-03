function [ new_img ] = scale_bilinear(current_img, sigma)

  [n_x, n_y, n_c] = size(current_img);
  new_img = zeros(factor * n_x, factor * n_y, n_c);

  for x = 1 : factor * n_x
    for y = 1 : factor * n_y
      x_1 = floor(x/factor)
      x_1 = floor(y/factor)
      x_2 = ceil(x/factor)
      y_2 = ceil(y/factor)
      new_img(x, y, :) = sum(current_img(x_1:x_2, y_1:y_2, :)) / 4
    end
  end

end