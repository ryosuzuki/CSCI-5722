function [ new_img ] = gaussian_filter(current_img, sigma)

  kernel_size = 2 * ceil(2 * sigma) + 1;
  index = -floor(kernel_size/2) : floor(kernel_size/2);
  [a, b] = meshgrid(index, index);
  kernel = exp( -(a.^2 + b.^2) / (2*sigma.^2) );
  kernel = kernel / sum(kernel(:));

  new_img = zeros(size(current_img));

  pad_x = floor(kernel_size/2);
  pad_y = floor(kernel_size/2);
  pad_img = padarray(current_img, [pad_x, pad_y]);
  [n_x, n_y, n_c] = size(pad_img);

  for c = 1 : n_c
    for x = 1 + pad_x : n_x - pad_x
      for y = 1 + pad_y : n_y - pad_y
        sub_img = double(pad_img(x-pad_x:x+pad_x, y-pad_y:y+pad_y, c)) .* kernel;
        new_img(x-pad_x, y-pad_y, c) = sum(sub_img(:));
      end
    end
  end

end