function hist = dct_coef_hist(coef_arrays, quant_table, channel, Nc, BT)
%   This function extracts histogram of DCT coefficients
%   Input:
%       coeff_arrays: 1x3 cell array of coefficients
%       Nc: number of analyzed DCT coeff in each 8x8 blocks
%       BT: the limit of histogram ==> number of bins = 2*BT + 1
%   Output:
%       hist: the histogram of DCT coefficients
    [h,w] = size(coef_arrays{channel});
    
    nblks = (floor(h/8)-1)*(floor(w/8)-1);
    %error here must be the variable channel
    coefs = coef_arrays{1};
    hist = zeros(Nc, (2*BT+1));
    
    zigzag_inds = zigzag(reshape(1:64, [8 8])');
    zigzag_inds = zigzag_inds(2:Nc+1);
    
    for b = 0:nblks-1
        i = 8*(floor(b / (floor(w/8)-1))) + 1;
        j = 8*mod(b, (floor(w/8)-1)) + 1;

        block = coefs(i:i+7, j:j+7);
        block = block(zigzag_inds);
        quant_values = quant_table{channel}(zigzag_inds);
        for n = 1:Nc
            v = block(n) * quant_values(n);
            if v > BT
                hist(n, end) = hist(n, end) + 1;
            elseif v < -BT
                hist(n, 1) = hist(n, 1) + 1;
            else
                hist(n, v+BT+1) = hist(n, v+BT+1) + 1;
            end
        end
    end
    hist = hist ./ nblks;
end

