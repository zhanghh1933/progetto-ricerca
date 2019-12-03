%file dct_coef_hist.m error on line 13, there is a 1, it must be the
%variable channel
%come lavorano le matrici di matlab ???
%se faccio matrice(int) cosa vado a prendere

%this function work only on Luminance Y channel

function hist = dct_hist_extraction(img, Nc, BT)
    ycbcr = rgb2ycbcr(img);
    Y = ycbcr(: , : , 1);
    zigzag_inds = zigzag(reshape(1:64, [8 8])');
    hist = zeros(Nc, (2*BT+1));
    zigzag_inds = zigzag_inds(2:Nc+1);
    
    %here must convert 
    
    %taking the Y channel we move the 0 to 127, so easy -127 to all pixel
    Y = int8(Y) - 128;
    [h,w] = size(Y);
    nblks = (floor(h/8)-1)*(floor(w/8)-1);
    
    
    for b = 0:nblks-1
        i = 8*(floor(b / (floor(w/8)-1))) + 1;
        j = 8*mod(b, (floor(w/8)-1)) + 1;
        
        %extracting the block we are working
        block = Y(i:i+7, j:j+7);
        %execute the DCT of the block
        blockDCT = dct2(block);
        %only the Nc value of the block
        %because Matlab is collum major we need to translate the matrix
        blockNc = blockDCT';
        blockNc = blockNc(zigzag_inds);
        
        for n = 1:Nc
            v = blockNc(n);
            v = round(v);
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