function qf = quality_factor(file_path)
%   This function read the quantization tables and return interpolated
%   quality factors
    
    std_luma_quant_table = ...
    [ 16,  11,  10,  16,  24,  40,  51,  61;
      12,  12,  14,  19,  26,  58,  60,  55;
      14,  13,  16,  24,  40,  57,  69,  56;
      14,  17,  22,  29,  51,  87,  80,  62;
      18,  22,  37,  56,  68, 109, 103,  77;
      24,  35,  55,  64,  81, 104, 113,  92;
      49,  64,  78,  87, 103, 121, 120, 101;
      72,  92,  95,  98, 112, 100, 103,  99];
 
    std_chroma_quant_table = ...
    [ 17,  18,  24,  47,  99,  99,  99,  99;
      18,  21,  26,  66,  99,  99,  99,  99;
      24,  26,  56,  99,  99,  99,  99,  99;
      47,  66,  99,  99,  99,  99,  99,  99;
      99,  99,  99,  99,  99,  99,  99,  99;
      99,  99,  99,  99,  99,  99,  99,  99;
      99,  99,  99,  99,  99,  99,  99,  99;
      99,  99,  99,  99,  99,  99,  99,  99];
 
    jobj=jpeg_read(file_path);
    qf = zeros(1, size(jobj.quant_tables, 2));
    for c = 1:size(jobj.quant_tables, 2)
        qt = jobj.quant_tables(c);
        qt = cell2mat(qt);
        
        if c == 1
            std_quant_table = std_luma_quant_table;
        else
            std_quant_table = std_chroma_quant_table;
        end
        
        ScFact = (sum(sum((100*qt - 50) ./ std_quant_table)) / 64);
        if ScFact == 1
            qf(c) = 100;
        elseif ScFact >= 100
            qf(c) = floor(5000 / ScFact);
        else
            qf(c) = floor((200 - ScFact)/2);
        end
    end
end

