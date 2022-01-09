function result = OnehotEncoding(y, n_labels)
    result = zeros(length(y), n_labels);

    for w = 1:1:length(y)
        result(w, y(w)) = 1.0;
    end

end
