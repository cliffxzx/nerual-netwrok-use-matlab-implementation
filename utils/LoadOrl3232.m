function [features, labels] = LoadOrl3232(path)
    features = [];
    labels = [];

    types = dir([path '*']);

    for w = 1:length(types)
        samples = dir([types(w).folder '/' types(w).name '/*.bmp']);

        for w1 = 1:length(samples)
            sample = imread([samples(w1).folder '/' samples(w1).name]);
            features(end + 1, :) = sample(:);
            labels(end + 1, :) = w;
        end

    end

    labels = OnehotEncoding(labels, length(types));
end
