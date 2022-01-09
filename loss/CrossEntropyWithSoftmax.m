classdef CrossEntropyWithSoftmax < handle

    properties
        y
        t
    end

    methods

        function self = CrossEntropyWithSoftmax(y, t)
            self.y = y;
            [~, argmax] = max(t, [], 2);
            self.t = t;

            self.y(self.y <= 0) = 1e-8;
        end

        function ret = forward(self)
            ret = -sum(log(self.y) .* self.t);
        end

        function ret = backward(self)
            ret = self.y - self.t;
        end

    end

end

function ret = softmax(x)
    e = exp(x - max(x));
    ret = e ./ sum(e);
end
