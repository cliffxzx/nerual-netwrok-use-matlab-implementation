classdef Softmax < handle

    properties
        out
    end

    methods

        function y = forward(self, x)
            exps = exp(x - max(x));
            self.out = exps ./ sum(exps);
            y = self.out;
        end

        function dy = backward(self, differential_out)
            dy = differential_out;
        end

    end

end
