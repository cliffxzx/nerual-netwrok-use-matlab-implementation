classdef Sigmoid < handle

    properties
        out
    end

    methods
        function y = forward(self, x)
            self.out = 1 ./ (1 + exp(-x));
            y = self.out;
        end

        function dy = backward(self, differential_out)
            dy = differential_out .* (1. - self.out) .* self.out;
        end

    end

end
