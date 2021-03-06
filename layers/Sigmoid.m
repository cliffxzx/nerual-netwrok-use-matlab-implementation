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
            dy = self.out .* (1 .- self.out) .* differential_out;
        end

    end

end
