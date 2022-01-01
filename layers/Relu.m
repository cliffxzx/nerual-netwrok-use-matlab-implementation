classdef Relu < handle

    properties
        out
    end

    methods
        %             0, if x < 0,
        % ReLU(x) = {              }
        %             x, otherwise
        function y = forward(self, x)
            self.out = max(0, x);
            y = self.out;
        end

        %                   0, if x < 0,
        % (d/dx)ReLU(x) = {              }
        %                   1, otherwise
        function dy = backward(self, differential_out)
            dy = (self.out > 0) .* differential_out;
        end

    end

end
