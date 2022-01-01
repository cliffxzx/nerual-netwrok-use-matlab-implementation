classdef MSE < handle

    properties
        diff
    end

    methods

        function self = MSE(y, t)
            self.diff = y - t;
        end

        function ret = forward(self)
            ret = sum(mean(self.diff.^2));
        end

        function ret = backward(self)
            ret = 2 * mean(self.diff, 2) / length(self.diff);
        end

    end

end
