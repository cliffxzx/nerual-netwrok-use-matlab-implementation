classdef RMSE < handle

    properties
        y
        mse
        out
    end

    methods

        function self = RMSE(y, t)
            self.mse = MSE(y, t);
            self.out = sqrt(self.mse.forward());
        end

        function ret = forward(self)
            ret = self.out;
        end

        function ret = backward(self)
            ret = ((2 * self.out).^ - 1) .* self.mse.backward();
        end

    end

end
