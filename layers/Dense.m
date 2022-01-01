classdef Dense < handle

    properties
        x
        learning_rate
        bias
        weight
        diffiential_bias
        diffiential_weight
    end

    methods

        function self = Dense(input_size, output_size, learning_rate)
            self.learning_rate = learning_rate;
            randn('seed', 42);
            self.weight = randn(input_size, output_size);
            self.bias = zeros(1, output_size);
        end

        function y = forward(self, x)
            self.x = x;
            y = self.x * self.weight + self.bias;
        end

        function dy = backward(self, differential_out)
            % df / dx = df / d(dense) * d(dense) / dx
            dy = differential_out * self.weight';

            self.diffiential_weight = self.x' * differential_out;
            self.diffiential_bias = sum(differential_out);

            self.weight = self.weight - self.learning_rate .* self.diffiential_weight;
            self.bias = self.bias - self.learning_rate .* self.diffiential_bias;
        end

    end

end
