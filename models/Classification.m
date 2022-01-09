classdef Classification < handle

    properties (Access = private)
        layers_path = addpath('layers')
        loss_path = addpath('loss')
    end

    properties
        layers
        loss
        learning_rate
    end

    methods

        function self = Classification(input_size, output_size)
            self.loss = @CrossEntropyWithSoftmax;
            self.learning_rate = 0.01;

            self.layers = {
                        Dense(input_size, 80, self.learning_rate)
                        Sigmoid()
                        Dense(80, output_size, self.learning_rate)
                        Softmax()
                        };
        end

        function y = forward(self, x)
            y = x;

            for w = 1:1:length(self.layers)
                y = self.layers{w}.forward(y);
            end

        end

        function dy = backward(self, differential_out)
            dy = differential_out;

            for w = length(self.layers):-1:1
                dy = self.layers{w}.backward(dy);
            end

        end

        function y = predict(self, x)
            y = self.forward(x);
        end

        function ret = accuracy(self, x, t)
            [~, y] = max(self.predict(x), [], 2);
            [~, arg_t] = max(t, [], 2);
            ret = sum(y == arg_t) / length(y);
        end

        function loss = fit(self, x, t)
            y = self.forward(x);
            loss_fn = self.loss(y, t);
            loss = mean(loss_fn.forward());
            differential_loss = loss_fn.backward();
            self.backward(differential_loss);
        end

    end

end
