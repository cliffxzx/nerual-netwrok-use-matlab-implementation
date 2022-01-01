clc;
clear all;
close all;
format shortG;
format compact;

addpath('models');

randn('seed', 42);
x = randn(600, 4);
y = 0.8 .* x(:, 1) .* x(:, 2) .* x(:, 3) .* x(:, 4) + x(:, 1).^2 + x(:, 2).^2 + x(:, 3).^3 + x(:, 4).^2 + x(:, 1) + x(:, 2) .* 0.7 - x(:, 2).^2 .* x(:, 3).^2 + 0.5 .* x(:, 1) .* x(:, 4).^2 + x(:, 4) .* x(:, 2).^3 + (-x(:, 1)) .* x(:, 2) + (x(:, 1) .* x(:, 2) .* x(:, 3) .* x(:, 4)).^3 + (x(:, 1) - x(:, 2) + x(:, 3) - x(:, 4)) + (x(:, 1) .* x(:, 4)) - (x(:, 2) .* x(:, 3)) - 2;

model = Regression(4, 1);
train_loss = [];
val_loss = [];
epoch = 700;

for w = 1:1:epoch
    train_loss = [train_loss mean(model.fit(x(1:400, :), y(1:400, :)))];
    val_loss = [val_loss mean(model.fit(x(400 + 1:end, :), y(400 + 1:end, :)))];

    fprintf('[it] %d [loss] train: %.4f, val: %.4f\n', w, train_loss(end), val_loss(end))
end

predict = model.predict(x);

figure(1);
plot(1:length(y), y, 1:length(predict), predict);
set(gcf, 'Position', [100 100 1500 400]);
legend('y', 'predict');

figure(2);
plot(1:length(train_loss), train_loss, 1:length(val_loss), val_loss);
axis([-inf inf -1 100])
legend('Traing', 'Validatation');
ylabel('RMSE');
xlabel('Epoch');
