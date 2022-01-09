clc;
clear all;
close all;
format shortG;
format compact;

addpath('utils');
addpath('models');
[x, y] = LoadOrl3232('dataset/orl3232/');

model = Classification(1024, 41);
train_loss = [];
val_loss = [];
epoch = 700;

randn('seed', 42);
p = randperm(length(y));

for w = 1:1:epoch
    train_loss = [train_loss mean(model.fit(x(p(1:200), :), y(p(1:200), :)))];
    val_loss = [val_loss mean(model.fit(x(p(200 + 1:end), :), y(p(200 + 1:end), :)))];

    fprintf('[it] %d [accuracy] %.4f [loss] train: %.4f, val: %.4f\n', w, model.accuracy(x, y), train_loss(end), val_loss(end))
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
ylabel('Croess Entropy');
xlabel('Epoch');
pause;
