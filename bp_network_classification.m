% =========================================================================
% Problem 2: 6-D 8-Class BP Network Classification (Strict Parameters)
% =========================================================================
%% 1. Load and Prepare Data
disp('Preparing data...');
% 1a. Scale inputs to a symmetric [-1, 1] range (The good practice from 3.6!)
% This is much better than z-score for this specific classic BP architecture.
x_min = min(x);
x_max = max(x);
x_scaled = 2 * ((x - x_min) ./ (x_max - x_min)) - 1;
% 1b. Flatten labels and One-Hot Encode (using 0.1 and 0.9 targets)
labels_1d = label(:);
num_classes = 8;
num_samples = length(labels_1d);
targets_one_hot = ones(num_samples, num_classes) * 0.1; % Base is 0.1
for idx = 1:num_samples
    class_val = labels_1d(idx);
if class_val >= 1 && class_val <= num_classes
        targets_one_hot(idx, class_val) = 0.9; % Active class is 0.9
end
end
% 1c. Stratified BALANCED Split (The key to breaking the 12.5% plateau)
num_train_per_class = 200;
train_idx = [];
test_idx = [];
rng(42); % Seed for reproducibility
for c = 1:num_classes
% Find all pixel indices belonging to class 'c'
    idx_for_class_c = find(labels_1d == c);
% Shuffle the pixels for this specific class
    idx_for_class_c = idx_for_class_c(randperm(length(idx_for_class_c)));
% Take 200 for training, and put the rest in the test set
    train_idx = [train_idx; idx_for_class_c(1:num_train_per_class)];
    test_idx = [test_idx; idx_for_class_c(num_train_per_class+1:end)];
end
% Shuffle the final training set so the network doesn't see all Class 1s,
% then all Class 2s, etc.
train_idx = train_idx(randperm(length(train_idx)));
% Create the final arrays expected by the training loop
train_data = x_scaled(train_idx, :);
train_labels = targets_one_hot(train_idx, :);
test_data = x_scaled(test_idx, :);
test_labels = targets_one_hot(test_idx, :);
fprintf('Balanced Training samples: %d\n', size(train_data, 1));
fprintf('Testing samples: %d\n\n', size(test_data, 1));
%% 2. STRICT Network Architecture & Parameters
input_dim = 6;
hidden_nodes = 10;   % STRICTLY 10 PEs
output_dim = 8;
learning_rate = 0.2;
momentum = 0.3;
stop_criterion_max = 0.05; % Target misclassification (1% - 5%)
%% 3. Weight Initialization
% STRICTLY drawn from U(-0.1, 0.1)
a = -0.1; b = 0.1;
W1 = a + (b-a) .* rand(hidden_nodes, input_dim + 1);
W2 = a + (b-a) .* rand(output_dim, hidden_nodes + 1);
dW1_prev = zeros(size(W1));
dW2_prev = zeros(size(W2));
%% 4. Training Loop (Tracking Learning Steps)
disp('Starting on-line training...');
N_train = size(train_data, 1);
max_epochs = 100;
total_steps = 0;
learning_history = [];
for epoch = 1:max_epochs
    errors_this_epoch = 0;
% Shuffle for on-line learning
    epoch_shuffle_idx = randperm(N_train);
    train_data_shuffled = train_data(epoch_shuffle_idx, :);
    train_labels_shuffled = train_labels(epoch_shuffle_idx, :);
for i = 1:N_train
        total_steps = total_steps + 1; % Track the 20k-30k steps
% --- FORWARD PASS ---
        curr_x = [1, train_data_shuffled(i, :)]'; % Bias in input layer
% Hidden layer (Tanh for symmetric [-1, 1] activation)
        v1 = W1 * curr_x;
        y1 = tanh(v1);
% Output layer (Sigmoid for 0 to 1 classification)
        y1_bias = [1; y1]; % Bias in hidden layer
        v2 = W2 * y1_bias;
        y2 = 1 ./ (1 + exp(-v2));
% --- ERROR CALCULATION ---
        target = train_labels_shuffled(i, :)';
        e = target - y2;
        [~, pred_class] = max(y2);
        [~, true_class] = max(target);
if pred_class ~= true_class
            errors_this_epoch = errors_this_epoch + 1;
end
% --- BACKWARD PASS ---
        delta2 = e .* (y2 .* (1 - y2));
        W2_no_bias = W2(:, 2:end);
        delta1 = (W2_no_bias' * delta2) .* (1 - y1.^2); % Tanh derivative
% --- WEIGHT UPDATE ---
        dW2 = (learning_rate * delta2 * y1_bias') + (momentum * dW2_prev);
        dW1 = (learning_rate * delta1 * curr_x') + (momentum * dW1_prev);
        W2 = W2 + dW2;
        W1 = W1 + dW1;
        dW2_prev = dW2;
        dW1_prev = dW1;
end
% Calculate misclassification rate for the epoch
    misclass_rate = errors_this_epoch / N_train;
    learning_history = [learning_history, misclass_rate];
    fprintf('Epoch %d (Total Steps: %d): Misclassification Rate = %.4f\n', ...
            epoch, total_steps, misclass_rate);
% Check stopping criterion
if misclass_rate <= stop_criterion_max
        fprintf('SUCCESS: Reached %.2f%% error after %d learning steps!\n', ...
                misclass_rate * 100, total_steps);
break;
end
end
%% 5. Plot Learning History
figure;
% Plotting against steps makes more sense for on-line learning
steps_per_epoch = (1:length(learning_history)) * N_train;
plot(steps_per_epoch, learning_history, '-o', 'LineWidth', 2);
title('Learning History vs. Steps');
xlabel('Learning Steps');
ylabel('Misclassification Rate');
grid on;
%% 6. Testing Phase
disp('Evaluating Test Data...');
N_test = size(test_data, 1);
test_preds = zeros(N_test, 1);
test_trues = zeros(N_test, 1);
for i = 1:N_test
    curr_x = [1, test_data(i, :)]';
    y1 = tanh(W1 * curr_x);
    y1_bias = [1; y1];
    y2 = 1 ./ (1 + exp(-(W2 * y1_bias)));
    [~, test_preds(i)] = max(y2);
    [~, test_trues(i)] = max(test_labels(i, :)');
end
test_accuracy = sum(test_preds == test_trues) / N_test;
fprintf('Final Test Accuracy: %.2f%%\n\n', test_accuracy * 100);
C_test = confusionmat(test_trues, test_preds);
disp('Test Confusion Matrix:');
disp(C_test);
%% 7. BONUS: Reconstruct the Spatial Classification Map
% To get the bonus points, we need to classify the ENTIRE image cube (x_scaled)
% and reshape it back into a 128x128 grid to visualize the spatial errors.
disp('Generating full image map for bonus visualization...');
full_image_preds = zeros(num_samples, 1);
for i = 1:num_samples
    curr_x = [1, x_scaled(i, :)]'; % FIXED: Changed x_norm to x_scaled
    y1 = tanh(W1 * curr_x);
    y1_bias = [1; y1];
    y2 = 1 ./ (1 + exp(-(W2 * y1_bias)));
    [~, full_image_preds(i)] = max(y2);
end
% Reshape the 1D predictions back into the 128x128 2D format
predicted_map_2D = reshape(full_image_preds, [128, 128]);
% Display side-by-side comparison using MATLAB's imagesc
figure;
subplot(1,2,1);
imagesc(label); % Original ground truth from workspace
title('Ground Truth Map');
axis image off;
subplot(1,2,2);
imagesc(predicted_map_2D); % Network predictions
title('Predicted Classification Map');
axis image off;
% Apply a nice colormap to distinguish the 8 classes
colormap(jet(8));
