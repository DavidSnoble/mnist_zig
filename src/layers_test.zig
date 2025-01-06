const std = @import("std");
const testing = std.testing;
const layers = @import("layers.zig");

test "Layer Initialization" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var layer = try layers.Layer.init(allocator, 3, 2, .relu);
    defer layer.deinit(allocator);

    try testing.expectEqual(@as(usize, 3), layer.input_size);
    try testing.expectEqual(@as(usize, 2), layer.output_size);
    try testing.expectEqual(@as(usize, 3 * 2), layer.weights.len);
    try testing.expectEqual(@as(usize, 2), layer.biases.len);
}

test "Layer Forward Pass - ReLU" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var layer = try layers.Layer.init(allocator, 3, 2, .relu);
    defer layer.deinit(allocator);

    // Fill weights and biases with predictable values for testing
    @memset(layer.weights, 1.0);
    @memset(layer.biases, 0.0);

    const input = [_]f32{ 1.0, -1.0, 0.5 };
    const output = try layer.forward(&input);

    try testing.expectApproxEqAbs(@as(f32, 0.5), output[0], 0.0001); // 1*1 + (-1)*1 + 0.5*1 = 1.5, after ReLU = 1.5
    try testing.expectApproxEqAbs(@as(f32, 0.5), output[1], 0.0001); // Same calculation
}

test "Layer Forward Pass - Softmax" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var layer = try layers.Layer.init(allocator, 2, 3, .softmax);
    defer layer.deinit(allocator);

    // Set weights and biases to predictable values
    @memset(layer.weights, 1.0);
    @memset(layer.biases, 0.0);

    const input = [_]f32{ 1.0, 1.0 };
    const output = try layer.forward(&input);

    // Softmax should distribute probability equally among three outputs
    const expected_value = 1.0 / 3.0;
    for (output) |o| {
        try testing.expectApproxEqAbs(expected_value, o, 0.0001);
    }
}

test "Layer Backward Pass - ReLU" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var layer = try layers.Layer.init(allocator, 2, 2, .relu);
    defer layer.deinit(allocator);

    // Predictable weights and biases for testing
    @memset(layer.weights, 1.0);
    @memset(layer.biases, 0.0);

    const input = [_]f32{ 1.0, -1.0 };
    const output_gradient = [_]f32{ 0.5, 0.0 }; // Gradient from the next layer

    const input_gradient = try layer.backward(&input, &output_gradient);

    // For ReLU, gradient is 1 where input > 0, 0 otherwise.
    // Here, we check if the weights were updated correctly:
    try testing.expectApproxEqAbs(@as(f32, 1.0 + 0.5 * 1.0), layer.weights[0], 0.0001); // 1 + 0.5 (since ReLU gradient for input[0] is 1)
    try testing.expectApproxEqAbs(@as(f32, 1.0), layer.weights[1], 0.0001); // No change since input[1] <= 0, ReLU gradient is 0
    try testing.expectApproxEqAbs(@as(f32, 0.5), layer.biases[0], 0.0001); // Bias update for first neuron
    try testing.expectApproxEqAbs(@as(f32, 0.0), layer.biases[1], 0.0001); // No change for second neuron

    // Check if input gradient calculations are correct
    try testing.expectApproxEqAbs(@as(f32, 0.5), input_gradient[0], 0.0001); // 0.5 * weight[0] = 0.5
    try testing.expectApproxEqAbs(@as(f32, 0.0), input_gradient[1], 0.0001); // No contribution from second neuron due to ReLU
}
