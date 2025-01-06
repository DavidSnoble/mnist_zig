const std = @import("std");
const math = std.math;

const learning_rate = 0.01;

pub const ActivationFunction = enum {
    relu,
    softmax,
};

pub const Layer = struct {
    weights: []f32,
    biases: []f32,
    input_size: usize,
    output_size: usize,
    activation: ActivationFunction,

    pub fn init(allocator: std.mem.Allocator, input_size: usize, output_size: usize, activation: ActivationFunction) !Layer {
        const weights = try allocator.alloc(f32, input_size * output_size);
        const biases = try allocator.alloc(f32, output_size);

        // Xavier/Glorot initialization for weights
        const scale = @sqrt(6.0 / @as(f32, @floatFromInt(input_size + output_size)));
        for (weights) |*w| {
            w.* = randomFloat(-scale, scale);
        }
        @memset(biases, 0);

        return Layer{
            .weights = weights,
            .biases = biases,
            .input_size = input_size,
            .output_size = output_size,
            .activation = activation,
        };
    }

    pub fn deinit(self: *Layer, allocator: std.mem.Allocator) void {
        allocator.free(self.weights);
        allocator.free(self.biases);
    }

    // Forward pass
    pub fn forward(self: *Layer, input: []const f32) ![]f32 {
        const allocator = std.heap.page_allocator;
        var output = try allocator.alloc(f32, self.output_size);
        errdefer allocator.free(output);

        for (0..self.output_size) |j| {
            var sum: f32 = self.biases[j];
            for (0..self.input_size) |i| {
                sum += input[i] * self.weights[j * self.input_size + i];
            }
            output[j] = switch (self.activation) {
                .relu => @max(0, sum),
                .softmax => softmax(sum, output),
            };
        }

        return output;
    }

    // Backward pass for backpropagation
    pub fn backward(self: *Layer, input: []const f32, output_gradient: []const f32) ![]f32 {
        const allocator = std.heap.page_allocator;
        var input_gradient = try allocator.alloc(f32, self.input_size);
        errdefer allocator.free(input_gradient);

        @memset(input_gradient, 0);
        // Compute gradients for weights and biases
        for (0..self.output_size) |j| {
            var sum: f32 = self.biases[j];
            for (0..self.input_size) |i| {
                sum += input[i] * self.weights[j * self.input_size + i];
            }
            const activation_gradient = switch (self.activation) {
                .relu => if (sum > 0) @as(f32, 1.0) else @as(f32, 0.0), // Use float literals
                .softmax => output_gradient[j], // Simplified; usually requires stored intermediate activations
            };

            self.biases[j] += learning_rate * output_gradient[j] * activation_gradient;

            for (0..self.input_size) |i| {
                const weight_gradient = output_gradient[j] * activation_gradient * input[i];
                self.weights[j * self.input_size + i] += learning_rate * weight_gradient;
                input_gradient[i] += self.weights[j * self.input_size + i] * output_gradient[j] * activation_gradient;
            }
        }

        return input_gradient;
    }
};

// Helper function to generate a random float
fn randomFloat(min: f32, max: f32) f32 {
    var rng = std.rand.DefaultPrng.init(0);
    return rng.random().float(f32) * (max - min) + min;
}

fn softmax(value: f32, outputs: []f32) f32 {
    var sum: f32 = 0;
    for (outputs) |v| {
        sum += @exp(v);
    }
    return @exp(value) / sum;
}

// Placeholder for learning rate; in a real scenario, this would be passed to methods

// Helper function to create a dense layer
pub fn createDenseLayer(allocator: std.mem.Allocator, input_size: usize, output_size: usize, activation: ActivationFunction) !Layer {
    return Layer.init(allocator, input_size, output_size, activation);
}
