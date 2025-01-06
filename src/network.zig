const std = @import("std");
const layers = @import("layers.zig");

pub const Network = struct {
    layers: std.ArrayList(layers.Layer),
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator) !Network {
        return Network{
            .layers = std.ArrayList(layers.Layer).init(allocator),
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *Network) void {
        for (self.layers.items) |*layer| {
            layer.deinit();
        }
        self.layers.deinit();
    }

    // Add a new layer to the network
    pub fn addLayer(self: *Network, layer: layers.Layer) !void {
        try self.layers.append(layer);
    }

    // Forward pass through the network
    pub fn forward(self: *Network, input: []const f32) ![]f32 {
        var current_output = try self.allocator.alloc(f32, input.len);
        std.mem.copy(f32, current_output, input);

        for (self.layers.items) |*layer| {
            const new_output = try layer.forward(current_output);
            self.allocator.free(current_output);
            current_output = new_output;
        }

        return current_output;
    }

    // Backward pass (placeholder for backpropagation)
    pub fn backward(self: *Network, gradient: []const f32) !void {
        var current_gradient = try self.allocator.alloc(f32, gradient.len);
        std.mem.copy(f32, current_gradient, gradient);

        // This would be done in reverse order of layers
        var i = self.layers.items.len;
        while (i > 0) {
            i -= 1;
            current_gradient = try self.layers.items[i].backward(current_gradient);
        }

        self.allocator.free(current_gradient);
    }

    // Update weights and biases (placeholder for optimization step)
    pub fn update(self: *Network, learning_rate: f32) !void {
        for (self.layers.items) |*layer| {
            try layer.update(learning_rate);
        }
    }
};

pub fn createSimpleNetwork(allocator: std.mem.Allocator) !Network {
    var network = try Network.init(allocator);
    errdefer network.deinit();

    // Example layers for a simple network:
    // Input layer (flattened 28x28 image = 784 neurons)
    // First hidden layer (128 neurons)
    // Second hidden layer (64 neurons)
    // Output layer (10 neurons for 10 digits)
    try network.addLayer(try layers.createDenseLayer(allocator, 784, 128, .relu));
    try network.addLayer(try layers.createDenseLayer(allocator, 128, 64, .relu));
    try network.addLayer(try layers.createDenseLayer(allocator, 64, 10, .softmax));

    return network;
}
