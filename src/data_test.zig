const std = @import("std");
const data = @import("data.zig");
const expect = std.testing.expect;
const expectError = std.testing.expectError;

test "readBigEndianInt" {
    const buf = [_]u8{ 0x00, 0x00, 0x01, 0x02 };
    try expect(data.readBigEndianInt(u32, &buf) == 0x00000102);
}

test "loadImages - valid file" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Assuming you have test data files in a 'data' directory
    const images = try data.loadImages(allocator, "data/t10k-images-idx3-ubyte");
    defer images.deinit();

    try expect(images.items.len == 10000); // MNIST test set has 10,000 images
}

test "loadLabels - valid file" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const labels = try data.loadLabels(allocator, "data/t10k-labels-idx1-ubyte");
    defer labels.deinit();

    try expect(labels.items.len == 10000); // MNIST test set has 10,000 labels
}

test "loadDataset - valid files" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var dataset = try data.loadDataset(allocator, "data/train-images-idx3-ubyte", "data/train-labels-idx1-ubyte");
    defer dataset.deinit();

    try expect(dataset.images.items.len == 60000); // MNIST training set has 60,000 images
    try expect(dataset.labels.items.len == 60000); // MNIST training set has 60,000 labels
}

test "normalizeImage" {
    var image = data.MNISTImage{ .data = [_]u8{255} ** (28 * 28) }; // All white image

    data.normalizeImage(&image);
    for (image.data) |pixel| {
        try expect(pixel == 255); // After normalization, all pixels should be 255 since we round back to u8
    }
}
