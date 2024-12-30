const std = @import("std");

pub const MNISTImage = struct { data: [28 * 28]u8 };

pub const MNISTLabel = u8;

pub const MNISTDataset = struct {
    images: std.ArrayList(MNISTImage),
    labels: std.ArrayList(MNISTLabel),

    pub fn deinit(self: *MNISTDataset) void {
        self.images.deinit();
        self.labels.deinit();
    }
};

pub fn readBigEndianInt(comptime T: type, buf: []const u8) T {
    var result: T = 0;
    for (buf) |byte| {
        result = (result << 8) | byte;
    }
    return result;
}

// function to load MNIST images
pub fn loadImages(allocator: std.mem.Allocator, file_path: []const u8) !std.ArrayList(MNISTImage) {
    var images = std.ArrayList(MNISTImage).init(allocator);
    errdefer images.deinit();

    const file = try std.fs.cwd().openFile(file_path, .{});
    defer file.close();

    var magic: [4]u8 = undefined;
    _ = try file.readAll(&magic);
    if (readBigEndianInt(u32, &magic) != 2051) return error.InvalidMagicNumber;

    var num_images: [4]u8 = undefined;
    _ = try file.readAll(&num_images);
    const count = readBigEndianInt(u32, &num_images);

    var rows: [4]u8 = undefined;
    var cols: [4]u8 = undefined;

    _ = try file.readAll(&rows);
    _ = try file.readAll(&cols);
    if (readBigEndianInt(u32, &rows) != 28 or readBigEndianInt(u32, &cols) != 28) return error.InvalidDimensions;

    var image_data: [28 * 28]u8 = undefined;

    var i: usize = 0;

    while (i < count) : (i += 1) {
        _ = try file.readAll(&image_data);
        try images.append(MNISTImage{ .data = image_data });
    }

    return images;
}

pub fn loadLabels(allocator: std.mem.Allocator, file_path: []const u8) !std.ArrayList(MNISTLabel) {
    var labels = std.ArrayList(MNISTLabel).init(allocator);
    errdefer labels.deinit();

    const file = try std.fs.cwd().openFile(file_path, .{});
    defer file.close();

    // Read the magic number (should be 2049 for labels)
    var magic: [4]u8 = undefined;
    _ = try file.readAll(&magic);
    if (readBigEndianInt(u32, &magic) != 2049) return error.InvalidMagicNumber;

    // Read number of labels
    var num_labels: [4]u8 = undefined;
    _ = try file.readAll(&num_labels);
    const count = readBigEndianInt(u32, &num_labels);

    // Read label data
    var label: u8 = undefined;
    var i: usize = 0;
    while (i < count) : (i += 1) {
        _ = try file.readAll(std.mem.asBytes(&label));
        try labels.append(label);
    }

    return labels;
}

pub fn loadDataset(allocator: std.mem.Allocator, images_path: []const u8, labels_path: []const u8) !MNISTDataset {
    const dataset = MNISTDataset{
        .images = try loadImages(allocator, images_path),
        .labels = try loadLabels(allocator, labels_path),
    };

    return dataset;
}

pub fn normalizeImage(image: *MNISTImage) void {
    for (&image.data) |*pixel| {
        const normalized: f64 = @as(f64, @floatFromInt(pixel.*)) / 255.0;
        pixel.* = @as(u8, @intFromFloat(@round(normalized * 255.0)));
    }
}
