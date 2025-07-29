#include <openvino/openvino.hpp>
#include <vector>
#include <cassert>
#include <cmath>
#include <iostream>
#include "openvino/op/interpolate.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/clamp.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/round.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/ops.hpp"

std::shared_ptr<ov::Model> create_unfold_model(size_t channels, size_t kernel) {
    using namespace ov;
    // Define dynamic input shape: [N, C, H, W]
    auto input_shape = PartialShape{Dimension::dynamic(), channels, Dimension::dynamic(), Dimension::dynamic()};
    auto input = std::make_shared<ov::op::v0::Parameter>(element::f32, input_shape);

    // ExtractImagePatches
    ov::Shape kernel_size = {kernel, kernel};
    ov::Strides strides = {kernel, kernel};
    ov::Shape dilations = {1, 1};
    auto unfold_op = std::make_shared<ov::op::v3::ExtractImagePatches>(
        input, kernel_size, strides, dilations, ov::op::PadType::VALID
    );

    // Get input shape
    auto input_shape_node = std::make_shared<ov::op::v0::ShapeOf>(input);
    auto h_in = std::make_shared<ov::op::v1::StridedSlice>(
        input_shape_node,
        ov::op::v0::Constant::create(element::i64, Shape{1}, {2}),
        ov::op::v0::Constant::create(element::i64, Shape{1}, {3}),
        std::vector<int64_t>{0},
        std::vector<int64_t>{0}
    );
    auto w_in = std::make_shared<ov::op::v1::StridedSlice>(
        input_shape_node,
        ov::op::v0::Constant::create(element::i64, Shape{1}, {3}),
        ov::op::v0::Constant::create(element::i64, Shape{1}, {4}),
        std::vector<int64_t>{0},
        std::vector<int64_t>{0}
    );
    auto kernel_const = ov::op::v0::Constant::create(element::i64, Shape{1}, {static_cast<int64_t>(kernel)});
    auto one = ov::op::v0::Constant::create(element::i64, Shape{1}, {1});
    auto h_out = std::make_shared<ov::op::v1::Add>(
        std::make_shared<ov::op::v1::Divide>(
            std::make_shared<ov::op::v1::Subtract>(h_in, kernel_const),
            kernel_const
        ),
        one
    );
    auto w_out = std::make_shared<ov::op::v1::Add>(
        std::make_shared<ov::op::v1::Divide>(
            std::make_shared<ov::op::v1::Subtract>(w_in, kernel_const),
            kernel_const
        ),
        one
    );

    // Reshape to [N, kernel_h, kernel_w, channels, H_out, W_out]
    auto minus_one = ov::op::v0::Constant::create(element::i64, Shape{1}, {-1});
    auto kernel_const_reshaped = ov::op::v0::Constant::create(element::i64, Shape{1}, {static_cast<int64_t>(kernel)});
    auto channels_const = ov::op::v0::Constant::create(element::i64, Shape{1}, {static_cast<int64_t>(channels)});
    auto reshape1_shape = std::make_shared<ov::op::v0::Concat>(
        NodeVector{
            minus_one,
            kernel_const_reshaped,
            kernel_const_reshaped,
            channels_const,
            h_out,
            w_out
        },
        0
    );
    auto reshape1 = std::make_shared<ov::op::v1::Reshape>(unfold_op, reshape1_shape, true);

    // Transpose to [N, channels, kernel_h, kernel_w, H_out, W_out]
    auto transpose_order = ov::op::v0::Constant::create(element::i64, Shape{6}, {0, 3, 1, 2, 4, 5});
    auto transpose = std::make_shared<ov::op::v1::Transpose>(reshape1, transpose_order);

    // Compute h_w_product
    auto h_w_product = std::make_shared<ov::op::v1::Multiply>(h_out, w_out);

    // Compute new_depth = channels * kernel * kernel
    auto new_depth = ov::op::v0::Constant::create(element::i64, Shape{1}, {static_cast<int64_t>(channels * kernel * kernel)});

    // Reshape to [N, new_depth, h_w_product]
    auto reshape2_shape = std::make_shared<ov::op::v0::Concat>(
        NodeVector{
            minus_one,
            new_depth,
            h_w_product
        },
        0
    );
    auto reshape2 = std::make_shared<ov::op::v1::Reshape>(transpose, reshape2_shape, true);

    // Create model
    auto result = std::make_shared<ov::op::v0::Result>(reshape2);
    return std::make_shared<ov::Model>(result, ParameterVector{input});
}

// Execute the unfold operation using OpenVINO model
ov::Tensor unfold_openvino(const ov::Tensor& images_tensor, size_t channels, size_t kernel) {
    using namespace ov;
    // Validate input
    ov::Shape images_shape = images_tensor.get_shape();
    OPENVINO_ASSERT(4 == images_shape.size(), "Input tensor must be 4D (NCHW).");
    const size_t images_h = images_shape.at(2);
    const size_t images_w = images_shape.at(3);
    OPENVINO_ASSERT(images_h >= kernel && images_w >= kernel, "Input height and width must be greater than or equal to kernel size.");

    // Create and compile model
    ov::Core core;
    auto model = create_unfold_model(channels, kernel);
    auto compiled_model = core.compile_model(model, "CPU");
    auto infer_request = compiled_model.create_infer_request();

    // Set input tensor and run inference
    infer_request.set_input_tensor(images_tensor);
    infer_request.infer();
    return infer_request.get_output_tensor();
}

// Original unfold function (unchanged)
ov::Tensor unfold(const ov::Tensor& images_tensor, size_t kernel) {
    ov::Shape images_shape = images_tensor.get_shape();

    OPENVINO_ASSERT(4 == images_shape.size(), "Input tensor must be 4D (NCHW).");

    const size_t bs = images_shape.at(0);
    const size_t images_c = images_shape.at(1);
    const size_t images_h = images_shape.at(2);
    const size_t images_w = images_shape.at(3);

    OPENVINO_ASSERT(images_h >= kernel && images_w >= kernel, "Input height and width must be greater than or equal to kernel size.");

    const size_t new_c = images_c * kernel * kernel;
    const size_t output_h = (images_h - kernel) / kernel + 1;
    const size_t output_w = (images_w - kernel) / kernel + 1;
    const size_t kernels_per_plane = output_h * output_w;

    ov::Tensor unfolded_tensor(ov::element::f32, {bs, new_c, kernels_per_plane});
    const float* images = images_tensor.data<float>();
    float* unfolded = unfolded_tensor.data<float>();
    for (size_t batch_idx = 0; batch_idx < bs; ++batch_idx) {
        for (size_t c_idx = 0; c_idx < images_c; ++c_idx) {
            for (size_t h_out = 0; h_out < output_h; ++h_out) {
                for (size_t w_out = 0; w_out < output_w; ++w_out) {
                    size_t h_idx = h_out * kernel;  // Calculate input height index
                    size_t w_idx = w_out * kernel;  // Calculate input width index

                    for (size_t kh = 0; kh < kernel; ++kh) {
                        for (size_t kw = 0; kw < kernel; ++kw) {
                            size_t input_idx = (batch_idx * images_c * images_h * images_w) +
                                                (c_idx * images_h * images_w) +
                                                ((h_idx + kh) * images_w) +
                                                (w_idx + kw);

                            size_t unfolded_c_idx = (c_idx * kernel * kernel) + (kh * kernel) + kw;
                            size_t unfolded_idx = (batch_idx * new_c * kernels_per_plane) +
                                                    unfolded_c_idx * kernels_per_plane +
                                                    (h_out * output_w + w_out);

                            unfolded[unfolded_idx] = images[input_idx];
                        }
                    }
                }
            }
        }
    }
    return unfolded_tensor;
}

// Test function to compare original and OpenVINO unfold implementations
bool test_unfold(size_t batch_size, size_t channels, size_t height, size_t width, size_t kernel) {
    // Create sample input tensor
    ov::Tensor input_tensor(ov::element::f32, {batch_size, channels, height, width});
    float* input_data = input_tensor.data<float>();
    
    // Fill input tensor with sequential values
    size_t idx = 0;
    for (size_t b = 0; b < batch_size; ++b) {
        for (size_t c = 0; c < channels; ++c) {
            for (size_t h = 0; h < height; ++h) {
                for (size_t w = 0; w < width; ++w) {
                    input_data[b * channels * height * width + c * height * width + h * width + w] = static_cast<float>(idx++);
                }
            }
        }
    }

    // Run original unfold function
    ov::Tensor original_output = unfold(input_tensor, kernel);

    // Run OpenVINO unfold function
    ov::Tensor openvino_output = unfold_openvino(input_tensor, channels, kernel);

    // Compare shapes
    if (original_output.get_shape() != openvino_output.get_shape()) {
        std::cerr << "Shape mismatch: original " << original_output.get_shape() 
                  << " vs OpenVINO " << openvino_output.get_shape() << std::endl;
        return false;
    }

    // Debug: Print values around index 64 and more
    const float* original_data = original_output.data<float>();
    const float* openvino_data = openvino_output.data<float>();
    for (size_t i = 60; i < 128; ++i) {
        std::cerr << "Index " << i << ": original " << original_data[i] 
                  << ", OpenVINO " << openvino_data[i] << std::endl;
    }

    // Compare values
    size_t total_elements = original_output.get_size();
    const float epsilon = 1e-5f;
    for (size_t i = 0; i < total_elements; ++i) {
        if (std::abs(original_data[i] - openvino_data[i]) > epsilon) {
            std::cerr << "Value mismatch at index " << i << ": original " << original_data[i] 
                      << " vs OpenVINO " << openvino_data[i] << std::endl;
            return false;
        }
    }

    std::cout << "Test passed: Outputs match." << std::endl;
    return true;
}

int main() {
    // Test parameters
    size_t batch_size = 2;
    size_t channels = 3;
    size_t height = 32;
    size_t width = 32;
    size_t kernel = 4;

    // Ensure input dimensions are valid
    assert(height >= kernel && width >= kernel);

    // Run test
    bool success = test_unfold(batch_size, channels, height, width, kernel);
    return success ? 0 : 1;
}