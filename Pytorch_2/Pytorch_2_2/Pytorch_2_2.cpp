#include <torch/torch.h>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <stdint.h>
#include <fstream>
#include <string>
#include <vector>

int main(){

auto dataset = torch::data::datasets::MNIST("./mnist")
    .map(torch::data::transforms::Normalize<>(0.5, 0.5))
    .map(torch::data::transforms::Stack<>());

auto data_loader = torch::data::make_data_loader(
    std::move(dataset),
    torch::data::DataLoaderOptions().batch_size(kBatchSize).workers(2));

for (torch::data::Example<>& batch : *data_loader) {
  std::cout << "Batch size: " << batch.data.size(0) << " | Labels: ";
  for (int64_t i = 0; i < batch.data.size(0); ++i) {
    std::cout << batch.target[i].item<int64_t>() << " ";
  }
  std::cout << std::endl;
}

}
