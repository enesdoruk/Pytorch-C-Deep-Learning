#include <torch/torch.h>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <stdint.h>
#include <fstream>
#include <string>
#include <vector>

class CustomDataset : public torch::data::datasets::Dataset<CustomDataset> {
  using Example = torch::data::Example<>;

  Data data;

 public:
  CustomDataset(const Data& data) : data(data) {}

  Example get(size_t index) {
    std::string path = options.datasetPath + data[index].first;
    auto mat = cv::imread(path);
    assert(!mat.empty());

    cv::resize(mat, mat, cv::Size(options.image_size, options.image_size));
    std::vector<cv::Mat> channels(3);
    cv::split(mat, channels);

    auto R = torch::from_blob(
        channels[2].ptr(),
        {options.image_size, options.image_size},
        torch::kUInt8);
    auto G = torch::from_blob(
        channels[1].ptr(),
        {options.image_size, options.image_size},
        torch::kUInt8);
    auto B = torch::from_blob(
        channels[0].ptr(),
        {options.image_size, options.image_size},
        torch::kUInt8);

    auto tdata = torch::cat({R, G, B})
                     .view({3, options.image_size, options.image_size})
                     .to(torch::kFloat);
    auto tlabel = torch::from_blob(&data[index].second, {1}, torch::kLong);
    return {tdata, tlabel};
  }

  torch::optional<size_t> size() const {
    return data.size();
  }
};

std::pair<Data, Data> readInfo() {
  Data train, test;

  std::ifstream stream(options.infoFilePath);
  assert(stream.is_open());

  long label;
  std::string path, type;

  while (true) {
    stream >> path >> label >> type;

    if (type == "train")
      train.push_back(std::make_pair(path, label));
    else
      assert(false);

    if (stream.eof())
      break;
  }

  int main() {
    torch::manual_seed(1);

    if (torch::cuda::is_available())
      options.device = torch::kCUDA;
    std::cout << "Running on: "
              << (options.device == torch::kCUDA ? "CUDA" : "CPU") << std::endl;

    auto data = readInfo();

    auto train_set =
        CustomDataset(data.first).map(torch::data::transforms::Stack<>());
    auto train_size = train_set.size().value();
    auto train_loader =
        torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
            std::move(train_set), options.train_batch_size);
}
