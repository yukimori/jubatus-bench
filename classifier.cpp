#include <algorithm>
#include <fstream>
#include <vector>
#include "common.hpp"
#include "jubatus/util/math/random.h"

int main(int argc, char **argv) {
  Args opt = parse_args("classifier", argc, argv);
  for (std::vector<std::string>::iterator it = opt.files.begin(); it != opt.files.end(); ++it) {
    const std::string& path = *it;
    const std::string model_path(path + ".model");
    std::cout << path << std::endl;

    data_t data = load_csv(path);

    if (opt.shuffle) {
      jubatus::util::math::random::mtrand rng(0);
      std::random_shuffle(data.begin(), data.end(), rng);
    }

    if (opt.cv) {
      shared_ptr<jubatus::core::driver::classifier> handle = create_classifier(opt.config.serialize());
      size_t train_size = data.size() / 4 * 3;
      double train_time = now_millisec();
      for (size_t i = 0; i < train_size; ++i) {
        std::pair<std::string, datum>& item = data[i];
        handle->train(item.first, item.second);
      }
      train_time = now_millisec() - train_time;
      int ok = 0, ng = 0;
      double validate_time = now_millisec();
      for (size_t i = train_size; i < data.size(); ++i) {
        std::pair<std::string, datum>& item = data[i];
        jubatus::core::classifier::classify_result ret = handle->classify(item.second);
        jubatus::core::classifier::classify_result_elem& top = ret[0];
        for (size_t j = 1; j < ret.size(); ++j) {
          if (top.score < ret[j].score)
            top = ret[j];
        }
        if (top.label == item.first) {
          ++ok;
        } else {
          ++ng;
        }
      }
      validate_time = now_millisec() - validate_time;
      std::cout << "[4-fold cross-validation]" << std::endl
                << "   train: " << train_time << " ms (" << train_size << " records)" << std::endl
                << "          " << (train_time / train_size) << " ms (avg latency)" << std::endl
                << "          " << (train_size / (train_time / 1000)) << " ops" << std::endl
                << "classify: " << validate_time << " ms (" << (data.size() - train_size) << " records)" << std::endl
                << "          " << (validate_time / (data.size() - train_size)) << " ms (avg latency)" << std::endl
                << "          " << ((data.size() - train_size) / (validate_time / 1000)) << " ops" << std::endl
                << "validate: ok=" << ok << ", ng=" << ng << std::endl;
    }

    shared_ptr<jubatus::core::driver::classifier> handle = create_classifier(opt.config.serialize());

    if (opt.train) {
      double time = now_millisec();
      for (data_t::iterator it = data.begin(); it != data.end(); ++it) {
        handle->train(it->first, it->second);
      }
      time = now_millisec() - time;
      std::cout << "   train: " << time << " ms (" << data.size() << " records)" << std::endl
                << "          " << (time / data.size()) << " ms (avg latency)" << std::endl
                << "          " << (data.size() / (time / 1000)) << " ops" << std::endl;
      if (!opt.validate && !opt.classify) {
        msgpack::sbuffer user_data_buf;
        jubatus::core::framework::stream_writer<msgpack::sbuffer> st(user_data_buf);
        jubatus::core::framework::jubatus_packer jp(st);
        jubatus::core::framework::packer packer(jp);
        handle->pack(packer);
        std::ofstream os (model_path.c_str(), std::ios::out|std::ios::binary|std::ios::trunc);
        os.write(user_data_buf.data(), user_data_buf.size());
      }
    }

    if (!opt.train && (opt.validate || opt.classify)) {
        std::ifstream f(model_path.c_str(), std::ios::in | std::ios::binary | std::ios::ate);
        std::string data(f.tellg(), '\0');
        f.seekg(0, std::ios::beg);
        f.read(const_cast<char*>(data.data()), data.size());
        msgpack::unpacked unpacked;
        msgpack::unpack(&unpacked, data.data(), data.size());
        handle->unpack(unpacked.get());
        std::cout << "model loaded" << std::endl;
    }

    if (opt.validate) {
      int ok = 0, ng = 0;
      for (data_t::iterator it = data.begin(); it != data.end(); ++it) {
        jubatus::core::classifier::classify_result ret = handle->classify(it->second);
        jubatus::core::classifier::classify_result_elem& top = ret[0];
        for (size_t j = 1; j < ret.size(); ++j) {
          if (top.score < ret[j].score)
            top = ret[j];
        }
        if (top.label == it->first) {
          ++ok;
        } else {
          ++ng;
        }
      }
      std::cout << "validate: ok=" << ok << ", ng=" << ng << std::endl;
    }

    if (opt.classify) {
      int n = opt.count;
      if (n <= 0)
        n = data.size();
      double time = now_millisec();
      int i = 0;
      for (data_t::iterator it = data.begin(); it != data.end() && i < n; ++it, ++i) {
        handle->classify(it->second);
      }
      time = now_millisec() - time;
      std::cout << "classify: " << time << " ms (" << n << " records)" << std::endl
                << "          " << (time / n) << " ms (avg latency)" << std::endl
                << "          " << (n / (time / 1000)) << " ops" << std::endl;
    }

    std::cout << std::endl;
  }
  return 0;
}
