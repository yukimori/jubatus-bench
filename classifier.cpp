#include <fstream>
#include <random>
#include <vector>
#include "common.hpp"
#include <jubatus/core/common/big_endian.hpp>
#include <jubatus/core/framework/packer.hpp>
#include <jubatus/core/framework/stream_writer.hpp>

typedef std::chrono::duration<double, std::ratio<1, 1000> > Milli;
using jubatus::core::fv_converter::datum;

int main(int argc, char **argv) {
  auto opt = parse_args("classifier", argc, argv);
  for (auto path : opt.files) {
    std::cout << path << std::endl;

    auto data = load_csv(path);

    if (opt.shuffle) {
      std::mt19937 rng(0);
      std::shuffle(data.begin(), data.end(), rng);
    }

    if (opt.cv) {
      auto handle = create_classifier(opt.config.serialize());
      auto train_size = data.size() / 4 * 3;
      Milli train_time = stopwatch([train_size, &data, &handle]() {
        for (auto i = 0; i < train_size; ++i) {
          std::pair<std::string, datum>& item = data[i];
          handle->train(item.first, item.second);
        }
      });
      int ok = 0, ng = 0;
      Milli validate_time = stopwatch([train_size, &data, &handle, &ok, &ng]() {
        for (auto i = train_size; i < data.size(); ++i) {
          std::pair<std::string, datum>& item = data[i];
          auto ret = handle->classify(item.second);
          auto& top = ret[0];
          for (auto j = 1; j < ret.size(); ++j) {
            if (top.score < ret[j].score)
              top = ret[j];
          }
          if (top.label == item.first) {
            ++ok;
          } else {
            ++ng;
          }
        }
      });
      std::cout << "[4-fold cross-validation]" << std::endl
                << "   train: " << train_time.count() << " ms (" << train_size << " records)" << std::endl
                << "          " << (train_time.count() / train_size) << " ms (avg latency)" << std::endl
                << "          " << (train_size / (train_time.count() / 1000)) << " ops" << std::endl
                << "classify: " << validate_time.count() << " ms (" << (data.size() - train_size) << " records)" << std::endl
                << "          " << (validate_time.count() / (data.size() - train_size)) << " ms (avg latency)" << std::endl
                << "          " << ((data.size() - train_size) / (validate_time.count() / 1000)) << " ops" << std::endl
                << "validate: ok=" << ok << ", ng=" << ng << std::endl;
    }

    auto handle = create_classifier(opt.config.serialize());

    if (opt.train) {
      Milli time = stopwatch([&data, &handle]() {
        for (auto it = data.cbegin(); it != data.cend(); ++it) {
          handle->train(it->first, it->second);
        }
      });
      std::cout << "   train: " << time.count() << " ms (" << data.size() << " records)" << std::endl
                << "          " << (time.count() / data.size()) << " ms (avg latency)" << std::endl
                << "          " << (data.size() / (time.count() / 1000)) << " ops" << std::endl;
      if (!opt.validate && !opt.classify) {
        msgpack::sbuffer user_data_buf;
        jubatus::core::framework::stream_writer<msgpack::sbuffer> st(user_data_buf);
        jubatus::core::framework::jubatus_packer jp(st);
        jubatus::core::framework::packer packer(jp);
        handle->pack(packer);
        std::ofstream os (path + ".model", std::ios::out|std::ios::binary|std::ios::trunc);
        os.write(user_data_buf.data(), user_data_buf.size());
      }
    }

    if (!opt.train && (opt.validate || opt.classify)) {
        std::ifstream f(path + ".model", std::ios::in | std::ios::binary | std::ios::ate);
        std::string data(f.tellg(), '\0');
        f.seekg(0, std::ios::beg);
        f.read(const_cast<char*>(data.data()), data.size());
        msgpack::unpacked unpacked;
        msgpack::unpack(&unpacked, data.data(), data.size());
        handle->unpack(unpacked.get());
    }

    if (opt.validate) {
      int ok = 0, ng = 0;
      for (auto it = data.cbegin(); it != data.cend(); ++it) {
        auto ret = handle->classify(it->second);
        auto& top = ret[0];
        for (auto j = 1; j < ret.size(); ++j) {
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
      auto n = opt.count;
      if (n <= 0)
        n = data.size();
      Milli time = stopwatch([n, &data, &handle]() {
          auto i = 0;
          for (auto it = data.cbegin(); it != data.cend() && i < n; ++it, ++i) {
            handle->classify(it->second);
        }
      });
      std::cout << "classify: " << time.count() << " ms (" << n << " records)" << std::endl
                << "          " << (time.count() / n) << " ms (avg latency)" << std::endl
                << "          " << (n / (time.count() / 1000)) << " ops" << std::endl;
    }

    std::cout << std::endl;
  }
  return 0;
}
