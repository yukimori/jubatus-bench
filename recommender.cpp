#include <fstream>
#include <random>
#include <vector>
#include "common.hpp"

using jubatus::core::fv_converter::datum;

int main(int argc, char **argv) {
  auto opt = parse_args("recommender", argc, argv);

  for (auto path : opt.files) {
    std::cout << path << std::endl;
    auto data = load_csv(path);
    if (opt.shuffle) {
      std::mt19937 rng(0);
      std::shuffle(data.begin(), data.end(), rng);
    }

    auto handle = create_recommender(opt.config.serialize());
    if (opt.train) {
      Milli time = stopwatch([&data, &handle]() {
        int i = 0;
        for (auto it = data.cbegin(); it != data.cend(); ++it, ++i) {
          handle->update_row(std::to_string(i), it->second);
        }
      });
      std::cout << " update: " << time.count() << " ms (" << data.size() << " records)" << std::endl
                << "         " << (time.count() / data.size()) << " ms (avg latency)" << std::endl
                << "         " << (data.size() / (time.count() / 1000)) << " ops" << std::endl;

      if (!opt.similar_row && !opt.decode_row) {
        msgpack::sbuffer user_data_buf;
        jubatus::core::framework::stream_writer<msgpack::sbuffer> st(user_data_buf);
        jubatus::core::framework::jubatus_packer jp(st);
        jubatus::core::framework::packer packer(jp);
        handle->pack(packer);
        std::ofstream os (path + ".model", std::ios::out|std::ios::binary|std::ios::trunc);
        os.write(user_data_buf.data(), user_data_buf.size());
      }
    }

    if (!opt.train && (opt.similar_row || opt.decode_row)) {
        std::ifstream f(path + ".model", std::ios::in | std::ios::binary | std::ios::ate);
        std::string data(f.tellg(), '\0');
        f.seekg(0, std::ios::beg);
        f.read(const_cast<char*>(data.data()), data.size());
        msgpack::unpacked unpacked;
        msgpack::unpack(&unpacked, data.data(), data.size());
        handle->unpack(unpacked.get());
    }

    auto n = opt.count;
    if (n <= 0) n = data.size();

    if (opt.similar_row) {
      auto size = opt.size;
      Milli time = stopwatch([n, size, &data, &handle]() {
        int i = 0;
        for (auto it = data.cbegin(); it != data.cend() && i < n; ++it, ++i) {
          handle->similar_row_from_datum(it->second, size);
        }
      });
      std::cout << "similar: " << time.count() << " ms (" << n << " records)" << std::endl
                << "         " << (time.count() / n) << " ms (avg latency)" << std::endl
                << "         " << (n / (time.count() / 1000)) << " ops" << std::endl;
    }

    if (opt.decode_row) {
      Milli time = stopwatch([n, &data, &handle]() {
        int i = 0;
        for (auto i = 0; i < n; ++i) {
          handle->decode_row(std::to_string(i));
        }
      });
      std::cout << " decode: " << time.count() << " ms (" << n << " ops)" << std::endl
                << "         " << (time.count() / n) << " ms (avg latency)" << std::endl
                << "         " << (n / (time.count() / 1000)) << " ops" << std::endl;
    }
    std::cout << std::endl;
  }
  return 0;
}
