#include <fstream>
#include <vector>
#include "common.hpp"

using jubatus::core::fv_converter::datum;

int main(int argc, char **argv) {
  Args opt = parse_args("recommender", argc, argv);

  for (std::vector<std::string>::iterator it = opt.files.begin(); it != opt.files.end(); ++it) {
    const std::string& path = *it;
    const std::string model_path(path + ".model");
    std::cout << path << std::endl;
    data_t data = load_csv(path);

#if 0
    if (opt.shuffle) {
      std::mt19937 rng(0);
      std::shuffle(data.begin(), data.end(), rng);
    }
#endif

    shared_ptr<jubatus::core::driver::recommender> handle = create_recommender(opt.config.serialize());
    if (opt.train) {
      double time = now_millisec();
      int i = 0;
      for (data_t::iterator it = data.begin(); it != data.end(); ++it, ++i) {
          handle->update_row(to_string(i), it->second);
      }
      time = now_millisec() - time;
      std::cout << " update: " << time << " ms (" << data.size() << " records)" << std::endl
                << "         " << (time / data.size()) << " ms (avg latency)" << std::endl
                << "         " << (data.size() / (time / 1000)) << " ops" << std::endl;
      std::cout << "    mem: " << (get_memory_usage()) << " [MiB]" << std::endl;

      if (!opt.similar_row && !opt.decode_row) {
        msgpack::sbuffer user_data_buf;
        jubatus::core::framework::stream_writer<msgpack::sbuffer> st(user_data_buf);
        jubatus::core::framework::jubatus_packer jp(st);
        jubatus::core::framework::packer packer(jp);
        handle->pack(packer);
        std::ofstream os (model_path.c_str(), std::ios::out|std::ios::binary|std::ios::trunc);
        os.write(user_data_buf.data(), user_data_buf.size());
      }
    }

    if (!opt.train && (opt.similar_row || opt.decode_row)) {
      std::ifstream f(model_path.c_str(), std::ios::in | std::ios::binary | std::ios::ate);
      std::string data(f.tellg(), '\0');
      f.seekg(0, std::ios::beg);
      f.read(const_cast<char*>(data.data()), data.size());
      msgpack::unpacked unpacked;
      msgpack::unpack(&unpacked, data.data(), data.size());
      handle->unpack(unpacked.get());
    }

    int n = opt.count;
    if (n <= 0) n = data.size();

    if (opt.similar_row) {
      int size = opt.size;
      double time = now_millisec();
      int i = 0;
      for (data_t::iterator it = data.begin(); it != data.end() && i < n; ++it, ++i) {
        handle->similar_row_from_datum(it->second, size);
      }
      time = now_millisec() - time;
      std::cout << "similar: " << time << " ms (" << n << " records)" << std::endl
                << "         " << (time / n) << " ms (avg latency)" << std::endl
                << "         " << (n / (time / 1000)) << " ops" << std::endl;
    }

    if (opt.decode_row) {
      double time = now_millisec();
      for (int i = 0; i < n; ++i) {
        handle->decode_row(to_string(i));
      }
      std::cout << " decode: " << time << " ms (" << n << " ops)" << std::endl
                << "         " << (time / n) << " ms (avg latency)" << std::endl
                << "         " << (n / (time / 1000)) << " ops" << std::endl;
    }
    std::cout << std::endl;
  }
  return 0;
}
