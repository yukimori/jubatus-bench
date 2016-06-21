#include <algorithm>
#include <fstream>
#include <vector>
#include "common.hpp"
#include "jubatus/util/math/random.h"
#include "jubatus/util/lang/cast.h"

using jubatus::core::fv_converter::datum;

int main(int argc, char **argv) {
  Args opt = parse_args("nn", argc, argv);

  const int hashnum = static_cast<int>(opt.config.get("parameter").get("hash_num").get<double>());
  const int num_of_features = 100;
  const int num_of_rows = opt.size;
  const int num_of_tests = opt.count;
  const std::string model_path("nn-" +
                               jubatus::util::lang::lexical_cast<std::string>(hashnum) +
                               "-" +
                               jubatus::util::lang::lexical_cast<std::string>(num_of_rows) +
                               ".model");

  jubatus::util::math::random::sfmt607rand rng(0);
  shared_ptr<jubatus::core::driver::nearest_neighbor> handle = create_nearest_neighbor(opt.config.serialize());

  if (opt.train) {
    double time = now_millisec();
    datum d;
    for (int i = 0; i < num_of_rows; ++i) {
      const std::string id = jubatus::util::lang::lexical_cast<std::string>(i);
      d.num_values_.clear();
      for (int j = 0; j < num_of_features; ++j) {
        const std::string k = jubatus::util::lang::lexical_cast<std::string>(j);
        d.num_values_.push_back(std::make_pair(k, rng.next_double()));
      }
      handle->set_row(id, d);
    }
    time = now_millisec() - time;

    std::cout << " train: " << time << " ms (" << num_of_rows << " records)" << std::endl
              << "        " << (time / num_of_rows) << " ms (avg latency)" << std::endl
              << "        " << (num_of_rows / (time / 1000)) << " ops" << std::endl;
    std::cout << "   mem: " << (get_memory_usage()) << " [MiB]" << std::endl;

    {
      msgpack::sbuffer user_data_buf;
      jubatus::core::framework::stream_writer<msgpack::sbuffer> st(user_data_buf);
      jubatus::core::framework::jubatus_packer jp(st);
      jubatus::core::framework::packer packer(jp);
      handle->pack(packer);
      std::ofstream os (model_path.c_str(), std::ios::out|std::ios::binary|std::ios::trunc);
      os.write(user_data_buf.data(), user_data_buf.size());
    }
  } else {
    std::ifstream f(model_path.c_str(), std::ios::in | std::ios::binary | std::ios::ate);
    std::string data(f.tellg(), '\0');
    f.seekg(0, std::ios::beg);
    f.read(const_cast<char*>(data.data()), data.size());
    msgpack::unpacked unpacked;
    msgpack::unpack(&unpacked, data.data(), data.size());
    handle->unpack(unpacked.get());
  }

  {
    {
      handle->neighbor_row_from_id(jubatus::util::lang::lexical_cast<std::string>(rng.next_int(num_of_rows)), 1);
    }

    double time = now_millisec();
    for (int i = 0; i < num_of_tests; ++i) {
      const std::string id = jubatus::util::lang::lexical_cast<std::string>(rng.next_int(num_of_rows));
      handle->neighbor_row_from_id(id, 1000);
    }
    time = now_millisec() - time;
    std::cout << " from_id: " << (time / num_of_tests) << " ms (avg latency)" << std::endl
              << "        " << (num_of_tests / (time / 1000)) << " ops" << std::endl;
  }

  return 0;
}
