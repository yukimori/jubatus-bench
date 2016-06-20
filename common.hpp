#ifndef JUBATUS_BENCH_COMMON_HPP
#define JUBATUS_BENCH_COMMON_HPP

#include <string>
#include <sstream>
#include <vector>
#include <utility>

#include <jubatus/core/fv_converter/datum.hpp>
#include <jubatus/core/driver/classifier.hpp>
#include <jubatus/core/driver/recommender.hpp>
#include <jubatus/core/driver/nearest_neighbor.hpp>
#include <jubatus/core/common/big_endian.hpp>
#include <jubatus/core/framework/packer.hpp>
#include <jubatus/core/framework/stream_writer.hpp>
#include <jubatus/util/lang/shared_ptr.h>

#include "picojson.h"

using jubatus::core::fv_converter::datum;
using jubatus::util::lang::shared_ptr;

typedef std::vector<std::pair<std::string, datum> > data_t;

struct Args {
  picojson::value config;
  std::vector<std::string> files;
  bool train;
  bool validate;
  bool classify;
  bool similar_row;
  bool decode_row;
  int count;
  int size;
  bool shuffle;
  bool cv;

  Args(): config(), files(),
          train(false), validate(false), classify(false),
          similar_row(false), decode_row(false),
          count(0), size(8),
          shuffle(false), cv(false) {}
};

Args parse_args(const std::string& api, int argc, char **argv);
std::string load_text(const std::string& path);
data_t load_csv(const std::string& path);

inline double now_millisec() {
  struct timespec tp;
  clock_gettime(CLOCK_MONOTONIC, &tp);
  return static_cast<double>(tp.tv_sec) * 1000.0 +
    static_cast<double>(tp.tv_nsec) * 0.000001;
}
double get_memory_usage();

shared_ptr<jubatus::core::driver::classifier> create_classifier(const std::string& config);
shared_ptr<jubatus::core::driver::nearest_neighbor> create_nearest_neighbor(const std::string& config);
shared_ptr<jubatus::core::driver::recommender> create_recommender(const std::string& config);

template<typename T>
std::string to_string(const T& v) {
  std::ostringstream os;
  os << v;
  return os.str();
}

#endif
