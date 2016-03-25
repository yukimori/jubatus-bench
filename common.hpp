#ifndef JUBATUS_BENCH_COMMON_HPP
#define JUBATUS_BENCH_COMMON_HPP

#include <chrono>
#include <string>
#include <vector>
#include <utility>
#include <jubatus/core/fv_converter/datum.hpp>
#include <jubatus/core/driver/classifier.hpp>
#include "picojson.h"

struct Args {
  picojson::value config;
  std::vector<std::string> files;
  bool train;
  bool validate;
  bool classify;
  int count;
  bool shuffle;
  bool cv;

  Args(): config(), files(),
          train(false), validate(false),
          classify(false), count(0),
          shuffle(false), cv(false) {}
};

Args parse_args(const std::string& api, int argc, char **argv);
std::string load_text(const std::string& path);
std::vector<std::pair<std::string, jubatus::core::fv_converter::datum> > load_csv(const std::string& path);

template< class Period = std::ratio<1> >
std::chrono::duration<double, Period> now() {
  return std::chrono::duration_cast<std::chrono::duration<double, Period> >(
    std::chrono::high_resolution_clock::now().time_since_epoch());
}

template< typename F, class Period = std::ratio<1> >
std::chrono::duration<double, Period> stopwatch(F&& func) {
  auto start = std::chrono::high_resolution_clock::now();
  func();
  return std::chrono::duration_cast<std::chrono::duration<double, Period> >(
    std::chrono::high_resolution_clock::now() - start);
}
std::shared_ptr<jubatus::core::driver::classifier> create_classifier(const std::string& config);

#endif
