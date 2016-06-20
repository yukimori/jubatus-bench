#include "common.hpp"
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <malloc.h>

#include <jubatus/util/data/unordered_map.h>
#include <jubatus/core/common/jsonconfig.hpp>
#include <jubatus/core/fv_converter/converter_config.hpp>
#include <jubatus/core/fv_converter/datum_to_fv_converter.hpp>
#include <jubatus/core/storage/column_table.hpp>
#include <jubatus/core/storage/storage_factory.hpp>
#include <jubatus/core/classifier/classifier_factory.hpp>
#include <jubatus/core/nearest_neighbor/nearest_neighbor_factory.hpp>
#include <jubatus/core/recommender/recommender_factory.hpp>

using jubatus::core::fv_converter::datum;
using jubatus::core::fv_converter::datum_to_fv_converter;
using jubatus::util::data::unordered_map;

Args parse_args(const std::string& api, int argc, char **argv)
{
  typedef unordered_map<std::string, bool*> flag_ops_t;
  Args out;
  flag_ops_t flag_ops;
  flag_ops.insert(std::make_pair("--classify", &out.classify));
  flag_ops.insert(std::make_pair("--train", &out.train));
  flag_ops.insert(std::make_pair("--validate", &out.validate));
  flag_ops.insert(std::make_pair("--similar-row", &out.similar_row));
  flag_ops.insert(std::make_pair("--decode-row", &out.decode_row));
  flag_ops.insert(std::make_pair("--cv", &out.cv));
  flag_ops.insert(std::make_pair("--shuffle", &out.shuffle));

  for (int i = 1; i < argc; ++i) {
    if (argv[i][0] != '-') {
      out.files.push_back(argv[i]);
      continue;
    }

    const std::string flg (argv[i]);
    {
      flag_ops_t::iterator it = flag_ops.find(flg);
      if (it != flag_ops.end()) {
        *(it->second) = true;
        continue;
      }
    }

    if (++i == argc)
      break;
    const std::string val (argv[i]);
    if (flg == "-f" || flg == "--config") {
      std::string err = picojson::parse(out.config, load_text(val));
      if (!err.empty())
        throw std::runtime_error(err);
      continue;
    }
    if (flg == "-n" || flg == "--count") {
      out.count = std::atoi(val.c_str());
      continue;
    }
    if (flg == "-s" || flg == "--size") {
      out.size = std::atoi(val.c_str());
      continue;
    }
  }

  if (api == "classifier") {
    if (!(out.train || out.validate || out.classify || out.cv)) {
      out.train = true;
      out.classify = true;
    }
  } else if (api == "recommender") {
    if (!(out.train || out.similar_row || out.decode_row)) {
      out.train = true;
      out.similar_row = true;
      out.decode_row = true;
    }
  }
  return out;
}

std::string load_text(const std::string& path)
{
  std::string out;
  std::ifstream f(path.c_str(), std::ios::in);
  f.seekg(0, std::ios_base::end);
  out.resize(f.tellg());
  f.seekg(0);
  f.read(const_cast<char*>(out.data()), out.size());
  return out;
}

data_t load_csv(const std::string& path)
{
    static const std::string SEP(",\"");
    data_t data;
    std::ifstream f(path.c_str(), std::ios::in);
    std::string line, cache;
    std::vector<std::string> keys;
    std::vector<std::string> cols;
    while (f && std::getline(f, line)) {
        std::string::size_type p0 = 0, prev = 0, p1;
        while ((p0 = line.find_first_of(SEP, p0)) != std::string::npos) {
            if (line[p0] == ',') {
                // found comma
                cols.push_back(line.substr(prev, p0 - prev));
            } else {
                // found double-quote
                p0 = p0 + 1;
                prev = p0;
                cache.clear();
                do {
                    while ((p1 = line.find_first_of('"', p0)) != std::string::npos) {
                        if (p1 + 1 < line.size() && line[p1 + 1] == '"') {
                            cache += line.substr(prev, p1 - prev + 1);
                            p0 = p1 + 2;
                            prev = p0;
                            continue;
                        }
                        cache += line.substr(prev, p1 - prev);
                        cols.push_back(cache);
                        p0 = p1 + 1;
                        break;
                    }
                    if (p1 != std::string::npos)
                        break;
                    cache += line.substr(prev, line.size() - prev) + "\n";
                    if (!f || !std::getline(f, line)) {
                        line.clear();
                        break;
                    }
                    p0 = prev = 0;
                } while (true);
            }
            p0 = p0 + 1;
            prev = p0;
        }
        if (prev <= line.size())
            cols.push_back(line.substr(prev, line.size() - prev));

        if (cols.size() >= 2) {
            datum d;
            for (size_t i = keys.size(); i < cols.size(); ++i) {
                std::ostringstream s;
                s << i;
                keys.push_back(s.str());
            }
            for (size_t i = 1; i < cols.size(); ++i) {
                double num = std::atof(cols[i].c_str());
                if (num == 0.0 && (cols[i].size() == 0 || cols[i][0] != '0')) {
                    d.string_values_.push_back(std::make_pair(keys[i], cols[i]));
                } else {
                    d.num_values_.push_back(std::make_pair(keys[i], num));
                }
            }
            data.push_back(std::make_pair(cols[0], d));
        }
        cols.clear();
    }
    return data;
}

shared_ptr<jubatus::core::driver::classifier> create_classifier(const std::string& config)
{
    jubatus::util::text::json::json cfg = jubatus::util::lang::lexical_cast<jubatus::util::text::json::json>(config);
    jubatus::core::fv_converter::converter_config fvconv_config;
    jubatus::util::text::json::from_json(cfg["converter"], fvconv_config);
    shared_ptr<datum_to_fv_converter> fvconv = jubatus::core::fv_converter::make_fv_converter(fvconv_config, NULL);
    return shared_ptr<jubatus::core::driver::classifier>(new jubatus::core::driver::classifier(
        jubatus::core::classifier::classifier_factory::create_classifier(
            static_cast<jubatus::util::text::json::json_string*>(
                cfg["method"].get())->get(),
            jubatus::core::common::jsonconfig::config(cfg["parameter"]),
            jubatus::core::storage::storage_factory::create_storage("local")),
        fvconv));
}

shared_ptr<jubatus::core::driver::recommender> create_recommender(const std::string& config) {
  std::string my_id;
  jubatus::util::text::json::json cfg = jubatus::util::lang::lexical_cast<jubatus::util::text::json::json>(config);
  jubatus::core::fv_converter::converter_config fvconv_config;
  jubatus::util::text::json::from_json(cfg["converter"], fvconv_config);
  return shared_ptr<jubatus::core::driver::recommender>(new jubatus::core::driver::recommender(
    jubatus::core::recommender::recommender_factory::create_recommender(
      static_cast<jubatus::util::text::json::json_string*>(cfg["method"].get())->get(),
      jubatus::core::common::jsonconfig::config(cfg["parameter"]),
      my_id),
    jubatus::core::fv_converter::make_fv_converter(fvconv_config, NULL)));
}

shared_ptr<jubatus::core::driver::nearest_neighbor> create_nearest_neighbor(const std::string& config) {
  std::string my_id;
  shared_ptr<jubatus::core::storage::column_table> table(new jubatus::core::storage::column_table);
  jubatus::util::text::json::json cfg = jubatus::util::lang::lexical_cast<jubatus::util::text::json::json>(config);
  jubatus::core::fv_converter::converter_config fvconv_config;
  jubatus::util::text::json::from_json(cfg["converter"], fvconv_config);
  return shared_ptr<jubatus::core::driver::nearest_neighbor>(new jubatus::core::driver::nearest_neighbor(
    jubatus::core::nearest_neighbor::create_nearest_neighbor(
      static_cast<jubatus::util::text::json::json_string*>(cfg["method"].get())->get(),
      jubatus::core::common::jsonconfig::config(cfg["parameter"]),
      table, my_id),
    jubatus::core::fv_converter::make_fv_converter(fvconv_config, NULL)));
}

double get_memory_usage() {
  malloc_trim(0);
  struct mallinfo mi = mallinfo();
  uint64_t bytes = 0;
  bytes += *(unsigned int*)&(mi.hblkhd);
  bytes += *(unsigned int*)&(mi.uordblks);
  return static_cast<double>(bytes) / (1024.0 * 1024.0);
}
