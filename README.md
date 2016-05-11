# NN高速化のベンチマーク実行
jubatus-benchをコンパイルし、benchmark.shを実行する。
測定結果はdata.csvに出力される

# jubatus-bench

[jubatus-core](https://github.com/jubatus/jubatus_core)のベンチマークプログラム．
[Jubatusカジュアルもくもく会 #2](http://jubatus.connpass.com/event/25728/)でNN高速化の効果測定用に作ったもの．

# コンパイル方法

* C++11に対応したgccが必要
* 1つ上のフォルダにjubatus_coreが必要
* jubatus_coreの中にlibフォルダがあり，その中にjubatus_coreの全ての*.soへのシンボリックリンクが作成されている必要あり
* (上記の条件の詳細はMakefileを参照)

```
$ make
```

# 使い方

## Classifier

* 教師データ形式
  * CSVファイルの最初のカラムはラベルである必要があるほか，ヘッダ行を含んではいけません
  * CSVファイルは複数指定することが出来ます
* 引数
  * --config オプションは必須でJubatusのJSONコンフィグを指定します
  * --trainを指定すると指定したCSVのデータを使って学習性能を測定します
  * --classifyを指定するとCSVのデータ全件を使って分類性能を測定します
    * --trainが指定されている場合は指定されたCSVデータを使って学習したモデルを利用します
    * --trainが指定されていない場合は，指定されたCSVファイル名+".model"という名前のモデルファイルを利用します
    * -n <N> or --count <N> を指定すると，指定した件数だけの分類を実施します
  * --validateを指定するとCSVのデータ全件を使って分類の成功/失敗を数えます
    * --trainフラグの指定有無により挙動の変化は--classifyと同様です
  * --cvを指定すると4分割交差検証を実施します
    * --shuffleフラグを指定するとデータをシャッフルしてから4分割します

```
$ cat nn.json
{
  "converter": {
    "num_filter_types": {},
    "num_filter_rules": [],
    "string_filter_types": {},
    "string_filter_rules": [],
    "num_types": {},
    "num_rules": [
      { "key": "*", "type": "num" }
    ],
    "string_types": {},
    "string_rules": [
      { "key": "*", "type": "space", "sample_weight": "bin", "global_weight": "bin" }
    ]
  },
  "method": "NN",
  "parameter": {
    "method": "euclid_lsh",
    "parameter": {
      "hash_num": 8192
    },
    "nearest_neighbor_num": 128,
    "local_sensitivity": 1
  }
};
$ ./classifier --config nn.json 20news-18828.csv dorothea_train.csv iris.csv
20news-18828.csv
   train: 99164.1 ms (18828 records)
          5.26684 ms (avg latency)
          189.867 ops
classify: 152699 ms (18828 records)
          8.11021 ms (avg latency)
          123.301 ops

dorothea_train.csv
   train: 22009.4 ms (800 records)
          27.5117 ms (avg latency)
          36.3481 ops
classify: 21916.8 ms (800 records)
          27.396 ms (avg latency)
          36.5017 ops

iris.csv
   train: 24.4667 ms (150 records)
          0.163111 ms (avg latency)
          6130.78 ops
classify: 32.274 ms (150 records)
          0.21516 ms (avg latency)
          4647.7 ops
$ ./classifier --config nn.json --train iris.csv
iris.csv
   train: 53.7946 ms (150 records)
          0.358631 ms (avg latency)
          2788.38 ops
$ ls -l iris.csv.model
-rw-r--r-- 1 kazuki kazuki 181495 Mar 26 01:01 iris.csv.model
$ ./classifier --config nn.json --classify iris.csv
iris.csv
classify: 77.2136 ms (150 records)
          0.514757 ms (avg latency)
          1942.66 ops
$ ./classifier --config nn.json --validate iris.csv
iris.csv
validate: ok=145, ng=5
$ ./classifier --config nn.json --cv iris.csv
iris.csv
[4-fold cross-validation]
   train: 44.662 ms (111 records)
          0.40236 ms (avg latency)
          2485.34 ops
classify: 16.6644 ms (39 records)
          0.427293 ms (avg latency)
          2340.31 ops
validate: ok=3, ng=36
$ ./classifier --config nn.json --classify -n 10 iris.csv
iris.csv
classify: 4.17518 ms (10 records)
          0.417518 ms (avg latency)
          2395.11 ops
```
