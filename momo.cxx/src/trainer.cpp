#include "trainer.h"

#include <chrono>
#include <cstring>
#include <ctime>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>

#include <crfsuite.hpp>
#include "crf_features.h"
#include "utils.h"

namespace momo {

// TSV パーサー
// ---------------------------------------------------------------------------

struct TsvRow {
    std::string char_str;
    std::string label;
    std::string ctype;
    std::string biose;
    int orig_idx = 0;
};

struct Sentence {
    std::vector<TsvRow> rows;
};

static std::vector<Sentence> load_tsv(const std::string& path) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error("TSVファイルを開けませんでした: " + path);

    std::vector<Sentence> sentences;
    Sentence current;
    std::string line;
    int lineno = 0;

    while (std::getline(f, line)) {
        ++lineno;
        if (!line.empty() && line.back() == '\r') line.pop_back();

        if (line.empty() || line[0] == '#') {
            if (!current.rows.empty()) {
                sentences.push_back(std::move(current));
                current = Sentence{};
            }
            continue;
        }

        std::vector<std::string> cols;
        std::string col;
        for (char c : line) {
            if (c == '\t') { cols.push_back(col); col.clear(); }
            else col += c;
        }
        cols.push_back(col);

        if (cols.size() < 4) {
            std::cerr << "⚠️  Line " << lineno << ": 列数不足 (skip)\n";
            continue;
        }

        TsvRow row;
        row.char_str = cols[0];
        row.label    = cols[1];
        row.ctype    = cols[2];
        row.biose    = cols[3];
        row.orig_idx = (cols.size() >= 5) ? std::stoi(cols[4]) : 0;
        current.rows.push_back(std::move(row));
    }
    if (!current.rows.empty()) sentences.push_back(std::move(current));

    return sentences;
}

// ---------------------------------------------------------------------------
// ラベル分離: 読みラベルと境界ラベルに分割
// Python の _split_labels() に相当
// ---------------------------------------------------------------------------

static void split_labels(
    const std::vector<std::string>& raw_labels,
    std::vector<std::string>& y_read,
    std::vector<std::string>& y_boundary)
{
    y_read.clear();
    y_boundary.clear();
    for (const auto& label : raw_labels) {
        bool has_split = (label.find(MORA_SPLIT) != std::string::npos);

        // 読みラベルから "+S" を除去
        std::string read_label = label;
        size_t pos = read_label.find(MORA_SPLIT);
        if (pos != std::string::npos) read_label.erase(pos, strlen(MORA_SPLIT));

        y_read.push_back(read_label);

        // LABEL_CONTINUE は境界を持たない
        if (label == "---") {
            y_boundary.push_back("0");
        } else {
            y_boundary.push_back(has_split ? "1" : "0");
        }
    }
}

// ---------------------------------------------------------------------------
// Trainer の実装
// ---------------------------------------------------------------------------

Trainer::Trainer() {}

void Trainer::train(const std::string& tsv_path, bool transitions) {
    auto t_start = std::chrono::high_resolution_clock::now();

    // --- 1. TSV読み込み ---
    std::cerr << "📖 TSV読み込み: " << tsv_path << "\n";
    auto sentences = load_tsv(tsv_path);
    std::cerr << "   " << sentences.size() << " 文\n";

    // --- 2. 特徴量・ラベル列の構築 ---
    std::vector<CRFSuite::ItemSequence> X;
    std::vector<std::vector<std::string>> Y_read;
    std::vector<std::vector<std::string>> Y_boundary;

    for (const auto& sent : sentences) {
        if (sent.rows.empty()) continue;

        std::vector<SourceUnit> source_seq;
        std::vector<std::string> raw_labels;

        for (const auto& row : sent.rows) {
            std::u32string text32 = utf8_to_utf32(row.char_str);
            CharType ctype = get_char_type(text32.empty() ? 0 : text32[0]);
            source_seq.push_back({text32, row.orig_idx, ctype});
            raw_labels.push_back(row.label);
        }

        auto features = compute_source_features(source_seq);

        CRFSuite::ItemSequence xseq;
        xseq.resize(features.size());
        for (size_t i = 0; i < features.size(); ++i) {
            for (const auto& [feat, val] : features[i]) {
                xseq[i].push_back(CRFSuite::Attribute(feat, val));
            }
        }
        X.push_back(std::move(xseq));

        std::vector<std::string> y_read, y_boundary;
        split_labels(raw_labels, y_read, y_boundary);
        Y_read.push_back(std::move(y_read));
        Y_boundary.push_back(std::move(y_boundary));
    }

    std::cerr << "✅ 特徴量構築完了 (" << X.size() << " シーケンス)\n";

    // --- 3. 出力パスの決定 ---
    std::string base = tsv_path;
    size_t dot = base.rfind('.');
    if (dot != std::string::npos) base = base.substr(0, dot);

    std::string path_read     = base + "_read.crfsuite";
    std::string path_boundary = base + "_boundary.crfsuite";

    // --- 4. 共通パラメータ設定 ---
    auto set_params = [&](CRFSuite::Trainer& trainer) {
        trainer.select("lbfgs", "crf1d");
        trainer.set("c1", "0.1");
        trainer.set("c2", "0.01");
        trainer.set("max_iterations", "70");
        trainer.set("feature.possible_transitions", transitions ? "1" : "0");
        trainer.set("feature.possible_states", "0");
    };

    // --- 5. 読みモデルの学習 ---
    std::cerr << "\n🏋️  [読みモデル] 学習開始: " << path_read << "\n";
    {
        CRFSuite::Trainer trainer;
        set_params(trainer);
        for (size_t i = 0; i < X.size(); ++i) {
            trainer.append(X[i], Y_read[i], 0);
        }
        trainer.train(path_read, -1);
    }
    std::cerr << "💾 [読みモデル] 完了: " << path_read << "\n";

    // --- 6. 境界モデルの学習 ---
    std::cerr << "\n🏋️  [境界モデル] 学習開始: " << path_boundary << "\n";
    {
        CRFSuite::Trainer trainer;
        set_params(trainer);
        for (size_t i = 0; i < X.size(); ++i) {
            trainer.append(X[i], Y_boundary[i], 0);
        }
        trainer.train(path_boundary, -1);
    }
    std::cerr << "💾 [境界モデル] 完了: " << path_boundary << "\n";

    // --- 7. version_info.json の生成 ---
    std::string json_path = base + "_version_info.json";
    {
        std::time_t now = std::time(nullptr);
        char timebuf[64];
        std::strftime(timebuf, sizeof(timebuf), "%Y-%m-%dT%H:%M:%SZ",
                      std::gmtime(&now));

        // ベースファイル名のみ（ディレクトリなし）
        std::string bname = base;
        size_t slash = bname.rfind('/');
        if (slash != std::string::npos) bname = bname.substr(slash + 1);
#ifdef _WIN32
        slash = bname.rfind('\\');
        if (slash != std::string::npos) bname = bname.substr(slash + 1);
#endif

        std::ofstream jf(json_path);
        jf << "{\n"
           << "  \"trained_at\": \"" << timebuf << "\",\n"
           << "  \"model_read\": \"" << bname << "_read.crfsuite\",\n"
           << "  \"model_boundary\": \"" << bname << "_boundary.crfsuite\",\n"
           << "  \"transitions\": " << (transitions ? "true" : "false") << "\n"
           << "}\n";
    }
    std::cerr << "📄 version_info: " << json_path << "\n";

    // --- 8. ZIP化の案内 ---
    // ZIPの作成はzipコマンドまたはPythonスクリプトに委ねる
    // （プラットフォーム依存を避けるため）
    std::cerr << "\n📦 ZIPパッケージ作成:\n";
    std::cerr << "   python3 -c \"\n";
    std::cerr << "import zipfile, json\n";
    std::cerr << "with zipfile.ZipFile('" << base << ".zip', 'w', zipfile.ZIP_DEFLATED) as zf:\n";
    std::cerr << "    zf.write('" << path_read << "', '" << base.substr(base.rfind('/') + 1) << "_read.crfsuite')\n";
    std::cerr << "    zf.write('" << path_boundary << "', '" << base.substr(base.rfind('/') + 1) << "_boundary.crfsuite')\n";
    std::cerr << "    zf.write('" << json_path << "', 'version_info.json')\n";
    std::cerr << "\"\n";

    auto elapsed = std::chrono::duration<double>(
        std::chrono::high_resolution_clock::now() - t_start).count();
    std::cerr << "\n✅ 学習完了 (合計 " << elapsed << " 秒)\n";
    std::cerr << "   読みモデル:   " << path_read << "\n";
    std::cerr << "   境界モデル:   " << path_boundary << "\n";
}

}  // namespace momo
