#include "predictor.h"

#include <algorithm>
#include <crfsuite.hpp>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace momo {

// ---------------------------------------------------------------------------
// 内部定数
// ---------------------------------------------------------------------------

// 安全な送り仮名（訓読みを誘発）
static bool is_safe_okurigana(char32_t cp) {
  // あいうえおかきくけこたちつてとなにぬねのまみむめもやゆよらりるれろわばびぶべぼ
  static const char32_t table[] = {
      U'あ', U'い', U'う', U'え', U'お', U'か', U'き', U'く', U'け', U'こ', U'た', U'ち', U'つ',
      U'て', U'と', U'な', U'に', U'ぬ', U'ね', U'の', U'ま', U'み', U'む', U'め', U'も', U'や',
      U'ゆ', U'よ', U'ら', U'り', U'る', U'れ', U'ろ', U'わ', U'ば', U'び', U'ぶ', U'べ', U'ぼ',
  };
  for (char32_t c : table) {
    if (cp == c) return true;
  }
  return false;
}

// 漢数字 → ASCII数字 変換
static const char* digit_to_ascii(char32_t cp) {
  switch (cp) {
    case U'〇':
      return "0";
    case U'一':
      return "1";
    case U'二':
      return "2";
    case U'三':
      return "3";
    case U'四':
      return "4";
    case U'五':
      return "5";
    case U'六':
      return "6";
    case U'七':
      return "7";
    case U'八':
      return "8";
    case U'九':
      return "9";
    default:
      return nullptr;
  }
}

// ---------------------------------------------------------------------------
// コンストラクタ / デストラクタ
// ---------------------------------------------------------------------------

Predictor::Predictor(const PredictorConfig& config) : _config(config) {
  _tagger_read = std::make_unique<CRFSuite::Tagger>();
  if (!_tagger_read->open(config.path_read)) {
    throw std::runtime_error("読みモデルを開けませんでした: " + config.path_read);
  }

  if (!config.path_boundary.empty()) {
    _tagger_boundary = std::make_unique<CRFSuite::Tagger>();
    if (!_tagger_boundary->open(config.path_boundary)) {
      throw std::runtime_error("境界モデルを開けませんでした: " + config.path_boundary);
    }
    _has_boundary = true;
  }

  if (!config.path_fallback_dict.empty()) {
    _load_fallback_dict(config.path_fallback_dict);
  }

  if (!config.path_custom_dict.empty()) {
    _load_custom_dict(config.path_custom_dict);
  }
}

Predictor::~Predictor() {
  if (_tagger_read) _tagger_read->close();
  if (_tagger_boundary) _tagger_boundary->close();
}

// ---------------------------------------------------------------------------
// 初期化ヘルパー
// ---------------------------------------------------------------------------

void Predictor::_load_fallback_dict(const std::string& path) {
  std::ifstream f(path);
  if (!f) throw std::runtime_error("フォールバック辞書を開けませんでした: " + path);

  std::string line;
  while (std::getline(f, line)) {
    if (line.empty() || line[0] == '#') continue;
    // フォーマット: 漢字\t音読み\t訓読み
    size_t t1 = line.find('\t');
    if (t1 == std::string::npos) continue;
    size_t t2 = line.find('\t', t1 + 1);
    if (t2 == std::string::npos) continue;

    std::u32string kanji = utf8_to_utf32(line.substr(0, t1));
    std::string on = line.substr(t1 + 1, t2 - t1 - 1);
    std::string kun = line.substr(t2 + 1);

    if (kanji.size() == 1) {
      _fallback_dict[kanji[0]] = {on, kun};
    }
  }
}

void Predictor::_load_custom_dict(const std::string& path) {
  std::ifstream f(path);
  if (!f) throw std::runtime_error("カスタム辞書を開けませんでした: " + path);

  std::string line;
  int lineno = 0;
  while (std::getline(f, line)) {
    ++lineno;
    if (line.empty() || line[0] == '#') continue;

    size_t t = line.find('\t');
    if (t == std::string::npos) {
      throw std::runtime_error("カスタム辞書 " + path + " の " + std::to_string(lineno) +
                               " 行目の形式が不正です（タブ区切りで表層形と読みが必要）");
    }

    std::u32string surface = utf8_to_utf32(line.substr(0, t));
    std::string reading_str = line.substr(t + 1);

    // '/' で分割（バックスラッシュエスケープ対応）
    std::vector<std::string> readings;
    std::string cur;
    for (size_t i = 0; i < reading_str.size(); ++i) {
      if (reading_str[i] == '\\' && i + 1 < reading_str.size()) {
        cur += reading_str[i + 1];
        ++i;
      } else if (reading_str[i] == '/') {
        readings.push_back(cur);
        cur.clear();
      } else {
        cur += reading_str[i];
      }
    }
    readings.push_back(cur);

    if (readings.size() != surface.size()) {
      throw std::runtime_error("カスタム辞書 " + path + " の " + std::to_string(lineno) +
                               " 行目: 表層形の文字数と読みブロック数が一致しません");
    }

    // DictIndex に登録（長い順に並べる）
    char32_t key = surface[0];
    _dict_index[key].push_back({surface, readings});
  }

  // 各エントリを長い順にソート
  for (auto& [key, entries] : _dict_index) {
    std::sort(entries.begin(), entries.end(),
              [](const auto& a, const auto& b) { return a.first.size() > b.first.size(); });
  }
}

// ---------------------------------------------------------------------------
// 推論エントリポイント
// ---------------------------------------------------------------------------

PredictionResult Predictor::predict(const std::string& utf8_text) {
  if (utf8_text.empty()) {
    return PredictionResult{};
  }

  std::u32string text = utf8_to_utf32(utf8_text);

  auto pre = _preprocess(text);
  if (pre.source_seq.empty()) {
    PredictionResult r;
    r.source_text = utf8_text;
    r.src_to_kana_index.resize(text.size());
    return r;
  }

  std::vector<std::string> raw_labels, boundary_labels;
  std::vector<float> raw_confidences;
  _run_inference(pre.source_seq, raw_labels, raw_confidences, boundary_labels);

  auto refined = _refine(pre.source_seq, raw_labels, raw_confidences, boundary_labels, pre.bypass_indices,
                         pre.ascii_overrides, pre.dict_overrides);

  return _assemble(utf8_text, text, pre.source_seq, refined, pre.bypass_indices);
}

// ---------------------------------------------------------------------------
// ステップ 1: 前処理
// ---------------------------------------------------------------------------

Predictor::PreprocessResult Predictor::_preprocess(const std::u32string& text) const {
  PreprocessResult result;
  auto units = get_units(text);

  int char_idx = 0;
  for (const auto& unit : units) {
    // ASCIIバイパス判定（ALPHA または NUM かつ印字可能ASCII文字列）
    bool is_ascii_bypass = false;
    if (unit.ctype == CharType::ALPHA || unit.ctype == CharType::NUM) {
      is_ascii_bypass = true;
      if (unit.text.empty() || unit.text.front() == U' ' || unit.text.front() == U'\t' || unit.text.back() == U' ' ||
          unit.text.back() == U'\t') {
        is_ascii_bypass = false;
      } else {
        for (char32_t c : unit.text) {
          if (!((c >= 0x21 && c <= 0x7E) || c == U' ' || c == U'\t')) {
            is_ascii_bypass = false;
            break;
          }
        }
      }
    }

    for (size_t i = 0; i < unit.text.size(); ++i) {
      result.source_seq.push_back({std::u32string(1, unit.text[i]), unit.orig_idx + static_cast<int>(i), unit.ctype});
      if (is_ascii_bypass) {
        result.bypass_indices.insert(char_idx);
        // 先頭文字には単語全体のUTF-8を、後続は LABEL_CONTINUE を設定
        if (i == 0) {
          result.ascii_overrides[char_idx] = utf32_to_utf8(unit.text);
        } else {
          result.ascii_overrides[char_idx] = LABEL_CONTINUE;
        }
      }
      ++char_idx;
    }
  }

  // カスタム辞書の最長一致スキャン
  if (!_dict_index.empty()) {
    int i = 0;
    int n = static_cast<int>(result.source_seq.size());
    while (i < n) {
      if (result.bypass_indices.count(i)) {
        ++i;
        continue;
      }
      auto [length, readings] = _find_longest_match(result.source_seq, i);
      if (length > 0) {
        for (int j = 0; j < length; ++j) {
          result.dict_overrides[i + j] = readings[j];
        }
        i += length;
      } else {
        ++i;
      }
    }
  }

  return result;
}

// ---------------------------------------------------------------------------
// ステップ 2: 推論
// ---------------------------------------------------------------------------

void Predictor::_run_inference(const std::vector<SourceUnit>& source_seq, std::vector<std::string>& raw_labels,
                               std::vector<float>& raw_confidences, std::vector<std::string>& boundary_labels) {
  auto features = compute_source_features(source_seq);
  int n = static_cast<int>(features.size());

  // FeatureDict → CRFSuite::ItemSequence に変換
  CRFSuite::ItemSequence xseq;
  xseq.resize(n);
  for (int i = 0; i < n; ++i) {
    for (const auto& [feat, val] : features[i]) {
      xseq[i].push_back(CRFSuite::Attribute(feat, val));
    }
  }
  // 読みモデル: viterbi でラベル列、marginal で確率を取得
  {
    _tagger_read->set(xseq);
    raw_labels = _tagger_read->viterbi();

    // marginal は set() 後に呼ぶ（pycrfsuite と同じパス）
    _tagger_read->set(xseq);
    raw_confidences.resize(n);
    for (int i = 0; i < n; ++i) {
      raw_confidences[i] = static_cast<float>(_tagger_read->marginal(raw_labels[i], i));
    }
  }

  // 境界モデル: viterbi のみ（marginal 不要）
  if (_has_boundary) {
    _tagger_boundary->set(xseq);
    auto b_labels = _tagger_boundary->viterbi();
    boundary_labels.resize(n);
    for (int i = 0; i < n; ++i) {
      boundary_labels[i] = b_labels[i];
    }
  } else {
    boundary_labels.assign(n, "0");
  }
}

// ---------------------------------------------------------------------------
// ステップ 3: 後処理
// ---------------------------------------------------------------------------

Predictor::RefineResult Predictor::_refine(const std::vector<SourceUnit>& source_seq,
                                           const std::vector<std::string>& raw_labels,
                                           const std::vector<float>& raw_confidences,
                                           const std::vector<std::string>& boundary_labels,
                                           const std::unordered_set<int>& bypass_indices,
                                           const std::unordered_map<int, std::string>& ascii_overrides,
                                           const std::unordered_map<int, std::string>& dict_overrides) const {
  int n = static_cast<int>(source_seq.size());
  RefineResult result;
  result.labels.resize(n);
  result.confidences.resize(n, 0.0f);
  result.has_splits.resize(n, false);
  result.decision_sources.resize(n, DecisionSource::CRF);

  std::string last_fallback;
  int parent_idx = -1;

  for (int i = 0; i < n; ++i) {
    char32_t ch = source_seq[i].text.empty() ? 0 : source_seq[i].text[0];
    CharType ctype = source_seq[i].ctype;
    const std::string& raw = raw_labels[i];

    if (raw != LABEL_CONTINUE && raw != LABEL_SKIP) {
      parent_idx = i;
    }

    if (bypass_indices.count(i)) {
      result.labels[i] = ascii_overrides.at(i);
      result.confidences[i] = 1.0f;
      result.decision_sources[i] = DecisionSource::BYPASS;
      result.has_splits[i] = false;
      last_fallback.clear();
      continue;
    }

    if (dict_overrides.count(i)) {
      result.labels[i] = dict_overrides.at(i);
      result.confidences[i] = 1.0f;
      result.decision_sources[i] = DecisionSource::DICT;
      result.has_splits[i] = (boundary_labels[i] == "1");
      last_fallback.clear();
      continue;
    }

    std::string label = raw;
    float confidence = raw_confidences.empty() ? 1.0f : raw_confidences[i];
    DecisionSource decision = DecisionSource::CRF;

    // 文字消失バグの救済ロジック
    if (label == LABEL_CONTINUE && (parent_idx == -1 || bypass_indices.count(parent_idx))) {
      auto it = _fallback_dict.find(ch);
      if (ctype == CharType::KANJI && it != _fallback_dict.end()) {
        label = it->second.on;
      } else {
        label = utf32_to_utf8(source_seq[i].text);
      }
      confidence = 0.0f;
      decision = DecisionSource::FALLBACK_ORPHAN;
    } else if (ctype == CharType::JAPANESE_NUMERIC) {
      // JAPANESE_NUMERIC: confidence が閾値未満のときルールベース変換
      // C API では marginal が取れないため常にルールベースを使う
      const char* digit = digit_to_ascii(ch);
      if (digit) {
        label = digit;
        decision = DecisionSource::FALLBACK_NUMERIC;
      } else {
        char32_t left_ch = (i > 0) ? (source_seq[i - 1].text.empty() ? 0 : source_seq[i - 1].text[0]) : 0;
        CharType left_ctype = (i > 0) ? source_seq[i - 1].ctype : CharType::OTHER;
        CharType right_ctype = (i < n - 1) ? source_seq[i + 1].ctype : CharType::OTHER;
        label = _kurai_fallback(ch, left_ch, left_ctype, right_ctype);
        decision = DecisionSource::FALLBACK_NUMERIC;
      }
      last_fallback.clear();
    } else {
      // KANJI フォールバック・々繰り返し
      auto it = _fallback_dict.find(ch);
      if (ctype == CharType::KANJI && it != _fallback_dict.end()) {
        char32_t next_ch = (i < n - 1) ? (source_seq[i + 1].text.empty() ? 0 : source_seq[i + 1].text[0]) : 0;
        CharType next_ctype = (i < n - 1) ? source_seq[i + 1].ctype : CharType::OTHER;
        if (next_ctype == CharType::HIRAGANA && is_safe_okurigana(next_ch)) {
          label = it->second.kun;
        } else {
          label = it->second.on;
        }
        last_fallback = label;
        decision = DecisionSource::FALLBACK_KANJI;
      } else if (ch == U'々' && !last_fallback.empty()) {
        label = last_fallback;
        decision = DecisionSource::FALLBACK_REPEAT;
      } else {
        if (ctype != CharType::SYMBOL) last_fallback.clear();
      }
    }

    result.labels[i] = label;
    result.confidences[i] = confidence;
    result.has_splits[i] = (boundary_labels[i] == "1");
    result.decision_sources[i] = decision;
  }

  return result;
}

// ---------------------------------------------------------------------------
// ステップ 4: 結果の組み立て
// ---------------------------------------------------------------------------

PredictionResult Predictor::_assemble(const std::string& utf8_text, const std::u32string& text,
                                      const std::vector<SourceUnit>& source_seq, const RefineResult& refined,
                                      const std::unordered_set<int>& bypass_indices) const {
  PredictionResult r;
  r.source_text = utf8_text;
  r.src_to_kana_index.resize(text.size());

  int kana_pos = 0;
  int n = static_cast<int>(source_seq.size());

  for (int i = 0; i < n; ++i) {
    int orig_idx = source_seq[i].orig_idx;
    const std::string& label = refined.labels[i];
    float conf = refined.confidences[i];
    DecisionSource dec = refined.decision_sources[i];

    if (label == LABEL_SKIP) {
      // 何も出力しない
    } else if (label != LABEL_CONTINUE) {
      if (bypass_indices.count(i)) {
        // ASCIIバイパス: label はブロック全体なので文字ごとに orig_idx をずらす
        std::u32string label32 = utf8_to_utf32(label);
        for (size_t j = 0; j < label32.size(); ++j) {
          std::string ch_utf8 = char32_to_utf8(label32[j]);
          r.kana_text += ch_utf8;
          int target_idx = orig_idx + static_cast<int>(j);
          r.kana_to_src_index.push_back(target_idx);
          r.confidences.push_back(conf);
          r.decision_sources.push_back(dec);
          if (target_idx < static_cast<int>(r.src_to_kana_index.size())) {
            r.src_to_kana_index[target_idx].push_back(kana_pos);
          }
          ++kana_pos;
        }
      } else {
        for (char c_utf8 : label) {
          // label はUTF-8文字列なので1バイトずつではなくコードポイント単位で扱う
          // ここでは label 全体を一度に追加し、kana_to_src_index は先頭orig_idxを使う
          // ループはUTF-8バイト単位ではなくコードポイント単位にすべきだが、
          // confidences/decision_sourcesはラベル先頭1回だけ登録する
          (void)c_utf8;
        }
        // 正しいコードポイント単位の処理
        r.kana_text += label;
        std::u32string label32 = utf8_to_utf32(label);
        for (size_t j = 0; j < label32.size(); ++j) {
          r.kana_to_src_index.push_back(orig_idx);
          r.confidences.push_back(conf);
          r.decision_sources.push_back(dec);
          if (orig_idx < static_cast<int>(r.src_to_kana_index.size())) {
            r.src_to_kana_index[orig_idx].push_back(kana_pos);
          }
          ++kana_pos;
        }
      }
    }

    if (refined.has_splits[i]) {
      r.kana_text += ' ';
      r.kana_to_src_index.push_back(orig_idx);
      r.confidences.push_back(conf);
      r.decision_sources.push_back(dec);
      if (orig_idx < static_cast<int>(r.src_to_kana_index.size())) {
        r.src_to_kana_index[orig_idx].push_back(kana_pos);
      }
      ++kana_pos;
    }
  }

  return r;
}

// ---------------------------------------------------------------------------
// フォールバックロジック
// ---------------------------------------------------------------------------

std::string Predictor::_kurai_fallback(char32_t ch, char32_t left_char, CharType left_ctype,
                                       CharType right_ctype) const {
  bool left_is_num = (left_ctype == CharType::JAPANESE_NUMERIC);
  bool right_is_num = (right_ctype == CharType::JAPANESE_NUMERIC);

  if (ch == U'十') {
    if (left_is_num && right_is_num) return LABEL_SKIP;
    if (left_is_num && !right_is_num) return "0";
    if (!left_is_num && right_is_num) return "1";
    return "10";
  }
  if (ch == U'百') {
    if (left_is_num && right_is_num) return "0";
    if (left_is_num && !right_is_num) return "00";
    if (!left_is_num && right_is_num) return "1";
    return "100";
  }
  if (ch == U'千') {
    if (right_is_num) return "0";
    return (left_char == U'三') ? utf32_to_utf8(U"\u30BC\u30F3")   // ゼン
                                : utf32_to_utf8(U"\u30BB\u30F3");  // セン
  }
  if (ch == U'万') return utf32_to_utf8(U"\u30DE\u30F3");        // マン
  if (ch == U'億') return utf32_to_utf8(U"\u30AA\u30AF");        // オク
  if (ch == U'兆') return utf32_to_utf8(U"\u30C1\u30E7\u30FC");  // チョー

  return utf32_to_utf8(std::u32string(1, ch));
}

// ---------------------------------------------------------------------------
// カスタム辞書ヘルパー
// ---------------------------------------------------------------------------

std::pair<int, std::vector<std::string>> Predictor::_find_longest_match(const std::vector<SourceUnit>& source_seq,
                                                                        int pos) const {
  if (source_seq[pos].text.empty()) return {0, {}};
  char32_t key = source_seq[pos].text[0];

  auto it = _dict_index.find(key);
  if (it == _dict_index.end()) return {0, {}};

  int n = static_cast<int>(source_seq.size());
  for (const auto& [surface, readings] : it->second) {
    int len = static_cast<int>(surface.size());
    if (pos + len > n) continue;
    bool match = true;
    for (int j = 0; j < len; ++j) {
      if (source_seq[pos + j].text.empty() || source_seq[pos + j].text[0] != surface[j]) {
        match = false;
        break;
      }
    }
    if (match) return {len, readings};
  }
  return {0, {}};
}

}  // namespace momo
