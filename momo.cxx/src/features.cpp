#include "features.h"

#include <string>

namespace momo {

// ---------------------------------------------------------------------------
// 内部定数
// ---------------------------------------------------------------------------

// 隣接文脈次第で JAPANESE_NUMERIC に昇格する位取り文字
static bool is_kurai_char(char32_t cp) {
  // 十百千万億兆
  return cp == U'十' || cp == U'百' || cp == U'千' || cp == U'万' || cp == U'億' || cp == U'兆';
}

// 小書きかな（ひらがな・カタカナ）
static bool is_small_kana(char32_t cp) {
  // ぁぃぅぇぉゃゅょゎ
  if (cp == U'ぁ' || cp == U'ぃ' || cp == U'ぅ' || cp == U'ぇ' || cp == U'ぉ' || cp == U'ゃ' || cp == U'ゅ' ||
      cp == U'ょ' || cp == U'ゎ')
    return true;
  // ァィゥェォャュョヮヵヶ
  if (cp == U'ァ' || cp == U'ィ' || cp == U'ゥ' || cp == U'ェ' || cp == U'ォ' || cp == U'ャ' || cp == U'ュ' ||
      cp == U'ョ' || cp == U'ヮ' || cp == U'ヵ' || cp == U'ヶ')
    return true;
  return false;
}

static bool is_ascii_printable(char32_t cp) { return cp >= 0x21 && cp <= 0x7E; }

static bool is_ascii_space(char32_t cp) { return cp == U' ' || cp == U'\t'; }

// ---------------------------------------------------------------------------
// ブロック種別マッチ関数
// それぞれ「マッチしたら true を返し、pos と out を更新する」規約。
// pos はブロック終端の次を指す。
// ---------------------------------------------------------------------------

/// [...]ブラケットブロック。 '[' を含まない内側テキストを返す。
static bool try_match_bracket(const std::u32string& text, size_t pos, std::u32string& out, size_t& next_pos) {
  if (text[pos] != U'[') return false;
  size_t end = pos + 1;
  while (end < text.size() && text[end] != U']') ++end;
  if (end >= text.size()) return false;  // 閉じ括弧なし → マッチ失敗
  out = text.substr(pos + 1, end - pos - 1);
  next_pos = end + 1;
  return true;
}

/// かな（ひらがな/カタカナ）＋小書きかなの連続ブロック。
/// 先頭が通常かな、2文字目以降に小書きかなが続く場合にまとめる。
static bool try_match_kana_block(const std::u32string& text, size_t pos, std::u32string& out, size_t& next_pos) {
  char32_t c0 = text[pos];
  CharType t0 = get_basic_char_category(c0);
  if (t0 != CharType::HIRAGANA && t0 != CharType::KATAKANA) return false;
  if (is_small_kana(c0)) return false;  // 先頭が小書きなら単独扱い

  size_t end = pos + 1;
  while (end < text.size() && is_small_kana(text[end])) ++end;
  if (end == pos + 1) return false;  // 後続の小書きなし → 単独扱い

  out = text.substr(pos, end - pos);
  next_pos = end;
  return true;
}

/// ASCII印字文字（0x21-0x7E）＋スペース/タブの連続ブロック。
/// 先頭・末尾はスペース/タブでないこと。
static bool try_match_ascii_block(const std::u32string& text, size_t pos, std::u32string& out, size_t& next_pos) {
  if (!is_ascii_printable(text[pos])) return false;

  size_t end = pos + 1;
  while (end < text.size() && (is_ascii_printable(text[end]) || is_ascii_space(text[end]))) {
    ++end;
  }
  // 末尾のスペース/タブを切り詰める
  while (end > pos + 1 && is_ascii_space(text[end - 1])) --end;

  out = text.substr(pos, end - pos);
  next_pos = end;
  return true;
}

/// 空白の連続ブロック。
static bool try_match_space(const std::u32string& text, size_t pos, std::u32string& out, size_t& next_pos) {
  char32_t c = text[pos];
  if (c != U' ' && c != U'\t' && c != U'\n' && c != U'\r' && c != 0x3000) return false;

  size_t end = pos + 1;
  while (end < text.size()) {
    char32_t nc = text[end];
    if (nc != U' ' && nc != U'\t' && nc != U'\n' && nc != U'\r' && nc != 0x3000) break;
    ++end;
  }
  out = text.substr(pos, end - pos);
  next_pos = end;
  return true;
}

// ---------------------------------------------------------------------------
// get_units()
// ---------------------------------------------------------------------------

std::vector<SourceUnit> get_units(const std::u32string& text) {
  // --- ステップ1: ブロック分割 ---
  struct RawUnit {
    std::u32string text;
    int orig_idx;
  };
  std::vector<RawUnit> raw_units;

  size_t pos = 0;
  while (pos < text.size()) {
    std::u32string out;
    size_t next_pos = pos;

    if (try_match_bracket(text, pos, out, next_pos)) {
      raw_units.push_back({out, static_cast<int>(pos)});
    } else if (try_match_kana_block(text, pos, out, next_pos)) {
      raw_units.push_back({out, static_cast<int>(pos)});
    } else if (try_match_ascii_block(text, pos, out, next_pos)) {
      raw_units.push_back({out, static_cast<int>(pos)});
    } else if (try_match_space(text, pos, out, next_pos)) {
      raw_units.push_back({out, static_cast<int>(pos)});
    } else {
      // 上記どれにも当たらない → 1文字
      raw_units.push_back({text.substr(pos, 1), static_cast<int>(pos)});
      next_pos = pos + 1;
    }

    if (next_pos <= pos) break;  // 無限ループ防止（通常は起こらない）
    pos = next_pos;
  }

  // --- ステップ2: 位取り文字の昇格判定 ---
  size_t n = raw_units.size();
  std::vector<bool> is_numeric(n, false);

  for (size_t i = 0; i < n; ++i) {
    const auto& val = raw_units[i].text;
    if (val.size() == 1 && get_char_type(val[0]) == CharType::JAPANESE_NUMERIC) {
      is_numeric[i] = true;
    }
  }

  // 左→右パス: 左隣が numeric なら位取り文字を昇格
  for (size_t i = 1; i < n; ++i) {
    const auto& val = raw_units[i].text;
    if (val.size() == 1 && is_kurai_char(val[0]) && is_numeric[i - 1]) {
      is_numeric[i] = true;
    }
  }

  // 右→左パス: 右隣が numeric なら位取り文字を昇格（「十一」の「十」など）
  for (size_t i = n - 1; i-- > 0;) {
    const auto& val = raw_units[i].text;
    if (val.size() == 1 && is_kurai_char(val[0]) && is_numeric[i + 1]) {
      is_numeric[i] = true;
    }
  }

  // --- ステップ3: ctype を確定して返す ---
  std::vector<SourceUnit> units;
  units.reserve(n);
  for (size_t i = 0; i < n; ++i) {
    CharType ctype;
    if (is_numeric[i]) {
      ctype = CharType::JAPANESE_NUMERIC;
    } else if (!raw_units[i].text.empty()) {
      ctype = get_char_type(raw_units[i].text[0]);
    } else {
      ctype = CharType::OTHER;
    }
    units.push_back({raw_units[i].text, raw_units[i].orig_idx, ctype});
  }

  return units;
}

// ---------------------------------------------------------------------------
// compute_source_features()
// ---------------------------------------------------------------------------

/// char32_t 1文字を UTF-8 文字列に変換してキーに組み込む用のヘルパー
static std::string unit_to_str(const std::u32string& u) {
  std::string result;
  for (char32_t cp : u) result += char32_to_utf8(cp);
  return result;
}

std::vector<FeatureDict> compute_source_features(const std::vector<SourceUnit>& source_seq) {
  std::vector<FeatureDict> result;
  size_t n = source_seq.size();
  result.reserve(n);

  for (size_t i = 0; i < n; ++i) {
    const std::string char_str = unit_to_str(source_seq[i].text);
    const std::string ctype_str = char_type_to_str(source_seq[i].ctype);

    const std::string prev2_char = i > 1 ? unit_to_str(source_seq[i - 2].text) : "";
    const std::string prev2_ctype = i > 1 ? char_type_to_str(source_seq[i - 2].ctype) : "";
    const std::string prev_char = i > 0 ? unit_to_str(source_seq[i - 1].text) : "";
    const std::string prev_ctype = i > 0 ? char_type_to_str(source_seq[i - 1].ctype) : "";
    const std::string next_char = i < n - 1 ? unit_to_str(source_seq[i + 1].text) : "";
    const std::string next_ctype = i < n - 1 ? char_type_to_str(source_seq[i + 1].ctype) : "";
    const std::string next2_char = i < n - 2 ? unit_to_str(source_seq[i + 2].text) : "";
    const std::string next2_ctype = i < n - 2 ? char_type_to_str(source_seq[i + 2].ctype) : "";

    FeatureDict features;
    features["bias"] = 1.0f;
    features["char=" + char_str] = 1.0f;
    features["type=" + ctype_str] = 1.0f;

    if (i > 0) {
      features["-1:char=" + prev_char] = 1.0f;
      features["-1:type=" + prev_ctype] = 1.0f;
      features["-1:bi=" + prev_char + char_str] = 1.0f;
      if (i > 1) {
        features["-2:char=" + prev2_char] = 1.0f;
        features["-2:type=" + prev2_ctype] = 1.0f;
        features["-2:-1:bi=" + prev2_char + prev_char] = 1.0f;
      }
    } else {
      features["BOS"] = 1.0f;
    }

    if (i < n - 1) {
      features["+1:char=" + next_char] = 1.0f;
      features["+1:type=" + next_ctype] = 1.0f;
      features["+1:bi=" + char_str + next_char] = 1.0f;
      if (i < n - 2) {
        features["+2:char=" + next2_char] = 1.0f;
        features["+2:type=" + next2_ctype] = 1.0f;
        features["+1:+2:bi=" + next_char + next2_char] = 1.0f;
      }
    } else {
      features["EOS"] = 1.0f;
    }

    if (i > 0) {
      features["type_transition=" + prev_ctype + "->" + ctype_str] = 1.0f;
    }

    if (i > 0 && i < n - 1) {
      features["tri=" + prev_char + char_str + next_char] = 1.0f;
      features["type_tri=" + prev_ctype + "-" + ctype_str + "-" + next_ctype] = 1.0f;
    }

    // KANJI 連続長
    if (source_seq[i].ctype == CharType::KANJI) {
      int run = 1;
      for (size_t j = i + 1; j < n && source_seq[j].ctype == CharType::KANJI; ++j) ++run;
      for (size_t j = i - 1; j < n && source_seq[j].ctype == CharType::KANJI; --j) ++run;

      std::string run_key = (run <= 4) ? std::to_string(run) : "5+";
      features["kanji_run_len=" + run_key] = 1.0f;

      if (i == 0 || source_seq[i - 1].ctype != CharType::KANJI) {
        features["kanji_pos_first"] = 1.0f;
      }

      // 左隣の JAPANESE_NUMERIC 連続長
      if (i > 0 && source_seq[i - 1].ctype == CharType::JAPANESE_NUMERIC) {
        int num_run = 0;
        for (size_t j = i - 1; j < n && source_seq[j].ctype == CharType::JAPANESE_NUMERIC; --j) ++num_run;
        std::string num_run_key = (num_run <= 4) ? std::to_string(num_run) : "5+";
        features["-1:japanese_numeric_run_len=" + num_run_key] = 1.0f;
      }
    }

    // JAPANESE_NUMERIC 連続長
    if (source_seq[i].ctype == CharType::JAPANESE_NUMERIC) {
      int run = 1;
      for (size_t j = i + 1; j < n && source_seq[j].ctype == CharType::JAPANESE_NUMERIC; ++j) ++run;
      for (size_t j = i - 1; j < n && source_seq[j].ctype == CharType::JAPANESE_NUMERIC; --j) ++run;

      std::string run_key = (run <= 4) ? std::to_string(run) : "5+";
      features["japanese_numeric_run_len=" + run_key] = 1.0f;
    }

    result.push_back(std::move(features));
  }

  return result;
}

}  // namespace momo
