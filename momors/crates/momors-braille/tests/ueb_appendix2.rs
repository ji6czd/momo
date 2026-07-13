//! UEB Appendix 2「Word List」を回帰コーパスとして流す。
//!
//! 出典: The Rules of Unified English Braille, Third Edition 2024 (ICEB)。Section 10（縮約）の
//! 用例語 1,028件と、その点字・適用規則。データは `dataset/ueb_appendix2_words.tsv`。
//!
//! 規範の用例をそのまま突き合わせるので、実装の穴がそのまま見える。ここを通すために
//! 追加したデータ（`[divisions]` の区切り、`initial_stems`）と機構（§10.4.2 / §10.5 / §10.7.4）は
//! `docs/ueb-grade2-corpus.md` に書いた。
//!
//! **KNOWN_FAIL** は「まだ規範に届いていない語」。増えたら退行、減ったらリストを更新すること。

use momors_braille::EnglishTranslator;

const CORPUS: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../../dataset/ueb_appendix2_words.tsv"
));

/// まだ規範どおりに書けない語。理由ごとに分けてある。
const KNOWN_FAIL: &[&str] = &[
    // typeform 符（イタリック ⠨⠂ / 太字 ⠘⠂）は未対応
    "Quickbraille",
    "SuperBraille",
    "GreatGrandChild",
    // §10.9.6 技術用語・略語では縮約しない（語リストが要る）
    "mst file",
    "Somesch River",
    // §10.1.3 頭字語（1文字ずつ読む語）では wordsign を使わない。発音の知識に依るので
    // 綴りからは決まらない（規範も "preferably" と書き、`CAN Network` は縮約する）
    "US (United States)",
    "IT (Information Technology)",
    // 語ごとの発音・語構成に依る例（データを足せば直るが、まだ入れていない）
    "ABOUTface!",
    "Al",
    "BEd",
    "Ch'ing",
    "cofounder",
    "Conn.",
    "hoity-toity",
    "mod cons",
    "where'er",
];

/// 点字スペース（U+2800）と ASCII スペースの差を吸収する。
fn norm(s: &str) -> String {
    s.replace('\u{2800}', " ").trim().to_owned()
}

#[test]
fn appendix2_word_list() {
    let translator = EnglishTranslator::from_embedded().expect("UEB grade2 テーブル");

    let mut failed: Vec<&str> = Vec::new();
    let mut total = 0usize;
    for line in CORPUS.lines() {
        if line.starts_with('#') || line.trim().is_empty() {
            continue;
        }
        let mut cols = line.split('\t');
        let (Some(text), Some(want)) = (cols.next(), cols.next()) else {
            continue;
        };
        total += 1;
        let got = translator.translate(text).braille_text().to_owned();
        if norm(&got) != norm(want) {
            failed.push(text);
        }
    }
    assert!(total > 1000, "コーパスが読めていない（{total} 件）");

    let unexpected: Vec<&&str> = failed.iter().filter(|t| !KNOWN_FAIL.contains(t)).collect();
    assert!(
        unexpected.is_empty(),
        "規範どおりに書けなくなった語がある（退行）: {unexpected:?}"
    );

    let fixed: Vec<&&str> = KNOWN_FAIL.iter().filter(|t| !failed.contains(t)).collect();
    assert!(
        fixed.is_empty(),
        "KNOWN_FAIL の語が通るようになった。リストから消すこと: {fixed:?}"
    );
}
