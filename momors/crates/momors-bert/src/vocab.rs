//! BERT の文字語彙（`vocab.txt`）。1 文字 = 1 トークン。
//!
//! 規則は `experimentals/bert/src/bert/encoder_core.py` の `CharVocab.token_id` と同じ:
//! 語彙にあればその id、無ければ NFKC で 1 文字になり語彙にあればその id、
//! どちらでもなければ `[UNK]`。

use std::collections::HashMap;

use unicode_normalization::UnicodeNormalization;

use crate::BertError;

#[derive(Debug)]
pub struct CharVocab {
    index: HashMap<char, i64>,
    pub pad: i64,
    pub unk: i64,
    pub cls: i64,
    pub sep: i64,
}

impl CharVocab {
    /// `vocab.txt`（1 行 1 トークン、行番号が id）の中身から作る。
    pub fn parse(text: &str) -> Result<Self, BertError> {
        let mut index = HashMap::new();
        let mut special = HashMap::new();
        for (id, token) in text.lines().enumerate() {
            let mut chars = token.chars();
            match (chars.next(), chars.next()) {
                (Some(c), None) => {
                    index.insert(c, id as i64);
                }
                _ => {
                    special.insert(token, id as i64);
                }
            }
        }
        let get = |name: &str| {
            special
                .get(name)
                .copied()
                .ok_or_else(|| BertError::Format(format!("vocab.txt に {name} がありません")))
        };
        Ok(Self {
            pad: get("[PAD]")?,
            unk: get("[UNK]")?,
            cls: get("[CLS]")?,
            sep: get("[SEP]")?,
            index,
        })
    }

    pub fn token_id(&self, ch: char) -> i64 {
        if let Some(&id) = self.index.get(&ch) {
            return id;
        }
        let mut buf = [0u8; 4];
        let mut n = ch.encode_utf8(&mut buf).nfkc();
        match (n.next(), n.next()) {
            (Some(c), None) => self.index.get(&c).copied().unwrap_or(self.unk),
            _ => self.unk,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn vocab() -> CharVocab {
        CharVocab::parse("[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\n漢\n字\nA\n").unwrap()
    }

    #[test]
    fn special_tokens_and_chars() {
        let v = vocab();
        assert_eq!((v.pad, v.unk, v.cls, v.sep), (0, 1, 2, 3));
        assert_eq!(v.token_id('漢'), 5);
        assert_eq!(v.token_id('字'), 6);
    }

    #[test]
    fn nfkc_fallback_then_unk() {
        let v = vocab();
        // 全角 Ａ は語彙に無いが、NFKC で A になる。
        assert_eq!(v.token_id('Ａ'), 7);
        assert_eq!(v.token_id('か'), v.unk);
    }
}
