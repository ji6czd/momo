//! ESP32-P4 向け momors シリアルコンソール REPL。
//!
//! - 起動時に `model` パーティション（`partitions.csv`）を `esp_partition_mmap` でデータ領域に
//!   マップし、その `&[u8]` から `Predictor::from_model_bytes` でモデルを展開する
//!   （フラッシュ → PSRAM への丸ごとコピーは発生しない。展開後の Vec は PSRAM）。
//! - モデルが無い/壊れている場合は点字変換のみ（入力をかなとみなす）で動く。
//! - 1 行入力 → `kana:` / `seg:` / `braille:` / `time:` を返す。`:` 始まりはコマンド。

use std::io::{BufRead, Write};
use std::time::Instant;

use esp_idf_sys as sys;
use momors_braille::{BrailleTranslator, JapaneseTranslator};
use momors_core::Predictor;

/// `partitions.csv` の model パーティション（type=0x40, subtype=0x00）。
const MODEL_PARTITION_TYPE: sys::esp_partition_type_t = 0x40;
const MODEL_PARTITION_SUBTYPE: sys::esp_partition_subtype_t = 0x00;
const MODEL_PARTITION_LABEL: &core::ffi::CStr = c"model";

/// ヒープ残量を "internal 496.4KB / psram 32.00MB" 形式で返す。
fn heap_summary() -> String {
    // SAFETY: 引数なしの読み取り専用 API。
    let internal = unsafe { sys::heap_caps_get_free_size(sys::MALLOC_CAP_INTERNAL) };
    let spiram = unsafe { sys::heap_caps_get_free_size(sys::MALLOC_CAP_SPIRAM) };
    format!(
        "internal {:.1}KB / psram {:.2}MB",
        internal as f64 / 1024.0,
        spiram as f64 / (1024.0 * 1024.0)
    )
}

/// 標準入力をブロッキング読みできるようにコンソールドライバを VFS に接続する。
///
/// ESP-IDF の既定はポーリング（非ブロッキング）で、`read_line` が即座に 0 を返す。
/// 関数名は esp-idf-sys 0.37 の bindings.h が v5 系で取り込む旧名ヘッダ
/// (`esp_vfs_usb_serial_jtag.h` / `esp_vfs_dev.h`) のもの。
fn install_console() {
    #[cfg(esp_idf_esp_console_usb_serial_jtag)]
    unsafe {
        let mut cfg = sys::usb_serial_jtag_driver_config_t {
            tx_buffer_size: 1024,
            rx_buffer_size: 1024,
        };
        sys::esp!(sys::usb_serial_jtag_driver_install(&mut cfg))
            .expect("usb_serial_jtag_driver_install");
        sys::esp_vfs_usb_serial_jtag_use_driver();
        sys::esp_vfs_dev_usb_serial_jtag_set_rx_line_endings(
            sys::esp_line_endings_t_ESP_LINE_ENDINGS_CR,
        );
        sys::esp_vfs_dev_usb_serial_jtag_set_tx_line_endings(
            sys::esp_line_endings_t_ESP_LINE_ENDINGS_CRLF,
        );
    }
    #[cfg(not(esp_idf_esp_console_usb_serial_jtag))]
    unsafe {
        sys::esp!(sys::uart_driver_install(
            0,
            1024,
            0,
            0,
            std::ptr::null_mut(),
            0
        ))
        .expect("uart_driver_install");
        sys::esp_vfs_dev_uart_use_driver(0);
        sys::esp_vfs_dev_uart_port_set_rx_line_endings(
            0,
            sys::esp_line_endings_t_ESP_LINE_ENDINGS_CR,
        );
        sys::esp_vfs_dev_uart_port_set_tx_line_endings(
            0,
            sys::esp_line_endings_t_ESP_LINE_ENDINGS_CRLF,
        );
    }
}

/// model パーティションをデータ領域にマップして `&'static [u8]` を返す。
///
/// マッピングはアプリの生存期間中ずっと保持する（ハンドルは解放しない）。
/// パーティションが無い、または先頭が `.mbm` のマジックでない（未書き込み = 0xFF）場合は `Err`。
fn map_model_partition() -> Result<&'static [u8], String> {
    // SAFETY: 引数はいずれも定数。戻り値はアプリ生存期間中有効な静的テーブルへのポインタ。
    let part = unsafe {
        sys::esp_partition_find_first(
            MODEL_PARTITION_TYPE,
            MODEL_PARTITION_SUBTYPE,
            MODEL_PARTITION_LABEL.as_ptr(),
        )
    };
    if part.is_null() {
        return Err("model パーティションが見つかりません".into());
    }
    // SAFETY: 上で非 NULL を確認済み。
    let (address, size) = unsafe { ((*part).address, (*part).size as usize) };
    let mut ptr: *const core::ffi::c_void = std::ptr::null();
    let mut handle: sys::esp_partition_mmap_handle_t = 0;
    // SAFETY: part は有効、出力ポインタは有効なローカル。
    let err = unsafe {
        sys::esp_partition_mmap(
            part,
            0,
            size,
            sys::esp_partition_mmap_memory_t_ESP_PARTITION_MMAP_DATA,
            &mut ptr,
            &mut handle,
        )
    };
    sys::esp!(err).map_err(|e| format!("esp_partition_mmap 失敗: {e}"))?;
    // SAFETY: mmap 成功時、ptr は size バイトの読み取り専用領域を指し、
    // handle を解放しない限りアプリ生存期間中有効。
    let bytes: &'static [u8] = unsafe { std::slice::from_raw_parts(ptr as *const u8, size) };
    if !bytes.starts_with(b"MOMO") {
        return Err(format!(
            "model パーティション (0x{address:x}, {}MB) にモデルが書き込まれていません（先頭 {:02x?}）",
            size / (1024 * 1024),
            &bytes[..4]
        ));
    }
    Ok(bytes)
}

/// `:bench` / `:probe` の既定テキスト（3 文・46 文字）。
const BENCH_TEXT: &str =
    "吾輩は猫である。今日は良い天気ですね。東京都渋谷区で3人の学生が本を読んだ。";

// ---- `:xip` ゼロコピー(借用ロード) の GO/NO-GO 判断用ベンチマーク ----
//
// 語彙表と CSC 配列を PSRAM に展開せず mmap 領域から直接読む案（`docs/zerocopy-model-plan.md`
// の「借用ロード」、`docs/esp32p4-plan.md` §6.1 ②）は、常駐 14.5MB とロード 2 秒を消す代わりに
// 読み出しをフラッシュ(XIP)に落とす。損得は「フラッシュ読みが PSRAM より何倍遅いか」だけで
// 決まるので、loader を書き換える前にそこだけを切り出して測る。
//
// 同じ内容のバッファをフラッシュ(model パーティションの mmap 領域)と PSRAM に用意し、
// 同じコード・同じ乱数列で 2 通りのアクセスパターンを回す。データの中身は関係ない
// （測っているのはアドレスを触るコストであって比較の意味ではない）ので、モデルの
// バイト列をそのまま固定幅レコードの配列とみなす。
//
// 判断の目安: 比が 〜2 倍なら語彙・CSC とも借用へ。2〜4 倍なら CSC だけ借用。
// それ以上ならロード時間だけのために払うには高い。

/// 語彙エントリ相当の固定幅レコード長（`FeatureKey` + feature_id）。
const XIP_REC: usize = 16;
/// CSC 1 列相当の連続読み長（nnz 約 32 × (u16 rowind + i8 data)）。
const XIP_COL: usize = 96;
/// 比較に使うバッファ長。L2 に収まらない大きさにして、両者ともミス主体で比べる。
const XIP_BUF: usize = 4 * 1024 * 1024;

fn xorshift(state: &mut u64) -> u64 {
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    x
}

/// 二分探索と同じ探索木の形でバッファを触る。戻り値は (累積値, 総プローブ数)。
///
/// 実際にキー比較はせず、乱数で決めた目標位置へ区間を狭めていく。触るアドレスの
/// 系列が本物の bsearch と同じであれば、メモリコストの比較としては十分。
#[inline(never)]
fn xip_probe_bsearch(buf: &[u8], n_rec: usize, iters: usize, seed: &mut u64) -> (u64, u64) {
    let mut acc = 0u64;
    let mut probes = 0u64;
    for _ in 0..iters {
        let target = (xorshift(seed) % n_rec as u64) as usize;
        let (mut lo, mut hi) = (0usize, n_rec);
        while lo < hi {
            let mid = lo + (hi - lo) / 2;
            acc = acc.wrapping_add(buf[mid * XIP_REC] as u64);
            probes += 1;
            if mid < target {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
    }
    (acc, probes)
}

/// ランダムな位置から 1 列分を連続読みする（CSC の列アクセス相当）。
#[inline(never)]
fn xip_probe_seq(buf: &[u8], iters: usize, seed: &mut u64) -> u64 {
    let mut acc = 0u64;
    let span = (buf.len() - XIP_COL) as u64;
    for _ in 0..iters {
        let off = (xorshift(seed) % span) as usize;
        for b in &buf[off..off + XIP_COL] {
            acc = acc.wrapping_add(*b as u64);
        }
    }
    acc
}

/// フラッシュ(XIP) と PSRAM で同じアクセスを回し、1 プローブ / 1 列あたりの時間と比を出す。
fn bench_xip(out: &mut impl Write, flash: &'static [u8], iters: usize) {
    if flash.len() < XIP_BUF {
        writeln!(out, "xip: model パーティションが小さすぎます").ok();
        return;
    }
    let flash_buf = &flash[..XIP_BUF];

    // SAFETY: PSRAM から XIP_BUF バイト確保し、フラッシュ側と同じ内容で埋める。
    let ptr = unsafe { sys::heap_caps_malloc(XIP_BUF, sys::MALLOC_CAP_SPIRAM) } as *mut u8;
    if ptr.is_null() {
        writeln!(out, "xip: PSRAM {}MB の確保に失敗", XIP_BUF / (1024 * 1024)).ok();
        return;
    }
    // SAFETY: ptr は XIP_BUF バイトの有効領域、flash_buf も同じ長さで重ならない。
    let psram_buf: &[u8] = unsafe {
        std::ptr::copy_nonoverlapping(flash_buf.as_ptr(), ptr, XIP_BUF);
        std::slice::from_raw_parts(ptr, XIP_BUF)
    };

    let n_rec = XIP_BUF / XIP_REC;
    writeln!(
        out,
        "xip: buf {}MB ({} recs x {}B), iters {}   flash {:p} / psram {:p}",
        XIP_BUF / (1024 * 1024),
        n_rec,
        XIP_REC,
        iters,
        flash_buf.as_ptr(),
        psram_buf.as_ptr()
    )
    .ok();

    // 両者とも同じ種から始めることで、触る位置の系列を完全に一致させる。
    const SEED: u64 = 0x9E37_79B9_7F4A_7C15;
    let mut results = [(0f64, 0f64); 2]; // [bsearch(ns/probe), seq(ns/col)] x [flash, psram]

    for (i, buf) in [flash_buf, psram_buf].into_iter().enumerate() {
        // ウォームアップ（初回のページイン/TLB を測定から外す）
        let mut warm = SEED;
        std::hint::black_box(xip_probe_bsearch(buf, n_rec, 64, &mut warm));

        let mut seed = SEED;
        let t = Instant::now();
        let (acc, probes) = xip_probe_bsearch(buf, n_rec, iters, &mut seed);
        let ns = t.elapsed().as_nanos();
        std::hint::black_box(acc);
        results[i].0 = ns as f64 / probes as f64;

        let mut seed = SEED;
        let t = Instant::now();
        let acc = xip_probe_seq(buf, iters, &mut seed);
        let ns = t.elapsed().as_nanos();
        std::hint::black_box(acc);
        results[i].1 = ns as f64 / iters as f64;
    }

    // SAFETY: 上で heap_caps_malloc した領域。以降 psram_buf は使わない。
    unsafe { sys::heap_caps_free(ptr as *mut core::ffi::c_void) };

    writeln!(out, "pattern	flash	psram	ratio").ok();
    writeln!(
        out,
        "bsearch probe	{:.0} ns	{:.0} ns	{:.2}x",
        results[0].0,
        results[1].0,
        results[0].0 / results[1].0
    )
    .ok();
    writeln!(
        out,
        "csc column	{:.0} ns	{:.0} ns	{:.2}x",
        results[0].1,
        results[1].1,
        results[0].1 / results[1].1
    )
    .ok();
    writeln!(
        out,
        "judge: 〜2.0x なら語彙・CSC とも借用ロードへ / 2〜4x は CSC のみ / それ以上は見送り"
    )
    .ok();
}

#[derive(Clone, Copy, PartialEq)]
enum Mode {
    Kana,
    Braille,
    All,
}

struct Engine {
    predictor: Option<Predictor>,
    translator: BrailleTranslator,
    mode: Mode,
    /// mmap した model パーティション（`:xip` のフラッシュ側バッファに使う）。
    model_bytes: Option<&'static [u8]>,
}

impl Engine {
    fn handle_text(&self, out: &mut impl Write, text: &str) {
        let start = Instant::now();
        let (kana, seg) = match &self.predictor {
            Some(p) => match p.predict(text) {
                Ok(r) => (r.kana_text().to_owned(), Some(r.format_source_segmented())),
                Err(e) => {
                    writeln!(out, "error: {e}").ok();
                    return;
                }
            },
            // モデルなし: 入力をかなとみなす
            None => (text.to_owned(), None),
        };
        let predict_ms = start.elapsed().as_millis();

        if self.mode != Mode::Braille {
            writeln!(out, "kana:    {kana}").ok();
            if let Some(seg) = &seg {
                writeln!(out, "seg:     {seg}").ok();
            }
        }
        if self.mode != Mode::Kana {
            match self.translator.translate(&kana) {
                Ok(r) => writeln!(out, "braille: {}", r.braille_text()).ok(),
                Err(e) => writeln!(out, "braille error: {e}").ok(),
            };
        }
        let total_ms = start.elapsed().as_millis();
        writeln!(
            out,
            "time:    predict {predict_ms} ms / total {total_ms} ms   heap: {}",
            heap_summary()
        )
        .ok();
    }

    /// 固定文を `n` 回 `predict` して、1 文字あたりの平均所要時間を出す。
    fn bench(&self, out: &mut impl Write, n: usize) {
        let Some(p) = &self.predictor else {
            writeln!(out, "model: none").ok();
            return;
        };
        let chars = BENCH_TEXT.chars().count();
        // ウォームアップ 1 回
        std::hint::black_box(p.predict(BENCH_TEXT).ok());
        let t = Instant::now();
        for _ in 0..n {
            std::hint::black_box(p.predict(BENCH_TEXT).ok());
        }
        let total_us = t.elapsed().as_micros();
        writeln!(
            out,
            "bench: {chars} chars x {n} = {:.1} ms/run, {:.0} us/char",
            total_us as f64 / 1000.0 / n as f64,
            total_us as f64 / (n * chars) as f64
        )
        .ok();
    }

    /// 1 文字ごとの 語彙引き/読み/境界 の内訳を出し、合計と平均をまとめる。
    fn probe(&self, out: &mut impl Write, text: &str) {
        let Some(p) = &self.predictor else {
            writeln!(out, "model: none").ok();
            return;
        };
        let (featurize_ns, rows) = p.char_latency_with_featurize(text);
        writeln!(
            out,
            "featurize: {:.0} us total, {:.0} us/char",
            featurize_ns as f64 / 1000.0,
            featurize_ns as f64 / 1000.0 / rows.len().max(1) as f64
        )
        .ok();
        let st = p.csc_stats();
        writeln!(
            out,
            "csc: cols {}, nnz total {}, max/col {}, dense cols {} (nnz {} = {:.1}%)",
            st.n_cols,
            st.total_nnz,
            st.max_nnz,
            st.dense_cols,
            st.dense_nnz,
            st.dense_nnz as f64 * 100.0 / st.total_nnz.max(1) as f64
        )
        .ok();
        writeln!(out, "char	keys	hits	nnz	lookup_ns	read_ns	boundary_ns").ok();
        let (mut lk, mut rd, mut bd, mut keys, mut hits, mut nnz) =
            (0u128, 0u128, 0u128, 0usize, 0usize, 0usize);
        for r in &rows {
            writeln!(
                out,
                "{}	{}	{}	{}	{}	{}	{}",
                r.text, r.n_keys, r.n_hits, r.nnz, r.lookup_ns, r.read_ns, r.boundary_ns
            )
            .ok();
            lk += r.lookup_ns;
            rd += r.read_ns;
            bd += r.boundary_ns;
            keys += r.n_keys;
            hits += r.n_hits;
            nnz += r.nnz;
        }
        let n = rows.len().max(1) as f64;
        writeln!(
            out,
            "avg/char: keys {:.1} hits {:.1} nnz {:.0} | lookup {:.0} us, read {:.0} us, boundary {:.0} us, sum {:.0} us",
            keys as f64 / n,
            hits as f64 / n,
            nnz as f64 / n,
            lk as f64 / n / 1000.0,
            rd as f64 / n / 1000.0,
            bd as f64 / n / 1000.0,
            (lk + rd + bd) as f64 / n / 1000.0
        )
        .ok();
    }

    fn handle_command(&mut self, out: &mut impl Write, cmd: &str) {
        match cmd.split_whitespace().collect::<Vec<_>>().as_slice() {
            [":stat"] => {
                writeln!(out, "heap: {}", heap_summary()).ok();
                let model = match &self.predictor {
                    Some(p) => format!(
                        "loaded (features {}, classes {})",
                        p.n_features(),
                        p.n_classes()
                    ),
                    None => "none (braille-only)".to_owned(),
                };
                writeln!(out, "model: {model}").ok();
            }
            [":mode", m] => {
                self.mode = match *m {
                    "kana" => Mode::Kana,
                    "braille" => Mode::Braille,
                    "all" => Mode::All,
                    _ => {
                        writeln!(out, "usage: :mode kana|braille|all").ok();
                        return;
                    }
                };
                writeln!(out, "mode: {m}").ok();
            }
            [":probe", rest @ ..] => {
                let text = if rest.is_empty() {
                    BENCH_TEXT
                } else {
                    cmd[":probe".len()..].trim()
                };
                self.probe(out, text);
            }
            [":bench", rest @ ..] => {
                let n: usize = rest.first().and_then(|s| s.parse().ok()).unwrap_or(5);
                self.bench(out, n);
            }
            [":phases", rest @ ..] => {
                let n: usize = rest.first().and_then(|s| s.parse().ok()).unwrap_or(5);
                match &self.predictor {
                    Some(p) => {
                        let r = p.predict_phases(BENCH_TEXT, n);
                        writeln!(out, "phases (us/char, {} chars): {}", r.chars, r.summary()).ok();
                    }
                    None => {
                        writeln!(out, "model: none").ok();
                    }
                }
            }
            [":xip", rest @ ..] => {
                let n: usize = rest.first().and_then(|s| s.parse().ok()).unwrap_or(20_000);
                match self.model_bytes {
                    Some(b) => bench_xip(out, b, n),
                    None => {
                        writeln!(out, "xip: model パーティションが未マップです").ok();
                    }
                }
            }
            [":help"] => {
                writeln!(out, ":stat                   ヒープ/モデル情報").ok();
                writeln!(out, ":mode kana|braille|all  出力の切替").ok();
                writeln!(
                    out,
                    ":bench [N]              固定文を N 回推論して 1 文字あたりの時間を出す"
                )
                .ok();
                writeln!(
                    out,
                    ":probe [text]           1 文字ごとの 語彙引き/読み/境界 の内訳 (ns)"
                )
                .ok();
                writeln!(
                    out,
                    ":phases [N]             predict の段階別内訳 (us/char)"
                )
                .ok();
                writeln!(
                    out,
                    ":xip [N]                フラッシュ(XIP) vs PSRAM の読み出しコスト比"
                )
                .ok();
                writeln!(out, "それ以外の行は日本語テキストとして変換します").ok();
            }
            _ => {
                writeln!(out, "unknown command: {cmd} (:help)").ok();
            }
        }
    }
}

fn main() {
    // esp-idf-sys のランタイムパッチをリンクさせるため最初に一度呼ぶ。
    sys::link_patches();
    install_console();

    let stdout = std::io::stdout();
    let mut out = stdout.lock();
    writeln!(out, "momors-esp32 {}", env!("CARGO_PKG_VERSION")).unwrap();
    writeln!(out, "heap at boot: {}", heap_summary()).unwrap();
    out.flush().unwrap();

    // ---- 点字テーブル（埋め込み TOML を起動時に解析）----
    let t = Instant::now();
    let japanese =
        JapaneseTranslator::from_embedded().expect("埋め込み点字テーブルの読み込みに失敗");
    let translator = BrailleTranslator::new(japanese, None);
    writeln!(
        out,
        "braille table: ready in {} ms   heap: {}",
        t.elapsed().as_millis(),
        heap_summary()
    )
    .unwrap();
    out.flush().unwrap();

    // ---- モデル ----
    let mut model_bytes: Option<&'static [u8]> = None;
    let predictor = match map_model_partition() {
        Ok(bytes) => {
            model_bytes = Some(bytes);
            writeln!(
                out,
                "model partition: mapped {:.1}MB at {:p}",
                bytes.len() as f64 / (1024.0 * 1024.0),
                bytes.as_ptr()
            )
            .unwrap();
            out.flush().unwrap();
            let t = Instant::now();
            match Predictor::from_model_bytes(bytes) {
                Ok(p) => {
                    writeln!(
                        out,
                        "model: loaded in {} ms (features {}, classes {})   heap: {}",
                        t.elapsed().as_millis(),
                        p.n_features(),
                        p.n_classes(),
                        heap_summary()
                    )
                    .unwrap();
                    Some(p)
                }
                Err(e) => {
                    writeln!(out, "model: load failed: {e}").unwrap();
                    None
                }
            }
        }
        Err(e) => {
            writeln!(out, "model: {e}").unwrap();
            None
        }
    };
    if predictor.is_none() {
        writeln!(
            out,
            "モデルなしで起動: 入力をかなとみなして点字に変換します"
        )
        .unwrap();
    }
    writeln!(out, "ready. :help でコマンド一覧").unwrap();
    out.flush().unwrap();

    let mut engine = Engine {
        predictor,
        translator,
        mode: Mode::All,
        model_bytes,
    };

    let stdin = std::io::stdin();
    let mut line = String::new();
    loop {
        write!(out, "> ").unwrap();
        out.flush().unwrap();
        line.clear();
        match stdin.lock().read_line(&mut line) {
            Ok(0) => continue,
            Ok(_) => {}
            Err(e) => {
                writeln!(out, "read error: {e}").unwrap();
                continue;
            }
        }
        let text = line.trim_end_matches(['\r', '\n']).trim();
        if text.is_empty() {
            continue;
        }
        if text.starts_with(':') {
            engine.handle_command(&mut out, text);
        } else {
            engine.handle_text(&mut out, text);
        }
        out.flush().unwrap();
    }
}
