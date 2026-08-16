//! 使い捨て検証用ベンチマーク（本番コードには一切組み込まない）。
//!
//! mmap 化した場合に推論ホットパス（語彙 bsearch・CSC スコア集計）が
//! PC 上でどれだけ遅くなるかを、実モデルのデータサイズで測る。
//! `docs/zerocopy-model-plan.md` の GO/NO-GO 判断材料。
//!
//! 実データ由来の `VocabEntry` / CSC 配列をそのまま固定幅バイナリとして
//! 一時ファイルへ書き出し、mmap して読み直す。ページフォールトを済ませた
//! 定常状態（ウォームアップ後）でのアクセスコストを比較する ── PC の
//! ページキャッシュは常に温まっているため、これは「常時稼働時の推論速度」
//! を代表する。コールドスタート（初回ページフォールト）は実機でしか
//! 正しく測れないため、ここでは対象にしない。
//!
//! 実行例:
//! ```text
//! MOMO_BENCH_MODEL=../../dataset/basic_data_4.mbm cargo test --release -p momors-core \
//!   --lib mmap_experiment -- --ignored --nocapture --test-threads=1
//! ```

#![cfg(test)]

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::time::{Duration, Instant};

use memmap2::Mmap;

use crate::feature::FeatureKey;
use crate::model::{MomoModel, VocabEntry};
use crate::weight_model::WeightModel;

/// mmap 版の語彙エントリ。`VocabEntry` と同内容を固定幅で並べる
/// （設計メモの「可変長 FeatureKey を畳む」案のうち固定幅化案を模擬）。
#[repr(C)]
#[derive(Clone, Copy)]
struct VocabEntryRaw {
    key: FeatureKey,
    feature_id: u32,
    cat_column: u32,
    cat_code: u32,
}

fn model_path() -> String {
    std::env::var("MOMO_BENCH_MODEL")
        .expect("MOMO_BENCH_MODEL 環境変数に .mbm ファイルのパスを指定してください")
}

/// `entries` を固定幅バイナリのまま一時ファイルへ書き出し、mmap して返す。
/// 同一プロセス内での往復のみを想定（クロスビルドでのレイアウト安定性は保証しない）。
fn mmap_of<T: Copy>(entries: &[T], tag: &str) -> (Mmap, std::path::PathBuf) {
    let bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(
            entries.as_ptr() as *const u8,
            std::mem::size_of_val(entries),
        )
    };
    let path =
        std::env::temp_dir().join(format!("momo_mmap_bench_{tag}_{}.bin", std::process::id()));
    std::fs::write(&path, bytes).expect("一時ファイル書き込み失敗");
    let file = std::fs::File::open(&path).expect("一時ファイル open 失敗");
    let mmap = unsafe { Mmap::map(&file).expect("mmap 失敗") };
    (mmap, path)
}

fn as_slice<T>(bytes: &[u8]) -> &[T] {
    let n = bytes.len() / std::mem::size_of::<T>();
    unsafe { std::slice::from_raw_parts(bytes.as_ptr() as *const T, n) }
}

fn vocab_find_mmap(entries: &[VocabEntryRaw], key: &FeatureKey) -> Option<u32> {
    entries
        .binary_search_by(|e| e.key.cmp(key))
        .ok()
        .map(|idx| entries[idx].feature_id)
}

fn hash_index(seed: impl Hash, modulus: usize) -> usize {
    let mut hasher = DefaultHasher::new();
    seed.hash(&mut hasher);
    (hasher.finish() as usize) % modulus
}

/// 実在キー(ヒット) 9割 + 実在しないキー(ミス) 1割を混ぜたクエリ列。
/// 実推論では大半が既知の特徴量キーだが、未知キー（bsearch が最後まで
/// 潜るケース）もゼロではないため両方を混ぜる。
fn build_queries(vocab: &[VocabEntry], n: usize) -> Vec<FeatureKey> {
    (0..n)
        .map(|i| {
            if i % 10 == 0 {
                let mut k = vocab[i % vocab.len()].key;
                k.cp[0] = k.cp[0].wrapping_add(0x1000_0000); // 実在しないキーにする
                k
            } else {
                vocab[hash_index(i, vocab.len())].key
            }
        })
        .collect()
}

fn median(durations: &mut [Duration]) -> Duration {
    durations.sort();
    durations[durations.len() / 2]
}

fn report(label: &str, mut times: Vec<Duration>, n_ops: usize) -> f64 {
    let med = median(&mut times);
    let ns_per_op = med.as_nanos() as f64 / n_ops as f64;
    println!("{label}: median={med:?} / {n_ops} ops -> {ns_per_op:.1} ns/op");
    ns_per_op
}

#[test]
#[ignore]
fn mmap_bench_vocab_lookup() {
    let model = MomoModel::load(std::path::Path::new(&model_path())).expect("モデル読み込み失敗");
    let vocab = &model.vocab;
    println!("vocab entries: {}", vocab.len());

    let raw: Vec<VocabEntryRaw> = vocab
        .iter()
        .map(|e| VocabEntryRaw {
            key: e.key,
            feature_id: e.feature_id,
            cat_column: e.cat_column,
            cat_code: e.cat_code,
        })
        .collect();
    let (mmap, tmp_path) = mmap_of(&raw, "vocab");
    let mmap_entries: &[VocabEntryRaw] = as_slice(&mmap);

    let n_queries = 200_000;
    let queries = build_queries(vocab, n_queries);

    // ウォームアップ: 初回ページフォールトを済ませ、定常状態を測る。
    for q in &queries {
        std::hint::black_box(model.vocab_find(q));
        std::hint::black_box(vocab_find_mmap(mmap_entries, q));
    }

    let reps = 15;
    let mut vec_times = Vec::with_capacity(reps);
    let mut mmap_times = Vec::with_capacity(reps);
    for _ in 0..reps {
        let t0 = Instant::now();
        for q in &queries {
            std::hint::black_box(model.vocab_find(q));
        }
        vec_times.push(t0.elapsed());

        let t0 = Instant::now();
        for q in &queries {
            std::hint::black_box(vocab_find_mmap(mmap_entries, q));
        }
        mmap_times.push(t0.elapsed());
    }

    let vec_ns = report("vocab_find [Vec ]", vec_times, n_queries);
    let mmap_ns = report("vocab_find [mmap]", mmap_times, n_queries);
    println!("ratio mmap/vec = {:.2}x", mmap_ns / vec_ns);

    let _ = std::fs::remove_file(&tmp_path);
}

#[test]
#[ignore]
fn mmap_bench_csc_scores() {
    let model = MomoModel::load(std::path::Path::new(&model_path())).expect("モデル読み込み失敗");
    println!(
        "n_features={} n_classes={} nnz={}",
        model.n_features,
        model.n_classes,
        model.csc_data.len()
    );

    let (colptr_mmap, colptr_path) = mmap_of(&model.csc_colptr, "colptr");
    let (rowind_mmap, rowind_path) = mmap_of(&model.csc_rowind, "rowind");
    let (data_mmap, data_path) = mmap_of(&model.csc_data, "data");
    let colptr: &[u32] = as_slice(&colptr_mmap);
    let rowind: &[u16] = as_slice(&rowind_mmap);
    let data: &[i8] = as_slice(&data_mmap);

    // 1トークンあたりのアクティブ特徴量数を実窓幅相当 (5〜15本) でランダム生成。
    let n_calls = 50_000;
    let feat_id_sets: Vec<Vec<u32>> = (0..n_calls)
        .map(|i| {
            let k = 5 + (i % 11);
            (0..k)
                .map(|j| hash_index((i, j), model.n_features as usize) as u32)
                .collect()
        })
        .collect();

    let compute_vec = |feat_ids: &[u32], scratch: &mut [i32]| {
        scratch.fill(0);
        for &feat_id in feat_ids {
            let s = model.csc_colptr[feat_id as usize] as usize;
            let e = model.csc_colptr[feat_id as usize + 1] as usize;
            for j in s..e {
                scratch[model.csc_rowind[j] as usize] += model.csc_data[j] as i32;
            }
        }
    };
    let compute_mmap = |feat_ids: &[u32], scratch: &mut [i32]| {
        scratch.fill(0);
        for &feat_id in feat_ids {
            let s = colptr[feat_id as usize] as usize;
            let e = colptr[feat_id as usize + 1] as usize;
            for j in s..e {
                scratch[rowind[j] as usize] += data[j] as i32;
            }
        }
    };

    // 結果が一致することを確認してから計測する（ズレていたら計測に意味がない）。
    let mut scratch_vec = vec![0i32; model.n_classes as usize];
    let mut scratch_mmap = vec![0i32; model.n_classes as usize];
    for ids in &feat_id_sets {
        compute_vec(ids, &mut scratch_vec);
        compute_mmap(ids, &mut scratch_mmap);
        assert_eq!(
            scratch_vec, scratch_mmap,
            "mmap版とVec版のスコアが一致すること"
        );
    }

    let reps = 15;
    let mut vec_times = Vec::with_capacity(reps);
    let mut mmap_times = Vec::with_capacity(reps);
    for _ in 0..reps {
        let t0 = Instant::now();
        for ids in &feat_id_sets {
            compute_vec(ids, &mut scratch_vec);
            std::hint::black_box(&scratch_vec);
        }
        vec_times.push(t0.elapsed());

        let t0 = Instant::now();
        for ids in &feat_id_sets {
            compute_mmap(ids, &mut scratch_mmap);
            std::hint::black_box(&scratch_mmap);
        }
        mmap_times.push(t0.elapsed());
    }

    let vec_ns = report("csc_scores [Vec ]", vec_times, n_calls);
    let mmap_ns = report("csc_scores [mmap]", mmap_times, n_calls);
    println!("ratio mmap/vec = {:.2}x", mmap_ns / vec_ns);

    let _ = std::fs::remove_file(&colptr_path);
    let _ = std::fs::remove_file(&rowind_path);
    let _ = std::fs::remove_file(&data_path);
}

// ============================================================
// メモリ使用量の実測（実常駐サイズ = Working Set）
// ============================================================
//
// 速度計測はウォームアップ後の定常状態を見るが、mmap の本来の狙いは
// 「実際に触った分しか常駐しない」こと。それを確かめるには実テキストを
// 流し、その過程で触れた語彙/CSC のページだけが常駐に反映されるかを
// OS 側の Working Set (Windows) で見る必要がある。
//
// 3本のテストを **別プロセスとして順番に** 実行すること
// （同一プロセス内で Vec 版を読んだ後に mmap 版を試すと、解放済みの
// ヒープがアロケータに保持されたままになり Working Set が正しく比較
// できない）。
//
// ```text
// MOMO_BENCH_MODEL=<path>.mbm MOMO_BENCH_CORPUS=<path>/basic_raw.tsv \
//   cargo test --release -p momors-core --lib -- --ignored --exact \
//   mmap_experiment::mmap_bench_memory_prepare --nocapture
// cargo test --release -p momors-core --lib -- --ignored --exact \
//   mmap_experiment::mmap_bench_memory_vec --nocapture
// cargo test --release -p momors-core --lib -- --ignored --exact \
//   mmap_experiment::mmap_bench_memory_mmap --nocapture
// ```

#[cfg(windows)]
fn mem_bench_dir() -> std::path::PathBuf {
    let dir = std::env::temp_dir().join("momo_mem_bench");
    std::fs::create_dir_all(&dir).expect("一時ディレクトリ作成失敗");
    dir
}

/// `entries` を固定幅バイナリのまま `path` へ書き出す（プロセスを跨いで
/// 共有するため、[`mmap_of`] と違い一時ファイル名は固定）。
#[cfg(windows)]
fn write_raw<T: Copy>(path: &std::path::Path, entries: &[T]) {
    let bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(
            entries.as_ptr() as *const u8,
            std::mem::size_of_val(entries),
        )
    };
    std::fs::write(path, bytes).expect("一時ファイル書き込み失敗");
}

#[cfg(windows)]
fn mmap_file(path: &std::path::Path) -> Mmap {
    let file = std::fs::File::open(path).expect("一時ファイル open 失敗");
    unsafe { Mmap::map(&file).expect("mmap 失敗") }
}

/// `path` を丸ごと読み、`T` の配列として整列済みバッファへコピーする。
/// （`Vec<u8>` から直接ポインタキャストすると `T` のアライメントが
/// 保証されないため、`Vec<T>` を確保してからバイト列として読み込む。）
#[cfg(windows)]
fn read_raw<T: Copy + Default>(path: &std::path::Path) -> Vec<T> {
    use std::io::Read;
    let len = std::fs::metadata(path).expect("メタデータ取得失敗").len() as usize;
    let n = len / std::mem::size_of::<T>();
    let mut buf: Vec<T> = vec![T::default(); n];
    let bytes: &mut [u8] = unsafe {
        std::slice::from_raw_parts_mut(buf.as_mut_ptr() as *mut u8, n * std::mem::size_of::<T>())
    };
    std::fs::File::open(path)
        .expect("open 失敗")
        .read_exact(bytes)
        .expect("read 失敗");
    buf
}

/// `basic_raw.tsv`（原文 \t 読み）の先頭 `n_lines` 行から原文列だけを取り、
/// 改行で連結して1つのテキストにする。実際の学習コーパスなので、
/// 実運用に近い語彙の出現分布になる。
#[cfg(windows)]
fn load_corpus_text(tsv_path: &str, n_lines: usize) -> String {
    std::fs::read_to_string(tsv_path)
        .expect("コーパス読み込み失敗")
        .lines()
        .take(n_lines)
        .filter_map(|line| line.split('\t').next())
        .collect::<Vec<_>>()
        .join("\n")
}

#[cfg(windows)]
fn corpus_path() -> String {
    std::env::var("MOMO_BENCH_CORPUS")
        .expect("MOMO_BENCH_CORPUS 環境変数に basic_raw.tsv のパスを指定してください")
}

#[cfg(windows)]
fn rss_bytes() -> (u64, u64) {
    use windows_sys::Win32::System::ProcessStatus::{
        GetProcessMemoryInfo, PROCESS_MEMORY_COUNTERS,
    };
    use windows_sys::Win32::System::Threading::GetCurrentProcess;
    unsafe {
        let mut counters: PROCESS_MEMORY_COUNTERS = std::mem::zeroed();
        let cb = std::mem::size_of::<PROCESS_MEMORY_COUNTERS>() as u32;
        let ok = GetProcessMemoryInfo(GetCurrentProcess(), &mut counters, cb);
        assert!(ok != 0, "GetProcessMemoryInfo に失敗しました");
        (
            counters.WorkingSetSize as u64,
            counters.PeakWorkingSetSize as u64,
        )
    }
}

#[cfg(windows)]
fn mb(bytes: u64) -> f64 {
    bytes as f64 / (1024.0 * 1024.0)
}

#[cfg(windows)]
fn bench_lines() -> usize {
    std::env::var("MOMO_BENCH_LINES")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(2000)
}

/// テキストを `lines_per_chunk` 行ずつに区切り、各チャンクの候補特徴量キー列を
/// `on_chunk` に渡す。実セッションは全文のキーを同時に保持しない
/// （逐次処理して使い終わったら捨てる）ため、ここでもチャンクをまたいで
/// キー列を保持しない。人名辞書照合は行わず（辞書なし固定）、シナリオA/B
/// で条件を揃える。
#[cfg(windows)]
fn for_each_chunk_keys(
    text: &str,
    lines_per_chunk: usize,
    mut on_chunk: impl FnMut(&[FeatureKey]),
) {
    let lines: Vec<&str> = text.lines().collect();
    for chunk in lines.chunks(lines_per_chunk.max(1)) {
        let chunk_text = chunk.join("\n");
        let source_seq = crate::featurize::to_source_seq(&chunk_text);
        let name_flags = vec![crate::name_dict::NAME_FLAG_OUT; source_seq.len()];
        let keys: Vec<FeatureKey> =
            crate::featurize::compute_source_features(&source_seq, &name_flags)
                .into_iter()
                .flatten()
                .collect();
        on_chunk(&keys);
    }
}

/// 準備ステップ: 実モデルの vocab/CSC を固定幅バイナリへ書き出す
/// （`mmap_bench_memory_mmap` はここで書き出したファイルだけを見て動く）。
/// `mmap_bench_memory_vec` / `_mmap` より先に単独プロセスで実行すること。
#[cfg(windows)]
#[test]
#[ignore]
fn mmap_bench_memory_prepare() {
    let model = MomoModel::load(std::path::Path::new(&model_path())).expect("モデル読み込み失敗");

    let n_lines = bench_lines();
    let text = load_corpus_text(&corpus_path(), n_lines);
    println!("corpus: 先頭{n_lines}行 / {}文字", text.chars().count());

    let mut touched = std::collections::HashSet::new();
    let mut n_queries = 0usize;
    for_each_chunk_keys(&text, 20, |keys| {
        n_queries += keys.len();
        touched.extend(keys.iter().filter_map(|k| model.vocab_find(k)));
    });
    println!("queries (延べ): {n_queries}");
    println!(
        "distinct feature touched: {} / {} ({:.2}%)",
        touched.len(),
        model.n_features,
        100.0 * touched.len() as f64 / model.n_features as f64
    );

    let dir = mem_bench_dir();
    let raw: Vec<VocabEntryRaw> = model
        .vocab
        .iter()
        .map(|e| VocabEntryRaw {
            key: e.key,
            feature_id: e.feature_id,
            cat_column: e.cat_column,
            cat_code: e.cat_code,
        })
        .collect();
    write_raw(&dir.join("vocab.bin"), &raw);
    write_raw(&dir.join("colptr.bin"), &model.csc_colptr);
    write_raw(&dir.join("rowind.bin"), &model.csc_rowind);
    write_raw(&dir.join("data.bin"), &model.csc_data);
    write_raw(&dir.join("meta.bin"), &[model.n_classes, model.n_features]);
    println!("prepared under {}", dir.display());
}

/// シナリオA（現行本番相当）: `MomoModel::load` でVecへ全展開し、実コーパスを
/// チャンク処理した後の Working Set を測る。全展開済みなので、ロード直後と
/// ワークロード後でほぼ変化しないはず（それ自体が「常に満額常駐」の裏付け）。
#[cfg(windows)]
#[test]
#[ignore]
fn mmap_bench_memory_vec() {
    let model = MomoModel::load(std::path::Path::new(&model_path())).expect("モデル読み込み失敗");
    let (rss0, peak0) = rss_bytes();
    println!(
        "[vec] after load                : WorkingSet={:.1}MB (peak {:.1}MB)",
        mb(rss0),
        mb(peak0)
    );

    let text = load_corpus_text(&corpus_path(), bench_lines());
    let mut scratch = vec![0i64; model.n_classes as usize];
    let mut hits = 0usize;
    let mut n_queries = 0usize;
    for_each_chunk_keys(&text, 20, |keys| {
        n_queries += keys.len();
        for key in keys {
            if let Some(feat_id) = model.vocab_find(key) {
                hits += 1;
                let s = model.csc_colptr[feat_id as usize] as usize;
                let e = model.csc_colptr[feat_id as usize + 1] as usize;
                for j in s..e {
                    scratch[model.csc_rowind[j] as usize] += model.csc_data[j] as i64;
                }
            }
        }
    });
    std::hint::black_box(&scratch);

    let (rss1, peak1) = rss_bytes();
    println!(
        "[vec] after workload ({hits}/{n_queries} hits): WorkingSet={:.1}MB (peak {:.1}MB)",
        mb(rss1),
        mb(peak1)
    );
}

/// シナリオB（mmap案）: Vecへは一切展開せず、準備済みの固定幅バイナリを
/// mmap し、同じ実コーパスをチャンク処理して触れた分だけをフォールトイン
/// させた後の Working Set を測る。
#[cfg(windows)]
#[test]
#[ignore]
fn mmap_bench_memory_mmap() {
    let dir = mem_bench_dir();
    let vocab_mmap = mmap_file(&dir.join("vocab.bin"));
    let colptr_mmap = mmap_file(&dir.join("colptr.bin"));
    let rowind_mmap = mmap_file(&dir.join("rowind.bin"));
    let data_mmap = mmap_file(&dir.join("data.bin"));
    let vocab: &[VocabEntryRaw] = as_slice(&vocab_mmap);
    let colptr: &[u32] = as_slice(&colptr_mmap);
    let rowind: &[u16] = as_slice(&rowind_mmap);
    let data: &[i8] = as_slice(&data_mmap);
    let meta: Vec<u32> = read_raw(&dir.join("meta.bin"));
    let n_classes = meta[0];

    println!(
        "on-disk size: vocab={:.1}MB colptr={:.1}MB rowind={:.1}MB data={:.1}MB",
        mb(vocab_mmap.len() as u64),
        mb(colptr_mmap.len() as u64),
        mb(rowind_mmap.len() as u64),
        mb(data_mmap.len() as u64)
    );

    let (rss0, peak0) = rss_bytes();
    println!(
        "[mmap] after mmap (未タッチ)      : WorkingSet={:.1}MB (peak {:.1}MB)",
        mb(rss0),
        mb(peak0)
    );

    let text = load_corpus_text(&corpus_path(), bench_lines());
    let mut scratch = vec![0i64; n_classes as usize];
    let mut hits = 0usize;
    let mut n_queries = 0usize;
    for_each_chunk_keys(&text, 20, |keys| {
        n_queries += keys.len();
        for key in keys {
            if let Some(feat_id) = vocab_find_mmap(vocab, key) {
                hits += 1;
                let s = colptr[feat_id as usize] as usize;
                let e = colptr[feat_id as usize + 1] as usize;
                for j in s..e {
                    scratch[rowind[j] as usize] += data[j] as i64;
                }
            }
        }
    });
    std::hint::black_box(&scratch);

    let (rss1, peak1) = rss_bytes();
    println!(
        "[mmap] after workload ({hits}/{n_queries} hits): WorkingSet={:.1}MB (peak {:.1}MB)",
        mb(rss1),
        mb(peak1)
    );
}

// ============================================================
// 1文字あたりの実推論コスト（読みモデル + 境界モデルの完全版）
// ============================================================
//
// 上の速度計測は語彙bsearchとCSC集計だけを単離したものだが、実際の
// predict() は文字ごとに境界モデル（GBDTなら木を毎文字走査）も評価する。
// `docs/esp32-mcu-feasibility.md` が「計算量ワイルドカード」と呼んでいる
// 部分を含めた、PC上での完全な1文字あたりコストを実測する。

/// テキストを `lines_per_chunk` 行ずつに区切り、各「文字（トークン）」ごとの
/// 候補特徴量キー列を `on_token` に渡す（[`for_each_chunk_keys`] と違い
/// チャンク内でも文字単位を保つ。境界モデルは文字ごとに評価するため）。
#[cfg(windows)]
fn for_each_token_keys(
    text: &str,
    lines_per_chunk: usize,
    mut on_token: impl FnMut(&[FeatureKey]),
) {
    let lines: Vec<&str> = text.lines().collect();
    for chunk in lines.chunks(lines_per_chunk.max(1)) {
        let chunk_text = chunk.join("\n");
        let source_seq = crate::featurize::to_source_seq(&chunk_text);
        let name_flags = vec![crate::name_dict::NAME_FLAG_OUT; source_seq.len()];
        let all_keys = crate::featurize::compute_source_features(&source_seq, &name_flags);
        for keys in &all_keys {
            on_token(keys);
        }
    }
}

#[cfg(windows)]
#[test]
#[ignore]
fn mmap_bench_char_latency() {
    let model = MomoModel::load(std::path::Path::new(&model_path())).expect("モデル読み込み失敗");
    println!(
        "boundary = {}",
        match &model.boundary {
            crate::boundary::Boundary::Linear { .. } => "Linear",
            crate::boundary::Boundary::Tree(_) => "Tree(GBDT)",
        }
    );

    let text = load_corpus_text(&corpus_path(), bench_lines());
    let mut scratch = vec![0i64; model.n_classes as usize];

    let mut run = || -> (usize, usize, usize) {
        let mut hits = 0usize;
        let mut n_queries = 0usize;
        let mut n_tokens = 0usize;
        let mut boundary_acc = 0f32;
        for_each_token_keys(&text, 20, |keys| {
            n_tokens += 1;
            n_queries += keys.len();
            for key in keys {
                if let Some(feat_id) = model.vocab_find(key) {
                    hits += 1;
                    let s = model.csc_colptr[feat_id as usize] as usize;
                    let e = model.csc_colptr[feat_id as usize + 1] as usize;
                    for j in s..e {
                        scratch[model.csc_rowind[j] as usize] += model.csc_data[j] as i64;
                    }
                }
            }
            boundary_acc += model.compute_boundary_score(keys);
        });
        std::hint::black_box(boundary_acc);
        (hits, n_queries, n_tokens)
    };

    run(); // ウォームアップ

    let reps = 10;
    let mut times = Vec::with_capacity(reps);
    let mut last = (0, 0, 0);
    for _ in 0..reps {
        let t0 = Instant::now();
        last = run();
        times.push(t0.elapsed());
    }
    std::hint::black_box(&scratch);

    let (hits, n_queries, n_tokens) = last;
    let med = median(&mut times);
    println!("tokens={n_tokens} queries={n_queries} hits={hits}");
    println!(
        "total={:?} -> {:.0} ns/char （読み候補{:.1}件/字・ヒット{:.1}件/字・境界モデル込み）",
        med,
        med.as_nanos() as f64 / n_tokens as f64,
        n_queries as f64 / n_tokens as f64,
        hits as f64 / n_tokens as f64,
    );
}

// ============================================================
// 境界GBDT木の走査コスト: Boxポインタ木 vs フラット配列(Vec/mmap)
// ============================================================
//
// ⚠️ 【この合成木ベンチマークの結論は実モデルで反証済み】
// 下記の測定では flat が Box より29%速いという結果が出たが、後日実際に
// boundary.rs をフラット化して実モデル(w=4/5/7)で計測したところ、逆に
// 40〜60%遅化した（撤回・Box実装に復帰済み。詳細は
// docs/zerocopy-model-plan.md の「境界GBDT木のフラット化 ― 実装後に撤回した
// 記録」）。原因は下の合成木が1splitあたりカテゴリ1〜8個を仮定していたのに
// 対し、実モデルは平均約28個/split（catsプール全体で約1MBの共有配列）だった
// こと。この差がキャッシュ局所性を逆転させた。このベンチマークは「合成データ
// だけでGO/NO-GO判断をしてはいけない」という反面教師として残す。
//
// 1文字コストの実測で境界GBDT（木300本）が総コストの8割強を占めるとわかった。
// 語彙bsearchはmmapでもほぼ無償だったが、GBDTは木ごとに散らばった小さい
// ヒープ確保（`Box<TreeNode>`）を辿るポインタチェイスであり、語彙の
// 単一フラットソート配列とはアクセスパターンが根本的に異なる ── 同じ
// 結論が成り立つ保証はない。
//
// `boundary.rs` の `TreeNode`/`TreeEnsemble.trees` はモジュール内 private
// なので実木は読めない。代わりに設計メモの規模（木300本・約1MB）に
// 合わせた合成木を組み、
//   (1) 今と同じ Box ポインタ木
//   (2) それをフラット配列へ直列化した Vec 版（同じ常駐メモリだが連続配置）
//   (3) 同じフラット配列を mmap した版
// の3段階を比較する。(1)→(2) で「フラット化そのものの効果」、(2)→(3) で
// 「mmap追加分のコスト」を切り分けられる ── はずだったが、上記の通り
// このベンチマークの前提（cats件数）が実データと乖離していたため機能しなかった。

#[cfg(windows)]
enum SynthNode {
    Leaf(f32),
    Split {
        column: u32,
        cats: Vec<u32>,
        left: Box<SynthNode>,
        right: Box<SynthNode>,
    },
}

#[cfg(windows)]
fn xorshift64(state: &mut u64) -> u64 {
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    x
}

/// 深さ `depth` の合成木を1本組む。列は `MAX_BOUNDARY_CAT_COLUMNS` の範囲、
/// カテゴリ集合は数個程度 ── 実際のGBDT境界モデルの規模感（設計メモの
/// 「木300本・約1MB」）に合わせた大まかな近似。実木の分岐形状そのものは
/// 再現していない（private のため読めない）。
#[cfg(windows)]
fn build_synth_tree(depth: u32, rng: &mut u64) -> SynthNode {
    if depth == 0 {
        return SynthNode::Leaf((xorshift64(rng) % 2000) as f32 / 1000.0 - 1.0);
    }
    let column = (xorshift64(rng) as usize % crate::boundary::MAX_BOUNDARY_CAT_COLUMNS) as u32;
    let n_cats = 1 + (xorshift64(rng) % 8) as usize;
    let mut cats: Vec<u32> = (0..n_cats)
        .map(|_| (xorshift64(rng) % 500) as u32)
        .collect();
    cats.sort_unstable();
    cats.dedup();
    SynthNode::Split {
        column,
        cats,
        left: Box::new(build_synth_tree(depth - 1, rng)),
        right: Box::new(build_synth_tree(depth - 1, rng)),
    }
}

#[cfg(windows)]
fn eval_synth(
    node: &SynthNode,
    col_codes: &[Option<u32>; crate::boundary::MAX_BOUNDARY_CAT_COLUMNS],
) -> f32 {
    match node {
        SynthNode::Leaf(v) => *v,
        SynthNode::Split {
            column,
            cats,
            left,
            right,
        } => {
            let go_left = match col_codes[*column as usize] {
                Some(code) => cats.binary_search(&code).is_ok(),
                None => true,
            };
            eval_synth(if go_left { left } else { right }, col_codes)
        }
    }
}

#[cfg(windows)]
#[repr(C)]
#[derive(Clone, Copy)]
struct FlatNode {
    is_leaf: u32,
    column: u32,
    cats_offset: u32,
    cats_len: u32,
    left_idx: u32,
    right_idx: u32,
    value: f32,
    _pad: u32,
}

/// `SynthNode` 木をフラット配列 (`nodes`) + 共有カテゴリプール (`cats_pool`) へ
/// 直列化する。複数の木で `cats_pool` を共有する（今の統合語彙と同じ発想）。
/// 戻り値: この木の根ノードの `nodes` 内インデックス。
#[cfg(windows)]
fn flatten_synth(node: &SynthNode, nodes: &mut Vec<FlatNode>, cats_pool: &mut Vec<u32>) -> u32 {
    match node {
        SynthNode::Leaf(v) => {
            let idx = nodes.len() as u32;
            nodes.push(FlatNode {
                is_leaf: 1,
                column: 0,
                cats_offset: 0,
                cats_len: 0,
                left_idx: 0,
                right_idx: 0,
                value: *v,
                _pad: 0,
            });
            idx
        }
        SynthNode::Split {
            column,
            cats,
            left,
            right,
        } => {
            let cats_offset = cats_pool.len() as u32;
            cats_pool.extend_from_slice(cats);
            let left_idx = flatten_synth(left, nodes, cats_pool);
            let right_idx = flatten_synth(right, nodes, cats_pool);
            let idx = nodes.len() as u32;
            nodes.push(FlatNode {
                is_leaf: 0,
                column: *column,
                cats_offset,
                cats_len: cats.len() as u32,
                left_idx,
                right_idx,
                value: 0.0,
                _pad: 0,
            });
            idx
        }
    }
}

#[cfg(windows)]
fn eval_flat(
    nodes: &[FlatNode],
    cats_pool: &[u32],
    idx: u32,
    col_codes: &[Option<u32>; crate::boundary::MAX_BOUNDARY_CAT_COLUMNS],
) -> f32 {
    let n = nodes[idx as usize];
    if n.is_leaf != 0 {
        return n.value;
    }
    let go_left = match col_codes[n.column as usize] {
        Some(code) => {
            let cats = &cats_pool[n.cats_offset as usize..(n.cats_offset + n.cats_len) as usize];
            cats.binary_search(&code).is_ok()
        }
        None => true,
    };
    eval_flat(
        nodes,
        cats_pool,
        if go_left { n.left_idx } else { n.right_idx },
        col_codes,
    )
}

#[cfg(windows)]
#[test]
#[ignore]
fn mmap_bench_tree_walk() {
    const N_TREES: usize = 300;
    const DEPTH: u32 = 6; // 2^7-1=127ノード/木 × 300本 ≈ 設計メモの「木300本・約1MB」規模
    const N_COLS: usize = crate::boundary::MAX_BOUNDARY_CAT_COLUMNS;

    let mut rng: u64 = 0x243F_6A88_85A3_08D3;
    let mut box_trees = Vec::with_capacity(N_TREES);
    let mut flat_nodes: Vec<FlatNode> = Vec::new();
    let mut cats_pool: Vec<u32> = Vec::new();
    let mut tree_roots: Vec<u32> = Vec::with_capacity(N_TREES);
    for _ in 0..N_TREES {
        let t = build_synth_tree(DEPTH, &mut rng);
        let root = flatten_synth(&t, &mut flat_nodes, &mut cats_pool);
        tree_roots.push(root);
        box_trees.push(t);
    }
    println!(
        "synthetic ensemble: trees={N_TREES} depth={DEPTH} nodes={} cats_pool={} (flat ~{:.2}MB)",
        flat_nodes.len(),
        cats_pool.len(),
        (flat_nodes.len() * std::mem::size_of::<FlatNode>() + cats_pool.len() * 4) as f64
            / (1024.0 * 1024.0)
    );

    let nodes_tmp =
        std::env::temp_dir().join(format!("momo_bench_tree_nodes_{}.bin", std::process::id()));
    let cats_tmp =
        std::env::temp_dir().join(format!("momo_bench_tree_cats_{}.bin", std::process::id()));
    write_raw(&nodes_tmp, &flat_nodes);
    write_raw(&cats_tmp, &cats_pool);
    let nodes_mmap = mmap_file(&nodes_tmp);
    let cats_mmap = mmap_file(&cats_tmp);
    let mmap_nodes: &[FlatNode] = as_slice(&nodes_mmap);
    let mmap_cats: &[u32] = as_slice(&cats_mmap);

    // 呼び出しごとに異なる col_codes（アクティブな列とコード）を用意する。
    // 実際は文字ごとに発火する特徴量が違うため、走査経路も文字ごとに変わる。
    let n_calls = 20_000;
    let col_codes_sets: Vec<[Option<u32>; N_COLS]> = (0..n_calls)
        .map(|i| {
            let mut cc = [None; N_COLS];
            for col in 0..N_COLS {
                if hash_index((i, col, 0u8), 100) < 55 {
                    cc[col] = Some(hash_index((i, col, 1u8), 500) as u32);
                }
            }
            cc
        })
        .collect();

    let eval_box =
        |cc: &[Option<u32>; N_COLS]| -> f32 { box_trees.iter().map(|t| eval_synth(t, cc)).sum() };
    let eval_flat_vec = |cc: &[Option<u32>; N_COLS]| -> f32 {
        tree_roots
            .iter()
            .map(|&r| eval_flat(&flat_nodes, &cats_pool, r, cc))
            .sum()
    };
    let eval_flat_mmap = |cc: &[Option<u32>; N_COLS]| -> f32 {
        tree_roots
            .iter()
            .map(|&r| eval_flat(mmap_nodes, mmap_cats, r, cc))
            .sum()
    };

    // ウォームアップ
    for cc in &col_codes_sets {
        std::hint::black_box(eval_box(cc));
        std::hint::black_box(eval_flat_vec(cc));
        std::hint::black_box(eval_flat_mmap(cc));
    }

    let reps = 10;
    let mut box_times = Vec::with_capacity(reps);
    let mut vec_times = Vec::with_capacity(reps);
    let mut mmap_times = Vec::with_capacity(reps);
    for _ in 0..reps {
        let t0 = Instant::now();
        for cc in &col_codes_sets {
            std::hint::black_box(eval_box(cc));
        }
        box_times.push(t0.elapsed());

        let t0 = Instant::now();
        for cc in &col_codes_sets {
            std::hint::black_box(eval_flat_vec(cc));
        }
        vec_times.push(t0.elapsed());

        let t0 = Instant::now();
        for cc in &col_codes_sets {
            std::hint::black_box(eval_flat_mmap(cc));
        }
        mmap_times.push(t0.elapsed());
    }

    let box_ns = report("tree_walk [Box     ]", box_times, n_calls);
    let vec_ns = report("tree_walk [FlatVec ]", vec_times, n_calls);
    let mmap_ns = report("tree_walk [FlatMmap]", mmap_times, n_calls);
    println!(
        "flat/box = {:.2}x, mmap/flatvec = {:.2}x, mmap/box = {:.2}x",
        vec_ns / box_ns,
        mmap_ns / vec_ns,
        mmap_ns / box_ns
    );

    let _ = std::fs::remove_file(&nodes_tmp);
    let _ = std::fs::remove_file(&cats_tmp);
}
