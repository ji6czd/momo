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
