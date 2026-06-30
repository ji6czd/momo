use std::path::Path;

fn main() {
    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    let dataset_src =
        Path::new(&manifest_dir).join("../../../dataset/japanese_grade1_braille.toml");
    let local_dst = Path::new(&manifest_dir).join("data/japanese_grade1_braille.toml");

    if dataset_src.exists() {
        std::fs::create_dir_all(local_dst.parent().unwrap()).unwrap();
        std::fs::copy(&dataset_src, &local_dst).unwrap();
        println!("cargo:rerun-if-changed={}", dataset_src.display());
    } else {
        println!("cargo:rerun-if-changed={}", local_dst.display());
    }
}
