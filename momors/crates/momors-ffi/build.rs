fn main() {
    let crate_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    let config_path = std::path::Path::new(&crate_dir).join("cbindgen.toml");

    let config = if config_path.exists() {
        cbindgen::Config::from_file(&config_path).expect("cbindgen.toml の読み込みに失敗")
    } else {
        cbindgen::Config::default()
    };

    cbindgen::Builder::new()
        .with_crate(&crate_dir)
        .with_config(config)
        .generate()
        .expect("cbindgen によるヘッダ生成に失敗")
        .write_to_file(std::path::Path::new(&crate_dir).join("momo.h"));

    // src/lib.rs が変更されたときだけ再実行する
    println!("cargo:rerun-if-changed=src/lib.rs");
    println!("cargo:rerun-if-changed=cbindgen.toml");
}
