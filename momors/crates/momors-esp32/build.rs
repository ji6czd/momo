fn main() {
    // sdkconfig 由来の cfg（esp-idf-sys が出力する）を check-cfg に宣言して警告を抑える
    println!("cargo::rustc-check-cfg=cfg(esp_idf_esp_console_usb_serial_jtag)");
    embuild::espidf::sysenv::output();
}
