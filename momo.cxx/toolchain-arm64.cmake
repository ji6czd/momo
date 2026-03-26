# ARM64 (aarch64) Linux クロスコンパイルツールチェーン
# Milk-V DuoS / Duo256M (SG2000/SG2002 - ARM64 選択時) 用

set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR aarch64)

# ツールチェーンのパス
# Linaro GCC ツールチェーンを使用
set(TOOLCHAIN_PATH "$ENV{HOME}/host-tools/gcc/gcc-linaro-7.3.1-2018.05-x86_64_aarch64-linux-gnu")

# コンパイラ設定
set(CMAKE_C_COMPILER ${TOOLCHAIN_PATH}/bin/aarch64-linux-gnu-gcc)
set(CMAKE_CXX_COMPILER ${TOOLCHAIN_PATH}/bin/aarch64-linux-gnu-g++)
set(CMAKE_AR ${TOOLCHAIN_PATH}/bin/aarch64-linux-gnu-ar)
set(CMAKE_RANLIB ${TOOLCHAIN_PATH}/bin/aarch64-linux-gnu-ranlib)
set(CMAKE_STRIP ${TOOLCHAIN_PATH}/bin/aarch64-linux-gnu-strip)

# find_program のコマンドを実行するときのプレフィックス
set(CMAKE_FIND_ROOT_PATH ${TOOLCHAIN_PATH}/aarch64-linux-gnu)

# クロスコンパイル時の動作設定
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)

# ARM64 用のコンパイラフラグ
set(CMAKE_C_FLAGS_INIT "-march=armv8-a")
set(CMAKE_CXX_FLAGS_INIT "-march=armv8-a")
