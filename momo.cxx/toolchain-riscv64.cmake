# RISC-V 64-bit Linux クロスコンパイルツールチェーン
# Milk-V ボード (CV1800B, CV181X などの RISC-V 64-bit ボード用)
# musl C ライブラリを使用

set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR riscv64)

# ツールチェーンのパス
# envsetup.sh と同じ toolchain-musl を使用
set(TOOLCHAIN_PATH "$ENV{HOME}/host-tools/gcc/riscv64-linux-musl-x86_64")

# コンパイラ設定
set(CMAKE_C_COMPILER ${TOOLCHAIN_PATH}/bin/riscv64-unknown-linux-musl-gcc)
set(CMAKE_CXX_COMPILER ${TOOLCHAIN_PATH}/bin/riscv64-unknown-linux-musl-g++)
set(CMAKE_AR ${TOOLCHAIN_PATH}/bin/riscv64-unknown-linux-musl-ar)
set(CMAKE_RANLIB ${TOOLCHAIN_PATH}/bin/riscv64-unknown-linux-musl-ranlib)
set(CMAKE_STRIP ${TOOLCHAIN_PATH}/bin/riscv64-unknown-linux-musl-strip)

# find_program のコマンドを実行するときのプレフィックス
set(CMAKE_FIND_ROOT_PATH ${TOOLCHAIN_PATH}/riscv64-unknown-linux-musl)

# クロスコンパイル時の動作設定
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)

# Milk-V ボード用のコンパイラフラグ
# CV1800B / CV181X の C906FDV コアに最適化
set(CMAKE_C_FLAGS_INIT "-mcpu=c906fdv -march=rv64imafdcv0p7xthead -mcmodel=medany -mabi=lp64d")
set(CMAKE_CXX_FLAGS_INIT "-mcpu=c906fdv -march=rv64imafdcv0p7xthead -mcmodel=medany -mabi=lp64d")
set(CMAKE_EXE_LINKER_FLAGS_INIT "-D_LARGEFILE_SOURCE -D_LARGEFILE64_SOURCE -D_FILE_OFFSET_BITS=64")
