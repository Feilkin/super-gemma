# Hardware probe reports

M0 exit criterion: the reports below, generated **on the target Framework Desktop**, checked in.
Reports from dev machines are smoke tests only — never feed them into the cache2 cost model or
kernel-variant selection.

Generate on the target:

```sh
cargo run --release -p sg-probe -- vulkan              > docs/probe/vulkan.json
cargo run --release -p sg-probe -- membw               > docs/probe/membw.json
cargo run --release -p sg-probe -- nvme <big-file>     > docs/probe/nvme.json
```

For the NVMe probe use a multi-GiB file on the NVMe that will host cache2 (the model GGUF is
ideal). `o_direct: false` in the output means the number is page-cache-inflated — rerun on a
filesystem that supports O_DIRECT.

Of particular interest for plan 02: `cooperative_matrix_configs` (drives which GEMM variants get
compiled) and `subgroup_size`.
