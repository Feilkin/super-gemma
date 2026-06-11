# Framework Desktop bring-up

One-time setup of the target box (AMD Ryzen AI Max+ 395, Radeon 8060S, 128 GB unified, NVMe,
Linux). Commands are guidance, not gospel — adjust for the distro actually installed and verify
each step's output.

## 1. Toolchain

```sh
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh   # rustup + stable
# rust-toolchain.toml pins stable + rustfmt + clippy automatically on first cargo run
curl -LsSf https://get.nexte.st/latest/linux | tar zxf - -C "$HOME/.cargo/bin"   # cargo-nextest
```

Sanity: `cargo --version`, `cargo nextest --version`, `git --version`.

## 2. GPU stack

- Mesa with RADV supporting gfx1151 and `VK_KHR_cooperative_matrix` (recent Mesa; Ada has already
  used coopmat on this box, so the installed stack is known-good — don't downgrade it).
- Verify: `vulkaninfo --summary` shows the Radeon 8060S on RADV, then:

```sh
cargo run --release -p sg-probe -- vulkan
```

Expect `shader_float16`, `storage_buffer16_bit_access`, `timeline_semaphore`,
`cooperative_matrix` all true and a non-empty `cooperative_matrix_configs` list.

- **GTT size**: the weights + KV buffers need ~32 GB of GPU-visible memory at full context; check
  `dmesg | grep -i gtt` for the limit and raise via `amdgpu.gttsize=` kernel param /
  `/etc/modprobe.d/` if it's below ~64 GB. Confirm against `sg-probe` heap sizes.

## 3. io_uring

Modern kernel assumed. Check it isn't disabled: `sysctl kernel.io_uring_disabled` should be 0
(or the sysctl absent on older kernels).

## 4. Model files

Keep models out of the repo. Suggested location: `/srv/models/`.

```sh
pip install -U "huggingface_hub[cli]"
hf download google/gemma-4-31B-it-qat-q4_0-gguf --local-dir /srv/models/gemma-4-31b-q4_0
hf download google/gemma-4-31B-it-assistant     --local-dir /srv/models/gemma-4-31b-assistant   # MTP drafter (M7.5)
```

Record sha256 sums of the downloaded weights in `docs/reference/` alongside a note of the
download date (a `tools/` script for this is fine to add).

Also fetch for the M1 fixture generators (plan 01):
`tokenizer_config.json` from `google/gemma-4-31B-it` (chat/tool template source).

## 5. Complete M0

Generate and check in the probe reports per `docs/probe/README.md` (use the GGUF file for the
NVMe read test). Then run the full gate:

```sh
cargo fmt --all --check
cargo clippy --workspace --all-targets -- -D warnings
cargo nextest run --workspace
```

## 6. (Optional now, required before merging hot-path work) Tier 2 runner

GitHub self-hosted runner on this box with labels `self-hosted, linux, framework`; then enable
the `schedule:`/PR triggers in `.github/workflows/target-box.yml`.
