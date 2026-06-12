//! Pre-recorded command graph tests (plan 02 step 8): a miniature decode
//! step — kv_append (K and V) → split-K sliding attention → reduce — is
//! recorded ONCE, then submitted repeatedly while the CPU rewrites only the
//! step buffer (and the new token's K/V rows) between submits. Verifies the
//! record-once/run-many mechanism, vulkano's auto-inserted barriers between
//! dependent dispatches, step-buffer-driven control, bit-determinism of a
//! re-driven graph, and timestamp timing. Skips without a GPU.

mod reference;

use reference::{Rng, assert_close, attention_head, from_f16_bits, through_f16, to_f16_bits};
use sg_gpu::{GpuContext, StepState};
use vulkano::buffer::{BufferContents, BufferUsage};
use vulkano::descriptor_set::WriteDescriptorSet;

const N_Q_HEADS: usize = 32;
const KV_HEADS: usize = 16;
const DIM: usize = 256;
const ROW: usize = KV_HEADS * DIM; // one token's K (or V) rows
const RING: usize = 1024;
const SCALE: f32 = 0.0625;
const N_SPLITS: u32 = 2;
const PART_STRIDE: usize = DIM + 2;
const STEPS: usize = 3;

#[derive(BufferContents, Clone, Copy)]
#[repr(C)]
struct PushSplitScale {
    n_splits: u32,
    scale: f32,
}

#[test]
fn recorded_graph_follows_step_updates() {
    let Some(ctx) = (match GpuContext::new() {
        Ok(ctx) => Some(ctx),
        Err(e) => {
            eprintln!("skipping: no usable GPU ({e})");
            None
        }
    }) else {
        return;
    };
    let append_k = ctx.load_kernel("kv_append_sliding").unwrap();
    let attn_k = ctx.load_kernel("attn_decode_sliding").unwrap();
    let red_k = ctx.load_kernel("attn_reduce_d256").unwrap();
    let mut rng = Rng::new(0x66AF);

    let q = through_f16(&rng.f32_vec(N_Q_HEADS * DIM));
    // K/V rows for each step, fed through the same 1-token src buffers.
    let k_rows: Vec<Vec<f32>> = (0..STEPS).map(|_| through_f16(&rng.f32_vec(ROW))).collect();
    let v_rows: Vec<Vec<f32>> = (0..STEPS).map(|_| through_f16(&rng.f32_vec(ROW))).collect();

    let q_buf = ctx
        .buffer_from_iter(to_f16_bits(&q), BufferUsage::STORAGE_BUFFER)
        .unwrap();
    let src_k = ctx
        .new_buffer::<u16>(ROW as u64, BufferUsage::STORAGE_BUFFER)
        .unwrap();
    let src_v = ctx
        .new_buffer::<u16>(ROW as u64, BufferUsage::STORAGE_BUFFER)
        .unwrap();
    let ring_k = ctx
        .new_buffer::<u16>((RING * ROW) as u64, BufferUsage::STORAGE_BUFFER)
        .unwrap();
    let ring_v = ctx
        .new_buffer::<u16>((RING * ROW) as u64, BufferUsage::STORAGE_BUFFER)
        .unwrap();
    let part = ctx
        .new_buffer::<f32>(
            (N_Q_HEADS * N_SPLITS as usize * PART_STRIDE) as u64,
            BufferUsage::STORAGE_BUFFER,
        )
        .unwrap();
    let out = ctx
        .new_buffer::<u16>((N_Q_HEADS * DIM) as u64, BufferUsage::STORAGE_BUFFER)
        .unwrap();
    let step = ctx.new_step_buffer().unwrap();

    // Recorded ONCE; everything per-step flows through `step` and the src
    // buffers. The timer brackets the whole step.
    let timer = ctx.new_timer(2).unwrap();
    let graph = ctx
        .record_graph(|rec| {
            rec.reset_timer(&timer)?
                .timestamp(&timer, 0)?
                .dispatch(
                    &append_k,
                    vec![
                        WriteDescriptorSet::buffer(0, src_k.clone()),
                        WriteDescriptorSet::buffer(1, ring_k.clone()),
                        WriteDescriptorSet::buffer(2, step.clone()),
                    ],
                    None::<u32>,
                    append_k.groups_for(ROW as u64),
                )?
                .dispatch(
                    &append_k,
                    vec![
                        WriteDescriptorSet::buffer(0, src_v.clone()),
                        WriteDescriptorSet::buffer(1, ring_v.clone()),
                        WriteDescriptorSet::buffer(2, step.clone()),
                    ],
                    None::<u32>,
                    append_k.groups_for(ROW as u64),
                )?
                .dispatch(
                    &attn_k,
                    vec![
                        WriteDescriptorSet::buffer(0, q_buf.clone()),
                        WriteDescriptorSet::buffer(1, ring_k.clone()),
                        WriteDescriptorSet::buffer(2, ring_v.clone()),
                        WriteDescriptorSet::buffer(3, part.clone()),
                        WriteDescriptorSet::buffer(4, step.clone()),
                    ],
                    Some(PushSplitScale {
                        n_splits: N_SPLITS,
                        scale: SCALE,
                    }),
                    [KV_HEADS as u32, N_SPLITS, 1],
                )?
                .dispatch(
                    &red_k,
                    vec![
                        WriteDescriptorSet::buffer(0, part.clone()),
                        WriteDescriptorSet::buffer(1, out.clone()),
                    ],
                    Some(N_SPLITS),
                    [N_Q_HEADS as u32, 1, 1],
                )?
                .timestamp(&timer, 1)?;
            Ok(())
        })
        .unwrap();

    let run = |outs: &mut Vec<Vec<u16>>| {
        for i in 0..STEPS {
            src_k
                .write()
                .unwrap()
                .copy_from_slice(&to_f16_bits(&k_rows[i]));
            src_v
                .write()
                .unwrap()
                .copy_from_slice(&to_f16_bits(&v_rows[i]));
            StepState {
                pos: i as u32,
                kv_len_sliding: (i + 1) as u32,
                ..Default::default()
            }
            .write_to(&step)
            .unwrap();
            ctx.submit_blocking(&graph).unwrap();
            outs.push(out.read().unwrap().to_vec());
        }
    };

    let mut outs = Vec::new();
    run(&mut outs);

    // Each step must equal full-softmax attention over the tokens appended
    // so far — the graph reacted to pos/kv_len without re-recording.
    for (i, out_bits) in outs.iter().enumerate() {
        let got = from_f16_bits(out_bits);
        let k_lin: Vec<f32> = k_rows[..=i].concat();
        let v_lin: Vec<f32> = v_rows[..=i].concat();
        for qh in [0usize, 5, 13, 31] {
            let want = attention_head(
                &q,
                &k_lin,
                &v_lin,
                0,
                qh,
                N_Q_HEADS,
                KV_HEADS,
                DIM,
                SCALE as f64,
                0,
                i,
            );
            assert_close(
                &got[qh * DIM..][..DIM],
                &want,
                1e-2,
                1e-2,
                &format!("graph step {i} qh={qh}"),
            );
        }
    }

    // Timestamps from the last submit: ordered and plausible.
    let ts = timer.read_ns().unwrap();
    assert!(ts[1] > ts[0], "timestamps not ordered: {ts:?}");
    let dur_ms = (ts[1] - ts[0]) / 1e6;
    assert!(dur_ms < 100.0, "graph 'GPU time' implausible: {dur_ms} ms");

    // Plan 02/06: re-driving the same graph through the same step sequence
    // is bit-identical (appends overwrite the same slots).
    let mut outs2 = Vec::new();
    run(&mut outs2);
    assert_eq!(outs, outs2, "re-driven graph diverged");
}
