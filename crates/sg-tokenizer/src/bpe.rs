//! The BPE merge engine, mirroring HF `tokenizers`' `Word::merge_all`.
//!
//! Symbols start as one vocab id per character (after `" " → "▁"`
//! normalization), with unknown characters expanded to `<0xXX>` byte tokens
//! *before* merging (that is where HF's `byte_fallback` hooks in — byte
//! tokens participate in merges, though Gemma 4's merge list never touches
//! them). Then the lowest-ranked adjacent pair is merged repeatedly; ties
//! break on the leftmost position, exactly like the reference heap.

use std::cmp::Reverse;
use std::collections::BinaryHeap;

use crate::vocab::Vocab;

/// One entry in the doubly linked symbol list. `usize::MAX` marks "none".
struct Sym {
    id: u32,
    prev: usize,
    next: usize,
    alive: bool,
}

const NONE: usize = usize::MAX;

/// Encode one special-token-free segment, appending ids to `out`.
pub(crate) fn encode_segment(vocab: &Vocab, text: &str, out: &mut Vec<u32>) {
    if text.is_empty() {
        return;
    }

    // Normalize and look up initial symbols.
    let mut syms: Vec<Sym> = Vec::with_capacity(text.len());
    let mut char_buf = [0u8; 4];
    for ch in text.chars() {
        let ch = if ch == ' ' { '\u{2581}' } else { ch }; // " " → "▁"
        let piece: &str = ch.encode_utf8(&mut char_buf);
        match vocab.id_of(piece) {
            Some(id) => push_sym(&mut syms, id),
            None => {
                // Byte fallback: the byte table is validated complete, so
                // every character can be spelled.
                for &b in piece.as_bytes() {
                    push_sym(&mut syms, vocab.byte_id(b));
                }
            }
        }
    }

    // Seed the heap with every adjacent mergeable pair. Entries are
    // (rank, left position); stale entries are skipped on pop.
    let mut heap: BinaryHeap<Reverse<(u32, usize, u32, u32)>> = BinaryHeap::new();
    let push_pair = |heap: &mut BinaryHeap<_>, syms: &[Sym], left: usize| {
        let right = syms[left].next;
        if right == NONE {
            return;
        }
        let (l, r) = (syms[left].id, syms[right].id);
        if let Some((rank, _)) = vocab.merge(l, r) {
            heap.push(Reverse((rank, left, l, r)));
        }
    };
    for i in 0..syms.len().saturating_sub(1) {
        push_pair(&mut heap, &syms, i);
    }

    while let Some(Reverse((rank, pos, l, r))) = heap.pop() {
        // Validate against the current list: the pair may have been consumed
        // by an earlier merge.
        if !syms[pos].alive || syms[pos].id != l {
            continue;
        }
        let right = syms[pos].next;
        if right == NONE || syms[right].id != r {
            continue;
        }
        let Some((cur_rank, merged)) = vocab.merge(l, r) else {
            continue;
        };
        if cur_rank != rank {
            continue;
        }

        // Merge `right` into `pos`.
        syms[pos].id = merged;
        syms[right].alive = false;
        let after = syms[right].next;
        syms[pos].next = after;
        if after != NONE {
            syms[after].prev = pos;
        }

        // New candidate pairs with both neighbors.
        if syms[pos].prev != NONE {
            push_pair(&mut heap, &syms, syms[pos].prev);
        }
        push_pair(&mut heap, &syms, pos);
    }

    out.extend(syms.iter().filter(|s| s.alive).map(|s| s.id));
}

fn push_sym(syms: &mut Vec<Sym>, id: u32) {
    let pos = syms.len();
    if let Some(last) = syms.last_mut() {
        last.next = pos;
    }
    syms.push(Sym {
        id,
        prev: if pos == 0 { NONE } else { pos - 1 },
        next: NONE,
        alive: true,
    });
}
