//! Encode/decode entry points: special-token matching around the BPE engine,
//! and the decoder (`▁` → space, byte-token fusion).

use crate::bpe;
use crate::detok::DetokBuffer;
use crate::vocab::{Vocab, VocabError};

/// Whether `encode` recognizes special tokens written out in the text.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpecialTokens {
    /// Match control/user-defined tokens (leftmost-longest), like HF's
    /// default. The prompt builder uses this for template text it generated
    /// itself.
    Match,
    /// Treat the text as plain content: `"<|turn>"` in user input stays
    /// literal text and can never inject a control token.
    Plain,
}

#[derive(Debug, thiserror::Error)]
pub enum DecodeError {
    #[error("token id {0} out of range for the vocab")]
    IdOutOfRange(u32),
}

/// The Gemma 4 tokenizer (BPE; see [`Vocab`] for provenance).
#[derive(Debug)]
pub struct Tokenizer {
    vocab: Vocab,
}

impl Tokenizer {
    pub fn new(vocab: Vocab) -> Self {
        Self { vocab }
    }

    /// Build from GGUF metadata (`tokenizer.ggml.*`).
    pub fn from_metadata(meta: &sg_gguf::Metadata) -> Result<Self, VocabError> {
        Ok(Self::new(Vocab::from_metadata(meta)?))
    }

    pub fn vocab(&self) -> &Vocab {
        &self.vocab
    }

    /// Encode `text` to token ids. No BOS is added (Gemma 4's
    /// `add_bos_token = false`; the chat template inserts `<bos>` itself).
    pub fn encode(&self, text: &str, specials: SpecialTokens) -> Vec<u32> {
        let mut out = Vec::with_capacity(text.len() / 3 + 4);
        match specials {
            SpecialTokens::Plain => bpe::encode_segment(&self.vocab, text, &mut out),
            SpecialTokens::Match => {
                let mut rest = text;
                while !rest.is_empty() {
                    match self.find_special(rest) {
                        Some((start, len, id)) => {
                            bpe::encode_segment(&self.vocab, &rest[..start], &mut out);
                            out.push(id);
                            rest = &rest[start + len..];
                        }
                        None => {
                            bpe::encode_segment(&self.vocab, rest, &mut out);
                            break;
                        }
                    }
                }
            }
        }
        out
    }

    /// Leftmost-longest special-token match: `(byte offset, length, id)`.
    fn find_special(&self, text: &str) -> Option<(usize, usize, u32)> {
        // All specials start with '<'; scan byte-wise and try the
        // longest-first list at each candidate. 24 patterns, so brute force
        // beats building an automaton.
        for (start, _) in text.match_indices('<') {
            for (piece, id) in self.vocab.specials() {
                if text[start..].starts_with(&**piece) {
                    return Some((start, piece.len(), *id));
                }
            }
        }
        None
    }

    /// Decode ids to text. Special tokens render as their literal piece;
    /// invalid byte-token runs become U+FFFD.
    pub fn decode(&self, ids: &[u32]) -> Result<String, DecodeError> {
        let mut buf = DetokBuffer::new();
        let mut out = String::new();
        for &id in ids {
            self.decode_streaming(id, &mut buf, &mut out)?;
        }
        self.flush_streaming(&mut buf, &mut out);
        Ok(out)
    }

    /// Streaming decode: append whatever `id` makes printable to `out`,
    /// holding incomplete UTF-8 byte-token runs in `buf` (SSE deltas must
    /// never split a code point).
    pub fn decode_streaming(
        &self,
        id: u32,
        buf: &mut DetokBuffer,
        out: &mut String,
    ) -> Result<(), DecodeError> {
        if let Some(b) = self.vocab.byte_of(id) {
            buf.push_byte(b, out);
            return Ok(());
        }
        let piece = self.vocab.piece(id).ok_or(DecodeError::IdOutOfRange(id))?;
        // A non-byte token terminates any pending byte run.
        buf.flush(out);
        if piece.contains('\u{2581}') {
            out.extend(piece.chars().map(|c| if c == '\u{2581}' { ' ' } else { c }));
        } else {
            out.push_str(piece);
        }
        Ok(())
    }

    /// Flush an unterminated byte-token run (end of generation).
    pub fn flush_streaming(&self, buf: &mut DetokBuffer, out: &mut String) {
        buf.flush(out);
    }
}
