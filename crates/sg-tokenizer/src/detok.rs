//! UTF-8-safe accumulation of `<0xXX>` byte-token runs for streaming decode.

/// Holds bytes from byte-fallback tokens until they form complete UTF-8.
///
/// Multi-byte code points (emoji, CJK in byte fallback) span several tokens;
/// SSE deltas must withhold the partial prefix instead of emitting broken
/// UTF-8. Invalid sequences degrade to U+FFFD with the same maximal-subpart
/// rule as `String::from_utf8_lossy`.
#[derive(Debug, Default)]
pub struct DetokBuffer {
    pending: Vec<u8>,
}

impl DetokBuffer {
    pub fn new() -> Self {
        Self::default()
    }

    /// True if a partial code point is being withheld.
    pub fn has_pending(&self) -> bool {
        !self.pending.is_empty()
    }

    /// Add one byte; move every *complete* code point (or replacement char
    /// for invalid bytes) to `out`, keeping only a valid-so-far suffix.
    pub(crate) fn push_byte(&mut self, b: u8, out: &mut String) {
        self.pending.push(b);
        loop {
            match str::from_utf8(&self.pending) {
                Ok(s) => {
                    out.push_str(s);
                    self.pending.clear();
                    return;
                }
                Err(e) => {
                    let valid = e.valid_up_to();
                    // SAFETY-free split: `valid` is a UTF-8 boundary per the error.
                    out.push_str(str::from_utf8(&self.pending[..valid]).expect("valid prefix"));
                    match e.error_len() {
                        // Invalid bytes: replace and keep scanning the rest.
                        Some(n) => {
                            out.push('\u{FFFD}');
                            self.pending.drain(..valid + n);
                        }
                        // Incomplete tail: withhold it.
                        None => {
                            self.pending.drain(..valid);
                            return;
                        }
                    }
                }
            }
        }
    }

    /// Emit any withheld partial code point as U+FFFD (run ended without
    /// completing it).
    pub(crate) fn flush(&mut self, out: &mut String) {
        if !self.pending.is_empty() {
            out.push('\u{FFFD}');
            self.pending.clear();
        }
    }
}
