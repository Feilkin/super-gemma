//! Vocabulary, merge table, byte-fallback map, and special tokens, built from
//! the GGUF-embedded tokenizer payload.
//!
//! Algorithm provenance: the official `tokenizer.json` of
//! `google/gemma-4-31B-it` (see `docs/reference/`) declares a **BPE** model
//! with `byte_fallback`, normalizer `Replace(" " → "▁")`, no effective
//! pre-tokenization, and 24 plain-matched added special tokens. The GGUF
//! carries the same data (`tokenizer.ggml.model = "gemma4"`, dummy scores,
//! merge list); construction validates every assumption it relies on.

use std::collections::HashMap;

use sg_gguf::Metadata;

/// llama.cpp token type ids as stored in `tokenizer.ggml.token_type`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TokenType {
    Undefined,
    Normal,
    Unknown,
    Control,
    UserDefined,
    Unused,
    Byte,
}

impl TokenType {
    fn from_raw(raw: i32) -> Option<Self> {
        Some(match raw {
            0 => Self::Undefined,
            1 => Self::Normal,
            2 => Self::Unknown,
            3 => Self::Control,
            4 => Self::UserDefined,
            5 => Self::Unused,
            6 => Self::Byte,
            _ => return None,
        })
    }
}

#[derive(Debug, thiserror::Error)]
pub enum VocabError {
    #[error(transparent)]
    Meta(#[from] sg_gguf::MetaError),
    #[error("tokenizer.ggml.model is `{0}`, expected `gemma4`")]
    WrongModel(String),
    #[error("tokenizer arrays disagree: {tokens} tokens, {scores} scores, {types} types")]
    ArrayLengths {
        tokens: usize,
        scores: usize,
        types: usize,
    },
    #[error("duplicate token piece {0:?}")]
    DuplicatePiece(String),
    #[error("token {id} has invalid token_type {raw}")]
    BadTokenType { id: u32, raw: i32 },
    #[error("byte token {0:?} is not of the form <0xXX>")]
    BadByteToken(String),
    #[error("byte-fallback table incomplete: {0} of 256 byte tokens present")]
    IncompleteByteTable(usize),
    #[error("merge entry {index} ({merge:?}) is not two space-separated pieces")]
    BadMergeFormat { index: usize, merge: String },
    #[error("merge entry {index} ({merge:?}) references pieces missing from the vocab")]
    MergePieceMissing { index: usize, merge: String },
    #[error("`{key}` = {id} but that token is {piece:?}, expected {expected:?}")]
    SpecialIdMismatch {
        key: &'static str,
        id: u32,
        piece: String,
        expected: &'static str,
    },
    #[error("metadata declares {key} = true; this tokenizer is built for false")]
    UnsupportedFlag { key: &'static str },
    #[error("token id {0} out of range")]
    IdOutOfRange(u64),
}

/// Immutable tokenizer tables.
#[derive(Debug)]
pub struct Vocab {
    pieces: Vec<Box<str>>,
    types: Vec<TokenType>,
    piece_to_id: HashMap<Box<str>, u32>,
    /// `(left, right) -> (rank, merged)`; rank is the merge-list index.
    merges: HashMap<(u32, u32), (u32, u32)>,
    /// Byte value `b` -> id of the `<0xXX>` token.
    byte_to_id: [u32; 256],
    /// id -> byte value for `TokenType::Byte` tokens.
    id_to_byte: HashMap<u32, u8>,
    /// Control + user-defined tokens, longest piece first (leftmost-longest
    /// matching).
    specials: Vec<(Box<str>, u32)>,
    bos: u32,
    eos: u32,
    pad: u32,
    unk: u32,
    /// `<turn|>` — Gemma 4's end-of-turn, the second stop token.
    eot: u32,
}

impl Vocab {
    /// Build and validate from GGUF metadata (`tokenizer.ggml.*`).
    pub fn from_metadata(meta: &Metadata) -> Result<Self, VocabError> {
        let model = meta.require_str("tokenizer.ggml.model")?;
        if model != "gemma4" {
            return Err(VocabError::WrongModel(model.to_owned()));
        }
        // The encoder hard-codes "no auto-BOS, no dummy space prefix"; refuse
        // a file that asks for the opposite instead of silently diverging.
        for key in [
            "tokenizer.ggml.add_bos_token",
            "tokenizer.ggml.add_space_prefix",
        ] {
            if meta.get_bool(key)? == Some(true) {
                return Err(VocabError::UnsupportedFlag {
                    key: if key.ends_with("bos_token") {
                        "tokenizer.ggml.add_bos_token"
                    } else {
                        "tokenizer.ggml.add_space_prefix"
                    },
                });
            }
        }

        let tokens = meta.require_str_array("tokenizer.ggml.tokens")?;
        let scores = meta.require_f32_array("tokenizer.ggml.scores")?;
        let raw_types = meta.require_i32_array("tokenizer.ggml.token_type")?;
        if tokens.len() != scores.len() || tokens.len() != raw_types.len() {
            return Err(VocabError::ArrayLengths {
                tokens: tokens.len(),
                scores: scores.len(),
                types: raw_types.len(),
            });
        }

        let mut pieces = Vec::with_capacity(tokens.len());
        let mut types = Vec::with_capacity(tokens.len());
        let mut piece_to_id = HashMap::with_capacity(tokens.len());
        let mut byte_to_id = [u32::MAX; 256];
        let mut id_to_byte = HashMap::new();
        let mut specials: Vec<(Box<str>, u32)> = Vec::new();
        let mut byte_count = 0usize;

        for (id, (piece, &raw_ty)) in tokens.iter().zip(raw_types).enumerate() {
            let id = id as u32;
            let ty =
                TokenType::from_raw(raw_ty).ok_or(VocabError::BadTokenType { id, raw: raw_ty })?;
            if piece_to_id
                .insert(piece.clone().into_boxed_str(), id)
                .is_some()
            {
                return Err(VocabError::DuplicatePiece(piece.clone()));
            }
            match ty {
                TokenType::Byte => {
                    let b = parse_byte_piece(piece)
                        .ok_or_else(|| VocabError::BadByteToken(piece.clone()))?;
                    byte_to_id[b as usize] = id;
                    id_to_byte.insert(id, b);
                    byte_count += 1;
                }
                TokenType::Control | TokenType::UserDefined => {
                    specials.push((piece.clone().into_boxed_str(), id));
                }
                _ => {}
            }
            pieces.push(piece.clone().into_boxed_str());
            types.push(ty);
        }
        if byte_count != 256 || byte_to_id.contains(&u32::MAX) {
            return Err(VocabError::IncompleteByteTable(byte_count));
        }
        specials.sort_by_key(|(piece, _)| std::cmp::Reverse(piece.len()));

        let merge_strs = meta.require_str_array("tokenizer.ggml.merges")?;
        let mut merges = HashMap::with_capacity(merge_strs.len());
        for (index, merge) in merge_strs.iter().enumerate() {
            let (left, right) = merge
                .split_once(' ')
                .filter(|(l, r)| !l.is_empty() && !r.is_empty() && !r.contains(' '))
                .ok_or_else(|| VocabError::BadMergeFormat {
                    index,
                    merge: merge.clone(),
                })?;
            let missing = || VocabError::MergePieceMissing {
                index,
                merge: merge.clone(),
            };
            let l = *piece_to_id.get(left).ok_or_else(missing)?;
            let r = *piece_to_id.get(right).ok_or_else(missing)?;
            let merged = *piece_to_id
                .get(format!("{left}{right}").as_str())
                .ok_or_else(missing)?;
            merges.insert((l, r), (index as u32, merged));
        }

        let special_id = |key: &'static str, expected: &'static str| -> Result<u32, VocabError> {
            let id = meta.require_uint(key)?;
            let id = u32::try_from(id).map_err(|_| VocabError::IdOutOfRange(id))?;
            match pieces.get(id as usize) {
                Some(p) if &**p == expected => Ok(id),
                Some(p) => Err(VocabError::SpecialIdMismatch {
                    key,
                    id,
                    piece: p.to_string(),
                    expected,
                }),
                None => Err(VocabError::IdOutOfRange(id as u64)),
            }
        };
        let bos = special_id("tokenizer.ggml.bos_token_id", "<bos>")?;
        let eos = special_id("tokenizer.ggml.eos_token_id", "<eos>")?;
        let pad = special_id("tokenizer.ggml.padding_token_id", "<pad>")?;
        let unk = special_id("tokenizer.ggml.unknown_token_id", "<unk>")?;
        let eot = *piece_to_id
            .get("<turn|>")
            .ok_or_else(|| VocabError::SpecialIdMismatch {
                key: "<turn|> lookup",
                id: 0,
                piece: String::new(),
                expected: "<turn|>",
            })?;

        Ok(Self {
            pieces,
            types,
            piece_to_id,
            merges,
            byte_to_id,
            id_to_byte,
            specials,
            bos,
            eos,
            pad,
            unk,
            eot,
        })
    }

    pub fn len(&self) -> usize {
        self.pieces.len()
    }

    pub fn is_empty(&self) -> bool {
        self.pieces.is_empty()
    }

    pub fn piece(&self, id: u32) -> Option<&str> {
        self.pieces.get(id as usize).map(|p| &**p)
    }

    pub fn token_type(&self, id: u32) -> Option<TokenType> {
        self.types.get(id as usize).copied()
    }

    pub fn id_of(&self, piece: &str) -> Option<u32> {
        self.piece_to_id.get(piece).copied()
    }

    pub(crate) fn merge(&self, left: u32, right: u32) -> Option<(u32, u32)> {
        self.merges.get(&(left, right)).copied()
    }

    pub fn byte_id(&self, b: u8) -> u32 {
        self.byte_to_id[b as usize]
    }

    /// Byte value if `id` is a `<0xXX>` byte token.
    pub fn byte_of(&self, id: u32) -> Option<u8> {
        self.id_to_byte.get(&id).copied()
    }

    /// Control + user-defined tokens, longest first.
    pub(crate) fn specials(&self) -> &[(Box<str>, u32)] {
        &self.specials
    }

    pub fn bos(&self) -> u32 {
        self.bos
    }

    pub fn eos(&self) -> u32 {
        self.eos
    }

    pub fn pad(&self) -> u32 {
        self.pad
    }

    pub fn unk(&self) -> u32 {
        self.unk
    }

    /// `<turn|>` (end of turn) — generation stops on this or [`eos`](Self::eos).
    pub fn eot(&self) -> u32 {
        self.eot
    }
}

fn parse_byte_piece(piece: &str) -> Option<u8> {
    let hex = piece.strip_prefix("<0x")?.strip_suffix('>')?;
    if hex.len() != 2 {
        return None;
    }
    u8::from_str_radix(hex, 16).ok()
}
