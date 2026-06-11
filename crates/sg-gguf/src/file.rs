//! Mmap-backed file handle that owns the bytes a [`Gguf`] borrows from.

use std::fs::File;
use std::io;
use std::path::{Path, PathBuf};

use memmap2::Mmap;

use crate::parse::{Gguf, GgufError};

/// A GGUF file mapped read-only into memory.
///
/// Owns the mapping; [`GgufFile::parse`] borrows it, so keep the `GgufFile`
/// alive as long as any tensor view.
#[derive(Debug)]
pub struct GgufFile {
    path: PathBuf,
    mmap: Mmap,
}

impl GgufFile {
    pub fn open(path: impl AsRef<Path>) -> io::Result<Self> {
        let path = path.as_ref().to_owned();
        let file = File::open(&path)?;
        // SAFETY: read-only mapping. If another process truncates the file
        // while mapped, reads fault — accepted for a local, operator-managed
        // model file (same stance as every other mmap-based GGUF loader).
        let mmap = unsafe { Mmap::map(&file)? };
        Ok(Self { path, mmap })
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn bytes(&self) -> &[u8] {
        &self.mmap
    }

    pub fn parse(&self) -> Result<Gguf<'_>, GgufError> {
        Gguf::parse(self.bytes())
    }
}
