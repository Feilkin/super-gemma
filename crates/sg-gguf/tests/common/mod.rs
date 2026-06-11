//! Minimal synthetic GGUF v3 writer for parser tests.
//!
//! Encodes just enough of the format to build valid mini-fixtures and, via
//! the raw escape hatches (`kv_raw`, `tensor_raw`, byte surgery on the
//! output), malformed variants. Independent of the parser under test.

#![allow(dead_code)] // each test binary uses a subset of the builder

pub const MAGIC: u32 = 0x4655_4747;

pub fn w_u32(out: &mut Vec<u8>, v: u32) {
    out.extend_from_slice(&v.to_le_bytes());
}

pub fn w_u64(out: &mut Vec<u8>, v: u64) {
    out.extend_from_slice(&v.to_le_bytes());
}

pub fn w_string(out: &mut Vec<u8>, s: &str) {
    w_u64(out, s.len() as u64);
    out.extend_from_slice(s.as_bytes());
}

struct TensorSpec {
    name: String,
    dims: Vec<u64>,
    raw_ty: u32,
    /// Explicit data-section offset; `None` = packed sequentially (aligned).
    offset: Option<u64>,
    data: Vec<u8>,
}

pub struct Builder {
    version: u32,
    alignment: u32,
    kv_count: u64,
    kvs: Vec<u8>,
    tensors: Vec<TensorSpec>,
}

impl Default for Builder {
    fn default() -> Self {
        Self::new()
    }
}

impl Builder {
    pub fn new() -> Self {
        Self {
            version: 3,
            alignment: 32,
            kv_count: 0,
            kvs: Vec::new(),
            tensors: Vec::new(),
        }
    }

    pub fn version(mut self, v: u32) -> Self {
        self.version = v;
        self
    }

    /// Emit a `general.alignment` KV and lay the data section out with it.
    pub fn alignment(mut self, a: u32) -> Self {
        self.alignment = a;
        self.kv_u32("general.alignment", a)
    }

    /// Append a KV with an arbitrary type id and pre-encoded payload.
    pub fn kv_raw(mut self, key: &str, raw_ty: u32, payload: &[u8]) -> Self {
        w_string(&mut self.kvs, key);
        w_u32(&mut self.kvs, raw_ty);
        self.kvs.extend_from_slice(payload);
        self.kv_count += 1;
        self
    }

    pub fn kv_u8(self, key: &str, v: u8) -> Self {
        self.kv_raw(key, 0, &[v])
    }

    pub fn kv_i8(self, key: &str, v: i8) -> Self {
        self.kv_raw(key, 1, &v.to_le_bytes())
    }

    pub fn kv_u16(self, key: &str, v: u16) -> Self {
        self.kv_raw(key, 2, &v.to_le_bytes())
    }

    pub fn kv_i16(self, key: &str, v: i16) -> Self {
        self.kv_raw(key, 3, &v.to_le_bytes())
    }

    pub fn kv_u32(self, key: &str, v: u32) -> Self {
        self.kv_raw(key, 4, &v.to_le_bytes())
    }

    pub fn kv_i32(self, key: &str, v: i32) -> Self {
        self.kv_raw(key, 5, &v.to_le_bytes())
    }

    pub fn kv_f32(self, key: &str, v: f32) -> Self {
        self.kv_raw(key, 6, &v.to_le_bytes())
    }

    pub fn kv_bool(self, key: &str, v: bool) -> Self {
        self.kv_raw(key, 7, &[u8::from(v)])
    }

    pub fn kv_str(self, key: &str, v: &str) -> Self {
        let mut p = Vec::new();
        w_string(&mut p, v);
        self.kv_raw(key, 8, &p)
    }

    pub fn kv_u64(self, key: &str, v: u64) -> Self {
        self.kv_raw(key, 10, &v.to_le_bytes())
    }

    pub fn kv_i64(self, key: &str, v: i64) -> Self {
        self.kv_raw(key, 11, &v.to_le_bytes())
    }

    pub fn kv_f64(self, key: &str, v: f64) -> Self {
        self.kv_raw(key, 12, &v.to_le_bytes())
    }

    fn kv_prim_array<const N: usize>(self, key: &str, elem_ty: u32, elems: &[[u8; N]]) -> Self {
        let mut p = Vec::new();
        w_u32(&mut p, elem_ty);
        w_u64(&mut p, elems.len() as u64);
        for e in elems {
            p.extend_from_slice(e);
        }
        self.kv_raw(key, 9, &p)
    }

    pub fn kv_u8_array(self, key: &str, vs: &[u8]) -> Self {
        let elems: Vec<[u8; 1]> = vs.iter().map(|v| [*v]).collect();
        self.kv_prim_array(key, 0, &elems)
    }

    pub fn kv_u16_array(self, key: &str, vs: &[u16]) -> Self {
        let elems: Vec<[u8; 2]> = vs.iter().map(|v| v.to_le_bytes()).collect();
        self.kv_prim_array(key, 2, &elems)
    }

    pub fn kv_i32_array(self, key: &str, vs: &[i32]) -> Self {
        let elems: Vec<[u8; 4]> = vs.iter().map(|v| v.to_le_bytes()).collect();
        self.kv_prim_array(key, 5, &elems)
    }

    pub fn kv_f32_array(self, key: &str, vs: &[f32]) -> Self {
        let elems: Vec<[u8; 4]> = vs.iter().map(|v| v.to_le_bytes()).collect();
        self.kv_prim_array(key, 6, &elems)
    }

    pub fn kv_bool_array(self, key: &str, vs: &[bool]) -> Self {
        let elems: Vec<[u8; 1]> = vs.iter().map(|v| [u8::from(*v)]).collect();
        self.kv_prim_array(key, 7, &elems)
    }

    pub fn kv_str_array(self, key: &str, vs: &[&str]) -> Self {
        let mut p = Vec::new();
        w_u32(&mut p, 8);
        w_u64(&mut p, vs.len() as u64);
        for v in vs {
            w_string(&mut p, v);
        }
        self.kv_raw(key, 9, &p)
    }

    pub fn kv_i64_array(self, key: &str, vs: &[i64]) -> Self {
        let elems: Vec<[u8; 8]> = vs.iter().map(|v| v.to_le_bytes()).collect();
        self.kv_prim_array(key, 11, &elems)
    }

    pub fn kv_f64_array(self, key: &str, vs: &[f64]) -> Self {
        let elems: Vec<[u8; 8]> = vs.iter().map(|v| v.to_le_bytes()).collect();
        self.kv_prim_array(key, 12, &elems)
    }

    /// Array of u32 arrays (exercises nested-array parsing).
    pub fn kv_nested_u32_array(self, key: &str, vss: &[&[u32]]) -> Self {
        let mut p = Vec::new();
        w_u32(&mut p, 9); // outer elem type: array
        w_u64(&mut p, vss.len() as u64);
        for vs in vss {
            w_u32(&mut p, 4); // inner elem type: u32
            w_u64(&mut p, vs.len() as u64);
            for v in *vs {
                w_u32(&mut p, *v);
            }
        }
        self.kv_raw(key, 9, &p)
    }

    /// Tensor packed sequentially in declaration order.
    pub fn tensor(self, name: &str, dims: &[u64], raw_ty: u32, data: &[u8]) -> Self {
        self.tensor_inner(name, dims, raw_ty, None, data)
    }

    /// Tensor with an explicit (possibly bogus) data-section offset.
    pub fn tensor_raw(
        self,
        name: &str,
        dims: &[u64],
        raw_ty: u32,
        offset: u64,
        data: &[u8],
    ) -> Self {
        self.tensor_inner(name, dims, raw_ty, Some(offset), data)
    }

    fn tensor_inner(
        mut self,
        name: &str,
        dims: &[u64],
        raw_ty: u32,
        offset: Option<u64>,
        data: &[u8],
    ) -> Self {
        self.tensors.push(TensorSpec {
            name: name.to_owned(),
            dims: dims.to_vec(),
            raw_ty,
            offset,
            data: data.to_vec(),
        });
        self
    }

    pub fn build(self) -> Vec<u8> {
        let align = self.alignment as u64;
        let align_up = |v: u64| v.div_ceil(align) * align;

        // Assign packed offsets where not explicitly overridden.
        let mut next = 0u64;
        let offsets: Vec<u64> = self
            .tensors
            .iter()
            .map(|t| {
                let off = t.offset.unwrap_or(next);
                next = align_up(off + t.data.len() as u64);
                off
            })
            .collect();

        let mut out = Vec::new();
        w_u32(&mut out, MAGIC);
        w_u32(&mut out, self.version);
        w_u64(&mut out, self.tensors.len() as u64);
        w_u64(&mut out, self.kv_count);
        out.extend_from_slice(&self.kvs);

        for (t, &off) in self.tensors.iter().zip(&offsets) {
            w_string(&mut out, &t.name);
            w_u32(&mut out, t.dims.len() as u32);
            for d in &t.dims {
                w_u64(&mut out, *d);
            }
            w_u32(&mut out, t.raw_ty);
            w_u64(&mut out, off);
        }

        if !self.tensors.is_empty() {
            let data_start = align_up(out.len() as u64) as usize;
            let data_len = self
                .tensors
                .iter()
                .zip(&offsets)
                .map(|(t, &off)| off as usize + t.data.len())
                .max()
                .unwrap_or(0);
            out.resize(data_start + data_len, 0);
            for (t, &off) in self.tensors.iter().zip(&offsets) {
                out[data_start + off as usize..][..t.data.len()].copy_from_slice(&t.data);
            }
        }
        out
    }
}
