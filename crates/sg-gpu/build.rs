//! Compiles WGSL shaders to SPIR-V at build time: naga-oil composition ->
//! naga validation -> naga spv-out. The runtime never compiles shaders.

use std::{env, fs, path::PathBuf};

fn main() {
    println!("cargo::rerun-if-changed=shaders");
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());

    for entry in fs::read_dir("shaders").expect("shaders/ directory") {
        let path = entry.expect("read shaders/ entry").path();
        if path.extension().and_then(|e| e.to_str()) != Some("wgsl") {
            continue;
        }
        let name = path.file_stem().unwrap().to_str().unwrap().to_owned();
        let spv = compile(&path);
        let bytes: Vec<u8> = spv.iter().flat_map(|w| w.to_le_bytes()).collect();
        fs::write(out_dir.join(format!("{name}.spv")), bytes).expect("write .spv");
    }
}

fn compile(path: &std::path::Path) -> Vec<u32> {
    let source = fs::read_to_string(path).expect("read WGSL source");
    let display = path.display().to_string();

    let mut composer = naga_oil::compose::Composer::default();
    let module = composer
        .make_naga_module(naga_oil::compose::NagaModuleDescriptor {
            source: &source,
            file_path: &display,
            ..Default::default()
        })
        .unwrap_or_else(|e| panic!("compose {display}: {e}"));

    let info = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    )
    .validate(&module)
    .unwrap_or_else(|e| panic!("validate {display}: {e:?}"));

    naga::back::spv::write_vec(&module, &info, &naga::back::spv::Options::default(), None)
        .unwrap_or_else(|e| panic!("spv-out {display}: {e}"))
}
