//! numpy↔Rust 填充孪生校验: 读取 data/_pmgen.py 生成的 case{N}.bin,
//! 用与 wasm 完全相同的 patchmatch_inpaint 原生跑一遍, 写出 case{N}_rust.bin。
//! 由 data/_pmcmp.py 做逐字节对比。
//!
//! 运行: cd shared && cargo test --release --test pm_parity
use std::io::Read;
use textcore::patchmatch_inpaint;

fn read_case(path: &str) -> (i32, i32, i32, f32, u32, Vec<f32>, Vec<u8>, Vec<u8>) {
    let mut buf = Vec::new();
    std::fs::File::open(path).unwrap().read_to_end(&mut buf).unwrap();
    let h = i32::from_le_bytes(buf[0..4].try_into().unwrap());
    let w = i32::from_le_bytes(buf[4..8].try_into().unwrap());
    let hs = i32::from_le_bytes(buf[8..12].try_into().unwrap());
    let dir = f32::from_le_bytes(buf[12..16].try_into().unwrap());
    let seed = u32::from_le_bytes(buf[16..20].try_into().unwrap());
    let n = (h * w) as usize;
    let mut off = 20;
    let sub: Vec<f32> = buf[off..off + n * 12]
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    off += n * 12;
    let mut subm = vec![0u8; n];
    subm.copy_from_slice(&buf[off..off + n]);
    off += n;
    let mut subsm = vec![0u8; n];
    if hs != 0 {
        subsm.copy_from_slice(&buf[off..off + n]);
    }
    (h, w, hs, dir, seed, sub, subm, subsm)
}

#[test]
fn parity_cases() {
    let dir = std::path::Path::new("../data/_pmparity");
    let mut i = 0;
    while dir.join(format!("case{i}.bin")).exists() {
        let (h, w, hs, d, seed, sub, subm, subsm) =
            read_case(&dir.join(format!("case{i}.bin")).to_string_lossy());
        let n = (h * w) as usize;
        let mut out = vec![0f32; n * 3];
        unsafe {
            patchmatch_inpaint(
                sub.as_ptr(), h, w, subm.as_ptr(),
                if hs != 0 { subsm.as_ptr() } else { subm.as_ptr() },
                hs, 7, d, seed, out.as_mut_ptr(),
            );
        }
        let bytes: Vec<u8> = out.iter().flat_map(|v| v.to_le_bytes()).collect();
        std::fs::write(dir.join(format!("case{i}_rust.bin")), bytes).unwrap();
        println!("case{i}: h={h} w={w} hs={hs} dir={d} -> written");
        i += 1;
    }
    if i == 0 {
        // 样本由 data/_pmgen.py 生成(data/ 不入库); 缺失时跳过而非失败
        println!("no parity cases found; run data/_pmgen.py first");
        return;
    }
}
