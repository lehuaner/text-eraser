// Temporary functional verification of the PatchMatch fill (native build).
// Confirms: hole gets filled, output is finite + in [0,255], background untouched.

#[test]
fn fill_removes_hole_and_stays_finite() {
    let h = 64;
    let w = 64;
    let mut rgb = vec![0f32; h * w * 3];
    for i in 0..h * w {
        rgb[i * 3] = 200.0;
        rgb[i * 3 + 1] = 190.0;
        rgb[i * 3 + 2] = 170.0;
    }
    let mut mask = vec![0u8; h * w];
    for y in 28..36 {
        for x in 16..48 {
            mask[y * w + x] = 255;
        }
    }
    // white text bar at the hole
    for y in 28..36 {
        for x in 16..48 {
            let i = (y * w + x) * 3;
            rgb[i] = 255.0;
            rgb[i + 1] = 255.0;
            rgb[i + 2] = 255.0;
        }
    }
    let mut out = vec![0f32; h * w * 3];
    unsafe {
        textcore::patchmatch_inpaint(
            rgb.as_ptr(),
            h as i32,
            w as i32,
            mask.as_ptr(),
            std::ptr::null(),
            0,
            7,
            -1.0,
            0,
            out.as_mut_ptr(),
        );
    }
    // all finite + in range
    for v in &out {
        assert!(v.is_finite() && *v >= 0.0 && *v <= 255.0, "bad pixel {}", v);
    }
    // hole must be filled (not pure white anymore)
    let mut changed = 0;
    for y in 28..36 {
        for x in 16..48 {
            let i = (y * w + x) * 3;
            if (out[i] - 255.0).abs() > 1.0
                || (out[i + 1] - 255.0).abs() > 1.0
                || (out[i + 2] - 255.0).abs() > 1.0
            {
                changed += 1;
            }
        }
    }
    assert!(changed > 0, "hole not filled");
    // background unchanged
    let i0 = (2 * w + 2) * 3;
    assert!((out[i0] - 200.0).abs() < 1.0, "bg r changed: {}", out[i0]);
    assert!((out[i0 + 1] - 190.0).abs() < 1.0, "bg g changed");
    assert!((out[i0 + 2] - 170.0).abs() < 1.0, "bg b changed");
}

#[test]
fn fill_with_sample_mask_excludes_hole_region() {
    let h = 48;
    let w = 48;
    let mut rgb = vec![0f32; h * w * 3];
    for i in 0..h * w {
        rgb[i * 3] = 120.0;
        rgb[i * 3 + 1] = 200.0;
        rgb[i * 3 + 2] = 90.0;
    }
    let mut mask = vec![0u8; h * w];
    for y in 20..28 {
        for x in 12..36 {
            mask[y * w + x] = 255;
        }
    }
    for y in 20..28 {
        for x in 12..36 {
            let i = (y * w + x) * 3;
            rgb[i] = 255.0;
            rgb[i + 1] = 255.0;
            rgb[i + 2] = 255.0;
        }
    }
    // sample mask = everything except the hole (like _run_fill)
    let mut sm = vec![0u8; h * w];
    for i in 0..h * w {
        sm[i] = if mask[i] == 0 { 255 } else { 0 };
    }
    let mut out = vec![0f32; h * w * 3];
    unsafe {
        textcore::patchmatch_inpaint(
            rgb.as_ptr(),
            h as i32,
            w as i32,
            mask.as_ptr(),
            sm.as_ptr(),
            1,
            7,
            -1.0,
            0,
            out.as_mut_ptr(),
        );
    }
    for v in &out {
        assert!(v.is_finite() && *v >= 0.0 && *v <= 255.0, "bad pixel {}", v);
    }
    let mut changed = 0;
    for y in 20..28 {
        for x in 12..36 {
            let i = (y * w + x) * 3;
            if (out[i] - 255.0).abs() > 1.0 {
                changed += 1;
            }
        }
    }
    assert!(changed > 0, "hole not filled (sample-mask path)");
}
