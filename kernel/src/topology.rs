/// Pinned subdivision level.
pub const SUBDIVISIONS: u32 = 4;

/// Vertex count at `SUBDIVISIONS`, that is 10 * 4^n + 2.
pub const N_VERTS: usize = 2562;

/// Face count at `SUBDIVISIONS`, that is 20 * 4^n.
pub const N_FACES: usize = 5120;

/// Build the blank, the unit sphere vertices as flat xyz triples split into f32
/// halves, every high half first and then every low half, and the faces as
/// flat index triples. Host only, since it needs dynamic maps and runs once per
/// process rather than once per nonce.
#[cfg(not(any(target_arch = "nvptx64", target_arch = "amdgpu")))]
pub fn build() -> (alloc::vec::Vec<f32>, alloc::vec::Vec<u32>) {
    use alloc::collections::BTreeMap;
    use alloc::vec;
    use alloc::vec::Vec;

    use crate::split::Split;

    let t = (1.0 + libm::sqrt(5.0)) / 2.0;
    let mut vertices: Vec<[f64; 3]> = vec![
        [-1.0, t, 0.0], [1.0, t, 0.0], [-1.0, -t, 0.0], [1.0, -t, 0.0],
        [0.0, -1.0, t], [0.0, 1.0, t], [0.0, -1.0, -t], [0.0, 1.0, -t],
        [t, 0.0, -1.0], [t, 0.0, 1.0], [-t, 0.0, -1.0], [-t, 0.0, 1.0],
    ];
    for v in &mut vertices {
        *v = unit(*v);
    }

    let mut faces: Vec<[u32; 3]> = vec![
        [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
        [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
        [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
        [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1],
    ];

    for _ in 0..SUBDIVISIONS {
        let mut midpoint: BTreeMap<(u32, u32), u32> = BTreeMap::new();
        let mut new_faces = Vec::with_capacity(faces.len() * 4);

        let mut get_mid = |a: u32, b: u32, verts: &mut Vec<[f64; 3]>| -> u32 {
            let key = if a < b { (a, b) } else { (b, a) };
            if let Some(&idx) = midpoint.get(&key) {
                return idx;
            }
            let va = verts[a as usize];
            let vb = verts[b as usize];
            let mid = unit([
                (va[0] + vb[0]) * 0.5,
                (va[1] + vb[1]) * 0.5,
                (va[2] + vb[2]) * 0.5,
            ]);
            let idx = verts.len() as u32;
            verts.push(mid);
            midpoint.insert(key, idx);
            idx
        };

        for f in &faces {
            let a = get_mid(f[0], f[1], &mut vertices);
            let b = get_mid(f[1], f[2], &mut vertices);
            let c = get_mid(f[2], f[0], &mut vertices);
            new_faces.push([f[0], a, c]);
            new_faces.push([f[1], b, a]);
            new_faces.push([f[2], c, b]);
            new_faces.push([a, b, c]);
        }
        faces = new_faces;
    }

    debug_assert_eq!(vertices.len(), N_VERTS);
    debug_assert_eq!(faces.len(), N_FACES);

    let coords = || vertices.iter().flat_map(|v| v.iter().map(|&x| Split::new(x)));
    let base = coords().map(|s| s.hi).chain(coords().map(|s| s.lo)).collect();
    let flat_faces = faces.iter().flat_map(|f| f.iter().copied()).collect();
    (base, flat_faces)
}

#[cfg(not(any(target_arch = "nvptx64", target_arch = "amdgpu")))]
fn unit(v: [f64; 3]) -> [f64; 3] {
    let len = libm::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
    [v[0] / len, v[1] / len, v[2] / len]
}
