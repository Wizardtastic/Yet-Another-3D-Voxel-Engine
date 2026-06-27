//! First-person camera + frustum for culling. Lives in `voxel-core` because both
//! the renderer (culling) and gameplay (look direction, ray origin) need it.

use glam::{Mat4, Vec3, Vec4};

/// First-person view camera. Yaw/pitch in radians; position in world space.
#[derive(Clone, Copy, Debug)]
pub struct Camera {
    pub pos: Vec3,
    pub yaw: f32,
    pub pitch: f32,
    /// Vertical field of view in radians.
    pub fov_y: f32,
    /// Aspect ratio (width / height).
    pub aspect: f32,
    /// Near and far clip plane distances (metres).
    pub near: f32,
    pub far: f32,
}

impl Default for Camera {
    fn default() -> Self {
        Self {
            pos: Vec3::ZERO,
            yaw: 0.0,
            pitch: 0.0,
            fov_y: std::f32::consts::FRAC_PI_3, // 60°
            aspect: 16.0 / 9.0,
            near: 0.05,
            far: 1024.0,
        }
    }
}

impl Camera {
    /// Unit forward vector derived from yaw/pitch (right-handed, -Z forward).
    #[inline]
    pub fn forward(self) -> Vec3 {
        let (cp, sp) = (self.pitch.cos(), self.pitch.sin());
        let (cy, sy) = (self.yaw.cos(), self.yaw.sin());
        Vec3::new(-cp * sy, sp, -cp * cy)
    }

    /// Unit right vector (perpendicular to forward, lying in the XZ plane).
    #[inline]
    pub fn right(self) -> Vec3 {
        let cy = self.yaw.cos();
        let sy = self.yaw.sin();
        Vec3::new(cy, 0.0, -sy).normalize()
    }

    /// World-space view direction with pitch clamped to horizontal (for movement).
    #[inline]
    pub fn forward_flat(self) -> Vec3 {
        let sy = self.yaw.sin();
        let cy = self.yaw.cos();
        Vec3::new(-sy, 0.0, -cy).normalize()
    }

    /// Look-at view matrix (world -> view space).
    pub fn view(self) -> Mat4 {
        // Clamp pitch to avoid gimbal lock (NaN from cross product when
        // forward is parallel to the up vector).
        let pitch = self.pitch.clamp(
            -std::f32::consts::FRAC_PI_2 + 0.001,
            std::f32::consts::FRAC_PI_2 - 0.001,
        );
        let forward = Vec3::new(
            -pitch.cos() * self.yaw.sin(),
            pitch.sin(),
            -pitch.cos() * self.yaw.cos(),
        );
        let target = self.pos + forward;
        Mat4::look_at_rh(self.pos, target, Vec3::Y)
    }

    /// Perspective projection matrix (view -> clip space).
    pub fn projection(self) -> Mat4 {
        Mat4::perspective_rh(self.fov_y, self.aspect, self.near, self.far)
    }

    /// Combined view-projection matrix.
    pub fn view_projection(self) -> Mat4 {
        self.projection() * self.view()
    }
}

/// A 6-plane frustum extracted from a view-projection matrix. Used for
/// per-chunk culling on the CPU before issuing draw calls.
#[derive(Clone, Copy, Debug)]
pub struct Frustum {
    /// Left, Right, Bottom, Top, Near, Far — each as `a*x + b*y + c*z + d >= 0`
    /// is "inside".
    pub planes: [Vec4; 6],
}

impl Frustum {
    /// Extract the six clip-space planes from a row-major view-projection.
    /// Convention: a point is inside when all six plane equations are >= 0.
    pub fn from_view_projection(vp: Mat4) -> Self {
        // Mat4 is column-major in glam; index via `col(i)` returns Vec4 columns.
        // m[i][j] = column i, row j. The combined matrix element M(row r, col c)
        // is `vp.col(c)[r]`.
        let col0 = vp.col(0);
        let col1 = vp.col(1);
        let col2 = vp.col(2);
        let col3 = vp.col(3);

        // Planes as (row3 +/- rowN), normalised later.
        // Left   = row4 + row1
        // Right  = row4 - row1
        // Bottom = row4 + row2
        // Top    = row4 - row2
        // Near   = row4 + row3
        // Far    = row4 - row3
        // where rowN = (col0[N], col1[N], col2[N], col3[N]) and row4 = (col3[3]...)
        // i.e. the w-row. We construct each plane vector and normalise.
        let rows = |r: usize| Vec4::new(col0[r], col1[r], col2[r], col3[r]);
        let r4 = rows(3);
        let r1 = rows(0);
        let r2 = rows(1);
        let r3 = rows(2);

        let mut planes = [
            r4 + r1, // left
            r4 - r1, // right
            r4 + r2, // bottom
            r4 - r2, // top
            r4 + r3, // near
            r4 - r3, // far
        ];
        for p in planes.iter_mut() {
            let len = Vec3::new(p.x, p.y, p.z).length();
            if len > 0.0 {
                *p /= len;
            }
        }
        Self { planes }
    }

    /// Test an axis-aligned bounding box (world space) against the frustum.
    /// Returns `true` if any part of the box is potentially visible.
    /// Uses the positive-vertex technique: for each plane, find the AABB corner
    /// most aligned with the plane normal (single O(1) test per plane).
    pub fn intersects_aabb(&self, min: Vec3, max: Vec3) -> bool {
        for p in &self.planes {
            // Select the corner most aligned with the plane normal.
            let px = if p.x >= 0.0 { max.x } else { min.x };
            let py = if p.y >= 0.0 { max.y } else { min.y };
            let pz = if p.z >= 0.0 { max.z } else { min.z };
            if p.x * px + p.y * py + p.z * pz + p.w < 0.0 {
                return false;
            }
        }
        true
    }
}
