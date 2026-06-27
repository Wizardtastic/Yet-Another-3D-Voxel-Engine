//! Background chunk streaming: generate, mesh, load and unload chunks around a
//! moving focus point (the player). All heavy work runs off the render thread
//! via a dedicated worker thread that dispatches `rayon` parallel batches for
//! generation and meshing.

use std::collections::HashMap;
use std::sync::Arc;

use anyhow::{Context, Result};
use flume::{unbounded, Receiver, Sender};
use rayon::prelude::*;

use glam::Vec3;
use voxel_core::{
    math::{block_to_chunk, chunk_origin, world_to_block, ChunkPos},
    Frustum, CHUNK_SIZE,
};

use crate::{
    chunk::Chunk,
    gen::TerrainGenerator,
    mesh::{ChunkMeshBundle, ChunkMesher},
    registry::BlockRegistry,
    world::World,
};

/// Tunable streaming parameters (in chunks).
#[derive(Clone, Copy, Debug)]
pub struct StreamConfig {
    /// Horizontal radius (in chunks) around the focus that must be loaded.
    pub load_radius: i32,
    /// Chunks beyond this radius are eligible for unload.
    pub unload_radius: i32,
    /// Vertical half-band (in chunks) around the focus Y.
    pub vertical_half: i32,
    /// Max chunks to generate per tick (keeps frame hitches bounded).
    pub gen_batch: usize,
    /// Max chunks to mesh per tick.
    pub mesh_batch: usize,
    /// Max chunks to structure (place cross-chunk features) per tick.
    pub structure_batch: usize,
}

impl Default for StreamConfig {
    fn default() -> Self {
        Self {
            load_radius: 6,
            unload_radius: 8,
            vertical_half: 3,
            gen_batch: 24,
            mesh_batch: 24,
            structure_batch: 8,
        }
    }
}

/// Events the worker emits to the main thread. The renderer consumes
/// `MeshReady`/`Unloaded` to upload/free GPU buffers.
#[derive(Clone, Debug)]
pub enum ChunkStreamEvent {
    /// A chunk finished generating and was inserted into the world.
    Generated(ChunkPos),
    /// A chunk's mesh was (re)built and is ready to upload.
    MeshReady {
        pos: ChunkPos,
        bundle: ChunkMeshBundle,
    },
    /// A chunk left the loaded set; free its GPU resources.
    Unloaded(ChunkPos),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum State {
    Generating,
    Generated,
    Structuring,
    Structured,
    Meshing,
    Meshed,
}

enum Cmd {
    Focus(Vec3),
    SunDir(Vec3),
    Frustum(Frustum),
    LoadRadius(u32),
    Remesh(ChunkPos),
    Shutdown,
}

/// Owns the worker thread. Drop shuts it down.
pub struct ChunkStreamer {
    cmd_tx: Sender<Cmd>,
    pub events: Receiver<ChunkStreamEvent>,
    handle: Option<std::thread::JoinHandle<()>>,
}

impl ChunkStreamer {
    /// Spawn a streamer attached to `world`. The world must outlive the streamer;
    /// the streamer holds an `Arc` clone so dropping the streamer stops touching it.
    pub fn spawn(world: Arc<World>, config: StreamConfig) -> Result<Self> {
        let (cmd_tx, cmd_rx) = unbounded();
        let (event_tx, event_rx) = unbounded();
        let handle = std::thread::Builder::new()
            .name("voxel-stream".into())
            .spawn(move || {
                run_worker(world, config, cmd_rx, event_tx);
            })
            .context("spawn stream worker thread")?;
        Ok(Self {
            cmd_tx,
            events: event_rx,
            handle: Some(handle),
        })
    }

    /// Update the streaming focus (player position in world space).
    pub fn set_focus(&self, pos: Vec3) {
        let _ = self.cmd_tx.send(Cmd::Focus(pos));
    }

    /// Update the sun direction for lighting computation.
    pub fn set_sun_dir(&self, dir: Vec3) {
        let _ = self.cmd_tx.send(Cmd::SunDir(dir));
    }

    /// Update the view frustum for mesh-building culling.
    pub fn set_frustum(&self, frustum: Frustum) {
        let _ = self.cmd_tx.send(Cmd::Frustum(frustum));
    }

    /// Change the load radius at runtime.
    pub fn set_load_radius(&self, radius: u32) {
        let _ = self.cmd_tx.send(Cmd::LoadRadius(radius));
    }

    /// Request a remesh of a chunk (after a gameplay edit).
    pub fn request_remesh(&self, pos: ChunkPos) {
        let _ = self.cmd_tx.send(Cmd::Remesh(pos));
    }

    /// Drain any pending events without blocking.
    pub fn poll_events(&self) -> Vec<ChunkStreamEvent> {
        self.events.drain().collect()
    }
}

impl Drop for ChunkStreamer {
    fn drop(&mut self) {
        let _ = self.cmd_tx.send(Cmd::Shutdown);
        if let Some(h) = self.handle.take() {
            let _ = h.join();
        }
    }
}

fn run_worker(
    world: Arc<World>,
    config: StreamConfig,
    cmd_rx: Receiver<Cmd>,
    event_tx: Sender<ChunkStreamEvent>,
) {
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(num_threads())
        .build()
        .expect("rayon pool");
    let mesher = ChunkMesher;
    let gen: Arc<TerrainGenerator> = world.terrain();
    let reg: Arc<BlockRegistry> = world.registry();

    let mut focus = Vec3::new(0.0, 80.0, 0.0);
    let mut sun_dir = Vec3::new(0.3, 0.9, 0.1).normalize();
    let mut frustum = None::<Frustum>;
    let mut load_radius = config.load_radius;
    let mut state: HashMap<ChunkPos, State> = HashMap::new();

    loop {
        // --- drain commands ---
        let mut remesh_requests: Vec<ChunkPos> = Vec::new();
        let mut focus_changed = false;
        for cmd in cmd_rx.drain() {
            match cmd {
                Cmd::Focus(p) => {
                    focus = p;
                    focus_changed = true;
                }
                Cmd::SunDir(d) => {
                    sun_dir = d;
                }
                Cmd::Frustum(f) => {
                    frustum = Some(f);
                }
                Cmd::LoadRadius(r) => {
                    load_radius = r as i32;
                    log::info!("load radius changed to {load_radius}");
                }
                Cmd::Remesh(pos) => remesh_requests.push(pos),
                Cmd::Shutdown => return,
            }
        }

        // Desired loaded set (cylindrical band around focus).
        let focus_chunk = block_to_chunk(world_to_block(focus));
        let desired: Vec<ChunkPos> = desired_set(focus_chunk, load_radius, config.vertical_half);

        // Unload chunks outside the unload radius.
        let unload_r2 = (config.unload_radius + 1) as i64 * (config.unload_radius + 1) as i64;
        let to_unload: Vec<ChunkPos> = state
            .keys()
            .filter(|p| {
                let dx = (p.x() - focus_chunk.x()) as i64;
                let dz = (p.z() - focus_chunk.z()) as i64;
                let dy = (p.y() - focus_chunk.y()) as i64;
                dx * dx + dz * dz > unload_r2 || dy.abs() > config.vertical_half as i64 + 1
            })
            .copied()
            .collect();
        for p in to_unload {
            state.remove(&p);
            world.remove_chunk(p);
            let _ = event_tx.send(ChunkStreamEvent::Unloaded(p));
        }

        // Honour direct remesh requests (e.g. from player edits).
        // Track which chunks need priority remesh (user-triggered edits).
        let mut priority_remesh_set: std::collections::HashSet<ChunkPos> =
            std::collections::HashSet::new();

        for p in remesh_requests.drain(..) {
            if let Some(s) = state.get_mut(&p) {
                if *s == State::Meshed || *s == State::Meshing {
                    *s = State::Structured; // force re-mesh without re-structuring
                    priority_remesh_set.insert(p);
                }
            }
            // Also remesh the 6 neighbours, since border faces may change.
            for n in neighbours(p) {
                if let Some(s) = state.get_mut(&n) {
                    if *s == State::Meshed {
                        *s = State::Structured;
                        priority_remesh_set.insert(n);
                    }
                }
            }
        }

        // --- PRIORITY remesh phase: immediately mesh user-triggered remeshes
        // (block break/place) before the regular distance-sorted queue. ---
        if !priority_remesh_set.is_empty() {
            let priority_remesh: Vec<ChunkPos> = priority_remesh_set.into_iter().collect();
            let world = world.clone();
            let reg = reg.clone();
            let sun_dir_for_mesh = sun_dir;
            for pos in &priority_remesh {
                state.insert(*pos, State::Meshing);
            }
            let meshes: Vec<(ChunkPos, ChunkMeshBundle)> = pool.install(|| {
                priority_remesh
                    .par_iter()
                    .map(|&pos| {
                        let bundle = world.with_chunk_for_mesh(pos, |chunk, sample, sample_water, sample_loaded| {
                            mesher.build(chunk, &reg, sample, sample_water, sample_loaded, sun_dir_for_mesh)
                        });
                        (pos, bundle)
                    })
                    .collect()
            });
            for (pos, bundle) in meshes {
                world.insert_mesh(pos, bundle.clone());
                state.insert(pos, State::Meshed);
                let _ = event_tx.send(ChunkStreamEvent::MeshReady { pos, bundle });
            }
        }

        // --- generation phase ---
        let gen_todo: Vec<ChunkPos> = desired
            .iter()
            .filter(|p| !state.contains_key(p))
            .copied()
            .collect();
        let gen_todo = sort_by_distance(gen_todo, focus_chunk);
        let gen_batch: Vec<ChunkPos> = gen_todo
            .into_iter()
            .take(config.gen_batch)
            .inspect(|p| {
                state.insert(*p, State::Generating);
            })
            .collect();

        if !gen_batch.is_empty() {
            let gen = gen.clone();
            let reg = reg.clone();
            let world_for_light = world.clone();
            let generated: Vec<(ChunkPos, Chunk)> = pool.install(|| {
                gen_batch
                    .par_iter()
                    .map(|&pos| {
                        let mut chunk = Chunk::new(pos);
                        gen.generate(&mut chunk, &reg);
                        gen.decorate(&mut chunk, &reg, |wx, wy, wz| world_for_light.get_block(wx, wy, wz));
                        // Compute lighting: ray-based sunlight + torchlight BFS.
                        let mut cross_updates = Vec::new();
                        crate::light::compute_all(
                            &mut chunk,
                            &reg,
                            sun_dir,
                            // sample_block: for cross-chunk block lookups
                            &|wx, wy, wz| world_for_light.get_block(wx, wy, wz),
                            // sample_torchlight: for cross-chunk torchlight
                            &|wx, wy, wz| world_for_light.get_torchlight_world(wx, wy, wz),
                            // cross_chunk_update: collect pending torchlight updates
                            &mut |pos, level| cross_updates.push((pos, level)),
                        );
                        // Apply cross-chunk torchlight updates to the world.
                        for (pos, level) in cross_updates {
                            world_for_light.set_torchlight_world(pos.0.x, pos.0.y, pos.0.z, level);
                        }
                        (pos, chunk)
                    })
                    .collect()
            });
            for (pos, chunk) in generated {
                world.insert_chunk(pos, chunk);
                state.insert(pos, State::Generated);
                let _ = event_tx.send(ChunkStreamEvent::Generated(pos));
                // Remesh loaded neighbours so their border faces and sunlight
                // are updated now that this chunk exists. They don't need to
                // re-run structures — their blocks haven't changed.
                for n in neighbours(pos) {
                    if let Some(s) = state.get_mut(&n) {
                        if *s == State::Meshed {
                            *s = State::Structured;
                        }
                    }
                }
            }
        }

        // --- structure phase: place cross-chunk features (dungeons, towers,
        // wells) on chunks that have finished generating. A chunk can only be
        // structured when its 8 horizontal neighbours are also ready
        // (Generated / Structuring / Structured / Meshed), or are outside the
        // desired loaded set (edge of load radius). This guarantees the
        // `sample` closure reads real neighbour data when verifying structure
        // placement conditions (e.g. "is the surface block grass?"). ---
        let desired_set: std::collections::HashSet<ChunkPos> =
            desired.iter().copied().collect();
        let struct_todo: Vec<ChunkPos> = desired
            .iter()
            .filter(|p| matches!(state.get(p), Some(State::Generated)))
            .filter(|p| {
                horizontal_neighbours(**p).iter().all(|n| match state.get(n) {
                    Some(State::Generated)
                    | Some(State::Structuring)
                    | Some(State::Structured)
                    | Some(State::Meshed) => true,
                    _ => !desired_set.contains(n),
                })
            })
            .copied()
            .collect();
        let struct_todo = sort_by_distance(struct_todo, focus_chunk);
        let struct_batch: Vec<ChunkPos> = struct_todo
            .into_iter()
            .take(config.structure_batch)
            .inspect(|p| {
                state.insert(*p, State::Structuring);
            })
            .collect();

        if !struct_batch.is_empty() {
            let gen = gen.clone();
            let reg = reg.clone();
            let world_for_sample = world.clone();
            let structured: Vec<(ChunkPos, Chunk)> = pool.install(|| {
                struct_batch
                    .par_iter()
                    .map(|&pos| {
                        // Clone the chunk out from under the lock so the
                        // worker can mutate it while the `sample` closure
                        // re-acquires a read lock for cross-chunk queries.
                        let chunk_clone = {
                            let chunks = world_for_sample.chunks_ref().read();
                            chunks.get(&pos).cloned()
                        };
                        let mut chunk = chunk_clone.unwrap_or_else(|| Chunk::new(pos));
                        gen.place_structures(
                            &mut chunk,
                            &reg,
                            &|x, y, z| world_for_sample.get_block(x, y, z),
                        );
                        chunk.dirty = true;
                        chunk.light_dirty = true;
                        (pos, chunk)
                    })
                    .collect()
            });
            for (pos, chunk) in structured {
                world.insert_chunk(pos, chunk);
                state.insert(pos, State::Structured);
                let _ = event_tx.send(ChunkStreamEvent::Generated(pos));
                // Demote Meshed neighbours so their border faces get rebuilt
                // (a structure may have spilled across the boundary).
                for n in neighbours(pos) {
                    if let Some(s) = state.get_mut(&n) {
                        if *s == State::Meshed {
                            *s = State::Structured;
                        }
                    }
                }
            }
        }

        // --- meshing phase: chunks that are structured and have structured
        // neighbours (so border face culling is correct enough).
        // If a frustum is available, skip chunks entirely outside it. ---
        let mesh_todo: Vec<ChunkPos> = desired
            .iter()
            .filter(|p| matches!(state.get(p), Some(State::Structured)))
            .copied()
            .filter(|p| {
                if let Some(ref f) = frustum {
                    let origin = chunk_origin(*p);
                    let min = origin.as_vec3();
                    let max = origin.as_vec3() + Vec3::splat(CHUNK_SIZE as f32);
                    f.intersects_aabb(min, max)
                } else {
                    true
                }
            })
            .collect();
        let mesh_todo = sort_by_distance(mesh_todo, focus_chunk);
        let mesh_batch: Vec<ChunkPos> = mesh_todo
            .into_iter()
            .take(config.mesh_batch)
            .inspect(|p| {
                state.insert(*p, State::Meshing);
            })
            .collect();

        if !mesh_batch.is_empty() {
            let world = world.clone();
            let reg = reg.clone();
            let mesher = &mesher;
            let sun_dir_for_mesh = sun_dir;
            let meshes: Vec<(ChunkPos, ChunkMeshBundle)> = pool.install(|| {
                mesh_batch
                    .par_iter()
                    .map(|&pos| {
                        let bundle = world.with_chunk_for_mesh(pos, |chunk, sample, sample_water, sample_loaded| {
                            mesher.build(chunk, &reg, sample, sample_water, sample_loaded, sun_dir_for_mesh)
                        });
                        (pos, bundle)
                    })
                    .collect()
            });
            for (pos, bundle) in meshes {
                world.insert_mesh(pos, bundle.clone());
                state.insert(pos, State::Meshed);
                let _ = event_tx.send(ChunkStreamEvent::MeshReady { pos, bundle });
            }
        }

        // If nothing happened this iteration and no commands are pending, idle
        // briefly to avoid a busy loop.
        if gen_batch.is_empty() && struct_batch.is_empty() && mesh_batch.is_empty() && !focus_changed
        {
            std::thread::sleep(std::time::Duration::from_millis(2));
        }
    }
}

fn num_threads() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4)
        .max(2)
}

/// The cylindrical band of chunks around `focus` that should be loaded.
fn desired_set(focus: ChunkPos, load_radius: i32, vertical_half: i32) -> Vec<ChunkPos> {
    let mut out = Vec::new();
    let r = load_radius;
    for dx in -r..=r {
        for dz in -r..=r {
            if dx * dx + dz * dz > r * r {
                continue;
            }
            for dy in -vertical_half..=vertical_half {
                let p = ChunkPos::new(focus.x() + dx, focus.y() + dy, focus.z() + dz);
                if p.y() < 0 || p.y() >= voxel_core::MAX_CHUNK_Y {
                    continue;
                }
                out.push(p);
            }
        }
    }
    out
}

/// The 8 XZ-plane neighbours of a chunk (used for the structure phase's
/// readiness gate — vertical neighbours are not required).
fn horizontal_neighbours(p: ChunkPos) -> [ChunkPos; 8] {
    [
        ChunkPos::new(p.x() - 1, p.y(), p.z() - 1),
        ChunkPos::new(p.x(), p.y(), p.z() - 1),
        ChunkPos::new(p.x() + 1, p.y(), p.z() - 1),
        ChunkPos::new(p.x() - 1, p.y(), p.z()),
        ChunkPos::new(p.x() + 1, p.y(), p.z()),
        ChunkPos::new(p.x() - 1, p.y(), p.z() + 1),
        ChunkPos::new(p.x(), p.y(), p.z() + 1),
        ChunkPos::new(p.x() + 1, p.y(), p.z() + 1),
    ]
}

fn neighbours(p: ChunkPos) -> [ChunkPos; 6] {
    [
        ChunkPos::new(p.x() - 1, p.y(), p.z()),
        ChunkPos::new(p.x() + 1, p.y(), p.z()),
        ChunkPos::new(p.x(), p.y() - 1, p.z()),
        ChunkPos::new(p.x(), p.y() + 1, p.z()),
        ChunkPos::new(p.x(), p.y(), p.z() - 1),
        ChunkPos::new(p.x(), p.y(), p.z() + 1),
    ]
}

fn sort_by_distance(mut v: Vec<ChunkPos>, focus: ChunkPos) -> Vec<ChunkPos> {
    v.sort_by_key(|p| {
        let dx = p.x() - focus.x();
        let dy = p.y() - focus.y();
        let dz = p.z() - focus.z();
        dx * dx + dy * dy + dz * dz
    });
    v
}
