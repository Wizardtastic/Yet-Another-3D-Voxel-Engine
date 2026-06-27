//! [`System`] trait and [`SystemSchedule`] for ordered system execution.
//!
//! Systems mutate the [`World`] in place. The schedule runs them in
//! registration order, sequentially, on a single thread. The
//! `can_parallelize` flag is provided for a future parallel scheduler:
//! systems that touch global resources (player state, etc.) declare
//! themselves exclusive.

use crate::world::World;

/// A system operates on the world. Systems run once per fixed step.
pub trait System: Send + Sync {
    /// Called once per fixed step. `dt` is the fixed timestep in seconds.
    fn run(&mut self, world: &mut World, dt: f32);

    /// Human-readable name for debugging and logging.
    fn name(&self) -> &str {
        "unnamed"
    }

    /// Whether this system is safe to run in parallel with other
    /// systems. The current scheduler is sequential; this flag is
    /// advisory and will be used by a future parallel scheduler.
    fn can_parallelize(&self) -> bool {
        true
    }
}

/// A closure-based system implementation.
pub struct FnSystem<F> {
    name: String,
    func: F,
    can_parallel: bool,
}

impl<F: FnMut(&mut World, f32) + Send + Sync> FnSystem<F> {
    /// Wrap a closure as a parallelizable system.
    pub fn new(name: impl Into<String>, func: F) -> Self {
        Self {
            name: name.into(),
            func,
            can_parallel: true,
        }
    }

    /// Wrap a closure as an exclusive (non-parallelizable) system.
    pub fn new_exclusive(name: impl Into<String>, func: F) -> Self {
        Self {
            name: name.into(),
            func,
            can_parallel: false,
        }
    }
}

impl<F: FnMut(&mut World, f32) + Send + Sync> System for FnSystem<F> {
    fn run(&mut self, world: &mut World, dt: f32) {
        (self.func)(world, dt);
    }
    fn name(&self) -> &str {
        &self.name
    }
    fn can_parallelize(&self) -> bool {
        self.can_parallel
    }
}

/// Ordered list of systems. Runs systems in registration order.
pub struct SystemSchedule {
    systems: Vec<Box<dyn System>>,
}

impl SystemSchedule {
    pub fn new() -> Self {
        Self { systems: Vec::new() }
    }

    /// Add a system to the schedule, returning the schedule for chaining.
    pub fn add_system<S: System + 'static>(mut self, system: S) -> Self {
        self.systems.push(Box::new(system));
        self
    }

    /// Add a closure as a system, returning the schedule for chaining.
    pub fn add_fn<F>(self, name: impl Into<String>, func: F) -> Self
    where
        F: FnMut(&mut World, f32) + Send + Sync + 'static,
    {
        self.add_system(FnSystem::new(name, func))
    }

    /// Add an exclusive (non-parallelizable) closure as a system.
    pub fn add_exclusive_fn<F>(self, name: impl Into<String>, func: F) -> Self
    where
        F: FnMut(&mut World, f32) + Send + Sync + 'static,
    {
        self.add_system(FnSystem::new_exclusive(name, func))
    }

    /// Run all systems in order.
    pub fn run(&mut self, world: &mut World, dt: f32) {
        for system in &mut self.systems {
            system.run(world, dt);
        }
    }

    /// Number of registered systems.
    pub fn len(&self) -> usize {
        self.systems.len()
    }

    pub fn is_empty(&self) -> bool {
        self.systems.is_empty()
    }
}

impl Default for SystemSchedule {
    fn default() -> Self {
        Self::new()
    }
}
