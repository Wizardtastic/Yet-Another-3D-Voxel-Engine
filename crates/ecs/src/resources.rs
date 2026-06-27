//! Type-indexed resource storage.
//!
//! Resources are singleton values associated with the [`World`](crate::World).
//! Each resource is identified by its `TypeId`, so two systems can share
//! state without explicitly passing it through function arguments.

use std::any::{Any, TypeId};
use std::collections::HashMap;

/// Container of world-wide singleton resources.
#[derive(Default)]
pub struct Resources {
    map: HashMap<TypeId, Box<dyn Any + Send + Sync>>,
}

impl Resources {
    pub fn new() -> Self {
        Self::default()
    }

    /// Insert (or replace) a resource of type `T`. Returns the previous
    /// value, if any.
    pub fn insert<T: Send + Sync + 'static>(&mut self, value: T) -> Option<T> {
        self.map
            .insert(TypeId::of::<T>(), Box::new(value) as Box<dyn Any + Send + Sync>)
            .and_then(|b| b.downcast::<T>().ok().map(|b| *b))
    }

    /// Borrow a resource of type `T`.
    pub fn get<T: Send + Sync + 'static>(&self) -> Option<&T> {
        self.map.get(&TypeId::of::<T>())?.downcast_ref::<T>()
    }

    /// Mutably borrow a resource of type `T`.
    pub fn get_mut<T: Send + Sync + 'static>(&mut self) -> Option<&mut T> {
        self.map.get_mut(&TypeId::of::<T>())?.downcast_mut::<T>()
    }

    /// Remove a resource of type `T`, returning it if present.
    pub fn remove<T: Send + Sync + 'static>(&mut self) -> Option<T> {
        self.map
            .remove(&TypeId::of::<T>())?
            .downcast::<T>()
            .ok()
            .map(|b| *b)
    }

    /// Returns true iff a resource of type `T` is present.
    pub fn contains<T: Send + Sync + 'static>(&self) -> bool {
        self.map.contains_key(&TypeId::of::<T>())
    }
}
