//! Archetype-based storage.
//!
//! Entities that share the same component composition live in the same
//! [`Archetype`]. Each archetype owns a set of typed columns
//! ([`TypedColumn<T>`]) — one per component type — plus a parallel
//! `Vec<Entity>` listing which entity occupies each row.
//!
//! All component columns use an interior `UnsafeCell<Vec<T>>` so that
//! queries can hand out shared `&T` and exclusive `&mut T` references
//! from a shared `&Archetype` borrow. The `QueryIter` consumes one
//! reference at a time, so we never produce aliasing `&mut` references.

use std::any::{Any, TypeId};
use std::cell::UnsafeCell;
use std::collections::BTreeMap;

use crate::component::Component;
use crate::entity::Entity;

/// Unique identifier for an archetype.
///
/// Two archetypes are equal iff they have the same sorted set of component
/// types.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash, Ord, PartialOrd)]
pub struct ArchetypeId(pub u32);

/// Type-erased column. Knows how to swap-remove a row, push a boxed value,
/// and produce typed `Any` views.
pub(crate) trait ErasedColumn: Send + Sync {
    /// The `TypeId` of the component this column stores.
    #[allow(dead_code)]
    fn type_id(&self) -> TypeId;

    /// Append a boxed value of the column's element type. Panics if the
    /// boxed value's runtime type does not match.
    fn push_any(&mut self, value: Box<dyn Any>);

    /// Remove the row at `index`, swapping the last row into its place.
    /// Returns the removed value as a boxed `Any`.
    fn take_any(&mut self, index: u32) -> Box<dyn Any>;

    /// Borrow the column itself as a typed `Any`.
    fn as_any(&self) -> &dyn Any;
}

/// Typed column for a specific component.
///
/// The internal `Vec<T>` is wrapped in an `UnsafeCell` so that
/// [`Archetype::get`] and [`Archetype::get_mut`] can hand out shared and
/// exclusive references from a shared `&Archetype` (which is what
/// `QueryIter` holds). All callers must ensure that no two `&mut T`
/// references into the same column are live simultaneously. The
/// `QueryIter` produces one mutable reference at a time and the caller
/// drops it before asking for the next.
pub(crate) struct TypedColumn<T: Component> {
    data: UnsafeCell<Vec<T>>,
}

// SAFETY: `TypedColumn<T>` synchronizes access to the inner `Vec<T>`
// through `UnsafeCell` and is always accessed via the column API on
// `Archetype`. All callers go through the [`World`](crate::World) which
// is the sole owner of each archetype, so concurrent `&Self` access
// from multiple threads is sound. Manual `Sync` is required because
// `UnsafeCell` is `!Sync` by default.
unsafe impl<T: Component> Sync for TypedColumn<T> {}

impl<T: Component> TypedColumn<T> {
    pub(crate) fn new() -> Self {
        Self { data: UnsafeCell::new(Vec::new()) }
    }

    /// Append a value to the column. Requires exclusive access; the
    /// caller is responsible for ensuring no live references alias this
    /// column at call time.
    pub(crate) fn push(&self, value: T) {
        // SAFETY: the column is owned by the `World` and not aliased at
        // this point. The caller is the `World` mutating it during
        // archetype construction.
        let vec: &mut Vec<T> = unsafe { &mut *self.data.get() };
        vec.push(value);
    }

    pub(crate) fn get(&self, index: u32) -> Option<&T> {
        // SAFETY: the column outlives any reference we hand out (the
        // `QueryIter` borrows the whole `World`).
        let vec: &Vec<T> = unsafe { &*self.data.get() };
        vec.get(index as usize)
    }

    pub(crate) fn get_mut(&self, index: u32) -> Option<&mut T> {
        // SAFETY: see module docs. The `QueryIter` hands out one `&mut T`
        // at a time.
        let vec: &mut Vec<T> = unsafe { &mut *self.data.get() };
        vec.get_mut(index as usize)
    }

    /// Overwrite the value at `index`. Requires exclusive access; the
    /// caller is responsible for ensuring no live references alias this
    /// column at call time.
    pub(crate) fn set(&self, index: u32, value: T) {
        // SAFETY: see `push`.
        let vec: &mut Vec<T> = unsafe { &mut *self.data.get() };
        vec[index as usize] = value;
    }
}

impl<T: Component> ErasedColumn for TypedColumn<T> {
    fn type_id(&self) -> TypeId {
        TypeId::of::<T>()
    }

    fn push_any(&mut self, value: Box<dyn Any>) {
        let typed = value.downcast::<T>().expect("type mismatch in push_any");
        self.push(*typed);
    }

    fn take_any(&mut self, index: u32) -> Box<dyn Any> {
        // SAFETY: see `TypedColumn::push`.
        let vec: &mut Vec<T> = unsafe { &mut *self.data.get() };
        let removed = vec.swap_remove(index as usize);
        Box::new(removed)
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// A constructor for an empty [`ErasedColumn`] of a particular concrete
/// component type. Used by the [`World`](crate::World) to materialize
/// columns at archetype-creation time when it only knows a `TypeId`.
pub(crate) type ColumnCtor = fn() -> Box<dyn ErasedColumn>;

/// An archetype: a table of SoA columns for one specific component
/// composition, plus the list of entities currently occupying each row.
pub struct Archetype {
    pub id: ArchetypeId,
    /// Sorted unique `TypeId`s of the components stored by this archetype.
    pub component_types: Vec<TypeId>,
    /// Parallel to `component_types`: pretty type names for diagnostics.
    pub component_names: Vec<&'static str>,
    /// Maps `TypeId` to the index in `columns` (and `component_names`).
    pub(crate) column_index: BTreeMap<TypeId, usize>,
    /// Type-erased columns, one per component type. Indexed by
    /// `column_index[type_id]`.
    pub(crate) columns: Vec<Box<dyn ErasedColumn>>,
    /// Entities occupying each row, in lock-step with each column.
    pub entities: Vec<Entity>,
}

impl Archetype {
    /// Construct a new empty archetype with no columns. Columns are
    /// installed by [`Archetype::set_columns`] immediately after.
    pub fn new(id: ArchetypeId) -> Self {
        Self {
            id,
            component_types: Vec::new(),
            component_names: Vec::new(),
            column_index: BTreeMap::new(),
            columns: Vec::new(),
            entities: Vec::new(),
        }
    }

    /// Install the column set. The entries are sorted by `TypeId` so the
    /// resulting archetype is canonical — two archetypes with the same
    /// component set always agree on the column order.
    pub(crate) fn set_columns(&mut self, mut entries: Vec<(TypeId, &'static str, ColumnCtor)>) {
        entries.sort_by_key(|(t, _, _)| *t);
        self.component_types.clear();
        self.component_names.clear();
        self.columns.clear();
        self.column_index.clear();
        for (i, (t, name, ctor)) in entries.into_iter().enumerate() {
            self.component_types.push(t);
            self.component_names.push(name);
            self.columns.push(ctor());
            self.column_index.insert(t, i);
        }
    }

    /// Number of entities currently in this archetype.
    pub fn len(&self) -> usize {
        self.entities.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entities.is_empty()
    }

    /// Returns true iff this archetype stores a column for `T`.
    pub fn has<T: Component>(&self) -> bool {
        self.column_index.contains_key(&TypeId::of::<T>())
    }

    /// Shared reference to component `T` of the entity at `index`, if any.
    pub fn get<T: Component>(&self, index: u32) -> Option<&T> {
        let col_idx = *self.column_index.get(&TypeId::of::<T>())?;
        let col = self.columns[col_idx].as_any();
        let typed = col.downcast_ref::<TypedColumn<T>>()?;
        typed.get(index)
    }

    /// Exclusive reference to component `T` of the entity at `index`.
    pub fn get_mut<T: Component>(&self, index: u32) -> Option<&mut T> {
        let col_idx = *self.column_index.get(&TypeId::of::<T>())?;
        let col = self.columns[col_idx].as_any();
        let typed = col.downcast_ref::<TypedColumn<T>>()?;
        typed.get_mut(index)
    }

    /// Entities in this archetype, in row order.
    pub fn entities(&self) -> &[Entity] {
        &self.entities
    }

    /// Remove the entity at `index` (swap-remove). Returns the entity
    /// that was moved into the vacated slot, if any — the caller must
    /// update that entity's `EntityLocation` to point at `index`.
    pub(crate) fn swap_remove_entity(&mut self, index: u32) -> Option<Entity> {
        let last = self.entities.len().saturating_sub(1);
        let moved = if (index as usize) != last { Some(self.entities[last]) } else { None };
        self.entities.swap_remove(index as usize);
        for col in &mut self.columns {
            col.take_any(index);
        }
        moved
    }
}
