//! The ECS `World`.
//!
//! Owns all entities, archetypes, and resources. Provides the primary
//! spawn/despawn/set/get API plus the resource and query entry points.

use std::any::{Any, TypeId};
use std::collections::HashMap;

use crate::archetype::{Archetype, ArchetypeId, ColumnCtor, ErasedColumn, TypedColumn};
use crate::component::{Bundle, Component};
use crate::entity::{Entity, EntityLocation};
use crate::query::{Query, QueryIter};
use crate::resources::Resources;

/// The ECS world. Single owner of all entity, archetype, and resource
/// state.
pub struct World {
    archetypes: Vec<Archetype>,
    /// Sorted unique `TypeId` sets → [`ArchetypeId`].
    archetype_by_components: HashMap<Vec<TypeId>, ArchetypeId>,
    /// `EntityLocation` per entity slot, indexed by `Entity.index`.
    entities: Vec<EntityLocation>,
    /// Current generation per entity slot, indexed by `Entity.index`.
    generations: Vec<u32>,
    /// Recycled entity indices, LIFO.
    free_list: Vec<u32>,
    /// Constructor for each component column type, keyed by `TypeId`.
    /// Populated lazily on first use of a component type.
    column_ctors: HashMap<TypeId, ColumnCtor>,
    /// World-wide singleton resources.
    pub resources: Resources,
}

/// Free function used as the column constructor for a component type.
fn make_column<T: crate::component::Component>() -> Box<dyn ErasedColumn> {
    Box::new(TypedColumn::<T>::new())
}

impl Default for World {
    fn default() -> Self {
        Self::new()
    }
}

impl World {
    /// Create an empty world.
    pub fn new() -> Self {
        Self {
            archetypes: Vec::new(),
            archetype_by_components: HashMap::new(),
            entities: Vec::new(),
            generations: Vec::new(),
            free_list: Vec::new(),
            column_ctors: HashMap::new(),
            resources: Resources::new(),
        }
    }

    // -----------------------------------------------------------------
    // Entity allocation
    // -----------------------------------------------------------------

    /// Allocate a fresh entity slot. If a slot is on the free list it is
    /// recycled and its generation is bumped; otherwise a new slot is
    /// appended.
    fn alloc_entity(&mut self) -> Entity {
        if let Some(index) = self.free_list.pop() {
            let gen = self.generations[index as usize] + 1;
            self.generations[index as usize] = gen;
            self.entities[index as usize] = EntityLocation {
                archetype: EntityLocation::EMPTY,
                index: 0,
            };
            Entity { index, generation: gen }
        } else {
            let index = self.entities.len() as u32;
            self.entities.push(EntityLocation {
                archetype: EntityLocation::EMPTY,
                index: 0,
            });
            self.generations.push(0);
            Entity { index, generation: 0 }
        }
    }

    /// Returns true iff the entity's generation matches the current
    /// generation of its slot.
    pub fn is_alive(&self, entity: Entity) -> bool {
        if entity.is_null() {
            return false;
        }
        self.generations
            .get(entity.index as usize)
            .copied()
            == Some(entity.generation)
    }

    /// Number of live entities (those whose generation matches their slot).
    pub fn entity_count(&self) -> u32 {
        // Generations match by construction; live count = slot count - free list.
        (self.entities.len() as u32) - (self.free_list.len() as u32)
    }

    /// Number of archetypes in the world.
    pub fn archetype_count(&self) -> usize {
        self.archetypes.len()
    }

    /// All archetypes currently in the world. Used by
    /// [`QueryIter`](crate::query::QueryIter).
    pub(crate) fn archetypes(&self) -> &[Archetype] {
        &self.archetypes
    }

    // -----------------------------------------------------------------
    // Type registry
    // -----------------------------------------------------------------

    /// Lazily register a column constructor for component type `T` if it
    /// is not already known.
    fn ensure_registered<T: Component>(&mut self) {
        let id = TypeId::of::<T>();
        if !self.column_ctors.contains_key(&id) {
            // `make_column::<T>` is a fn-item of type `ColumnCtor`.
            self.column_ctors.insert(id, make_column::<T>);
        }
    }

    /// Look up the constructor for a known type. Panics if the type was
    /// never registered (which means no component of that type was ever
    /// touched).
    fn ctor_for(&self, type_id: TypeId) -> ColumnCtor {
        self.column_ctors
            .get(&type_id)
            .copied()
            .expect("component type used without ensure_registered")
    }

    // -----------------------------------------------------------------
    // Archetype lookup / creation
    // -----------------------------------------------------------------

    /// Look up an archetype by its sorted component type set, creating
    /// it on demand.
    fn get_or_create_archetype(&mut self, types: &[TypeId]) -> ArchetypeId {
        let key: Vec<TypeId> = {
            let mut t = types.to_vec();
            t.sort();
            t.dedup();
            t
        };
        if let Some(&id) = self.archetype_by_components.get(&key) {
            return id;
        }
        let id = ArchetypeId(self.archetypes.len() as u32);
        let mut arch = Archetype::new(id);
        let entries: Vec<(TypeId, &'static str, ColumnCtor)> = key
            .iter()
            .map(|&t| (t, std::any::type_name::<dyn ErasedColumn>(), self.ctor_for(t)))
            // We want the actual type name of T, not `dyn ErasedColumn`. Replace
            // the name with a placeholder; the real name is filled in by the
            // registry at construction time.
            .map(|(t, _name, ctor)| (t, "<component>", ctor))
            .collect();
        // We don't know the real name here — the column's element type is
        // erased. Replace placeholder with the type's `TypeId` for now
        // (debugging only). Callers that need a name can ask the column
        // for its `type_id`.
        arch.set_columns(entries);
        self.archetype_by_components.insert(key, id);
        self.archetypes.push(arch);
        id
    }

    // -----------------------------------------------------------------
    // Archetype transitions
    // -----------------------------------------------------------------

    /// Move `entity` from its current archetype to the archetype with
    /// `to_remove` types removed and `to_add` types added.
    ///
    /// Returns the value taken from the first column in `to_remove` (if
    /// any), as a `Box<dyn Any>`. Panics if more than one type is in
    /// `to_remove`.
    fn transition(
        &mut self,
        entity: Entity,
        to_remove: &[TypeId],
        to_add: Vec<(TypeId, Box<dyn Any>)>,
    ) -> Option<Box<dyn Any>> {
        let loc = self.entities[entity.index as usize];
        let old_arch_id = loc.archetype;
        let old_idx = loc.index;

        // Compute the new archetype's component set.
        let old_types: Vec<TypeId> = if old_arch_id == EntityLocation::EMPTY {
            Vec::new()
        } else {
            self.archetypes[old_arch_id as usize].component_types.clone()
        };

        let mut new_types = old_types.clone();
        new_types.retain(|t| !to_remove.contains(t));
        for (t, _) in &to_add {
            if !new_types.contains(t) {
                new_types.push(*t);
            }
        }
        new_types.sort();
        new_types.dedup();

        let new_arch_id = self.get_or_create_archetype(&new_types);

        // Same archetype: just overwrite the value. (Only relevant when
        // to_remove and to_add are both empty and the entity is already
        // in the target archetype; this should be handled by callers.)
        if old_arch_id == new_arch_id.0
            && old_arch_id != EntityLocation::EMPTY
            && to_add.is_empty()
            && to_remove.is_empty()
        {
            return None;
        }

        // Build a fast-lookup set of `to_add` types.
        let to_add_set: std::collections::HashSet<TypeId> =
            to_add.iter().map(|(t, _)| *t).collect();

        // Capture the last row in the old archetype's entities vec
        // before we touch anything — we need it to identify the entity
        // that will be swap-removed into `old_idx`.
        let old_last: usize = if old_arch_id == EntityLocation::EMPTY {
            0
        } else {
            self.archetypes[old_arch_id as usize]
                .entities
                .len()
                .saturating_sub(1)
        };

        // First pass: drain the old archetype's columns at `old_idx`,
        // collecting (type, value) pairs to push to the new archetype.
        // We collect the type IDs first via a shared borrow so the
        // mutable borrow of the columns is not held while we make
        // trait-object method calls (which the borrow checker is
        // conservative about with `Box<dyn Trait>`).
        let old_col_types: Vec<TypeId> = if old_arch_id == EntityLocation::EMPTY {
            Vec::new()
        } else {
            self.archetypes[old_arch_id as usize]
                .columns
                .iter()
                // Use fully qualified syntax: `c.type_id()` would resolve
                // to `Any::type_id` (returning the wrapper's TypeId)
                // rather than our `ErasedColumn::type_id` (which returns
                // the component's TypeId).
                .map(|c| crate::archetype::ErasedColumn::type_id(&**c))
                .collect()
        };

        let mut moves: Vec<(TypeId, Box<dyn Any>)> = Vec::new();
        let mut taken: Option<Box<dyn Any>> = None;
        if old_arch_id != EntityLocation::EMPTY {
            let old_arch = &mut self.archetypes[old_arch_id as usize];
            for (i, t) in old_col_types.into_iter().enumerate() {
                if to_remove.contains(&t) {
                    let value = old_arch.columns[i].take_any(old_idx);
                    if taken.is_some() {
                        panic!("transition: multiple types in to_remove");
                    }
                    taken = Some(value);
                } else if !to_add_set.contains(&t) {
                    // Type is in both old and new: move the value.
                    moves.push((t, old_arch.columns[i].take_any(old_idx)));
                }
                // else: type is in to_add, we drop the old value (it'll
                // be replaced by the new one).
            }
        }

        // Second pass: push all values into the new archetype.
        {
            let new_arch = &mut self.archetypes[new_arch_id.0 as usize];
            for (t, value) in moves {
                let new_col_idx = new_arch
                    .column_index
                    .get(&t)
                    .copied()
                    .expect("missing column in new archetype");
                new_arch.columns[new_col_idx].push_any(value);
            }
            for (t, value) in to_add {
                let new_col_idx = new_arch
                    .column_index
                    .get(&t)
                    .copied()
                    .expect("missing column in new archetype after get_or_create");
                new_arch.columns[new_col_idx].push_any(value);
            }
        }

        // Remove the entity from the old archetype's `entities` vec.
        // The columns were already drained in the first pass, so we
        // must NOT call `swap_remove_entity` (which would drain them a
        // second time). We just swap-remove from `entities` directly.
        let moved = if old_arch_id != EntityLocation::EMPTY {
            let old_arch = &mut self.archetypes[old_arch_id as usize];
            let moved_entity = if (old_idx as usize) != old_last {
                Some(old_arch.entities[old_last])
            } else {
                None
            };
            old_arch.entities.swap_remove(old_idx as usize);
            moved_entity
        } else {
            None
        };

        // Add the entity to the new archetype.
        let new_idx = self.archetypes[new_arch_id.0 as usize].entities.len() as u32;
        self.archetypes[new_arch_id.0 as usize].entities.push(entity);

        // Update the entity's location.
        self.entities[entity.index as usize] = EntityLocation {
            archetype: new_arch_id.0,
            index: new_idx,
        };

        // The "moved" entity was at `old_last` in the old archetype; the
        // swap-remove put it at `old_idx`. Update its location.
        if let Some(moved_entity) = moved {
            self.entities[moved_entity.index as usize].index = old_idx;
        }

        taken
    }

    // -----------------------------------------------------------------
    // Public API
    // -----------------------------------------------------------------

    /// Spawn a new entity from a bundle of components.
    pub fn spawn<B: Bundle>(&mut self, bundle: B) -> Entity {
        let entity = self.alloc_entity();
        bundle.add_to(self, entity);
        entity
    }

    /// Despawn an entity. Returns `true` if the entity was alive.
    pub fn despawn(&mut self, entity: Entity) -> bool {
        if !self.is_alive(entity) {
            return false;
        }
        let loc = self.entities[entity.index as usize];
        if loc.archetype != EntityLocation::EMPTY {
            let moved = self.archetypes[loc.archetype as usize].swap_remove_entity(loc.index);
            if let Some(moved_entity) = moved {
                self.entities[moved_entity.index as usize].index = loc.index;
            }
        }
        // Bump generation and recycle the slot.
        self.generations[entity.index as usize] += 1;
        self.free_list.push(entity.index);
        true
    }

    /// Set (or insert) component `T` on `entity`. If the entity already
    /// has `T` the value is replaced in place; otherwise the entity is
    /// moved to an archetype that includes `T`.
    pub fn set<T: Component>(&mut self, entity: Entity, value: T) {
        assert!(self.is_alive(entity), "set: entity {:?} is not alive", entity);
        self.ensure_registered::<T>();

        let loc = self.entities[entity.index as usize];
        let type_id = TypeId::of::<T>();

        if loc.archetype == EntityLocation::EMPTY {
            // Brand new entity, transition to {T}.
            self.transition(entity, &[], vec![(type_id, Box::new(value) as Box<dyn Any>)]);
        } else {
            let arch_id = loc.archetype;
            let col_idx = self.archetypes[arch_id as usize]
                .column_index
                .get(&type_id)
                .copied();
            if let Some(col_idx) = col_idx {
                // Entity has T; replace in place.
                let col = self.archetypes[arch_id as usize].columns[col_idx].as_any();
                let typed = col
                    .downcast_ref::<TypedColumn<T>>()
                    .expect("column downcast mismatch in set");
                typed.set(loc.index, value);
            } else {
                // Entity doesn't have T; transition.
                self.transition(
                    entity,
                    &[],
                    vec![(type_id, Box::new(value) as Box<dyn Any>)],
                );
            }
        }
    }

    /// Borrow component `T` of `entity`, if present.
    pub fn get<T: Component>(&self, entity: Entity) -> Option<&T> {
        if !self.is_alive(entity) {
            return None;
        }
        let loc = self.entities[entity.index as usize];
        if loc.archetype == EntityLocation::EMPTY {
            return None;
        }
        self.archetypes[loc.archetype as usize].get::<T>(loc.index)
    }

    /// Mutably borrow component `T` of `entity`, if present.
    pub fn get_mut<T: Component>(&mut self, entity: Entity) -> Option<&mut T> {
        if !self.is_alive(entity) {
            return None;
        }
        let loc = self.entities[entity.index as usize];
        if loc.archetype == EntityLocation::EMPTY {
            return None;
        }
        self.archetypes[loc.archetype as usize].get_mut::<T>(loc.index)
    }

    /// Remove component `T` from `entity`, returning the removed value.
    /// The entity is moved to the archetype that excludes `T`.
    pub fn remove<T: Component>(&mut self, entity: Entity) -> Option<T> {
        if !self.is_alive(entity) {
            return None;
        }
        let loc = self.entities[entity.index as usize];
        if loc.archetype == EntityLocation::EMPTY {
            return None;
        }
        let type_id = TypeId::of::<T>();
        let arch_id = loc.archetype;
        if !self.archetypes[arch_id as usize]
            .column_index
            .contains_key(&type_id)
        {
            return None;
        }
        let taken = self.transition(entity, &[type_id], vec![]);
        taken.and_then(|b| b.downcast::<T>().ok().map(|b| *b))
    }

    /// Returns true iff `entity` has a component of type `T`.
    pub fn has<T: Component>(&self, entity: Entity) -> bool {
        self.get::<T>(entity).is_some()
    }

    /// Begin a query for components matching `Q`.
    pub fn query<Q: Query>(&self) -> QueryIter<'_, Q> {
        QueryIter::new(self)
    }

    // -----------------------------------------------------------------
    // Resource API (forwards to self.resources)
    // -----------------------------------------------------------------

    pub fn insert_resource<T: Send + Sync + 'static>(&mut self, value: T) -> Option<T> {
        self.resources.insert(value)
    }

    pub fn resource<T: Send + Sync + 'static>(&self) -> Option<&T> {
        self.resources.get::<T>()
    }

    pub fn resource_mut<T: Send + Sync + 'static>(&mut self) -> Option<&mut T> {
        self.resources.get_mut::<T>()
    }

    pub fn remove_resource<T: Send + Sync + 'static>(&mut self) -> Option<T> {
        self.resources.remove::<T>()
    }
}
