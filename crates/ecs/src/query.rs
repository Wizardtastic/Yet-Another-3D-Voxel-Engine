//! Query API for iterating entities that match a component pattern.
//!
//! A `Query` is anything that knows how to test an [`Archetype`] for a
//! match and how to fetch a typed view of one row. The most common
//! queries are bare references like `&A` (read-only single component),
//! `&mut A` (mutable single component), and tuples of references such
//! as `(&A, &B)` for reading two components in lock-step.

use std::marker::PhantomData;

use crate::archetype::Archetype;
use crate::component::Component;
use crate::entity::Entity;

/// A query for entities matching a set of component types.
///
/// Implemented for `&A`, `&mut A`, and tuples of references up to length
/// 4. See the `impl_query_tuple!` macro below for the tuple impls.
pub trait Query {
    type Item<'a>;

    /// Returns true iff this query matches `archetype` — i.e. all
    /// component types the query requires are present in the archetype.
    fn matches(archetype: &Archetype) -> bool;

    /// Fetch the item for the entity at `index` in `archetype`.
    fn fetch<'a>(archetype: &'a Archetype, index: usize) -> Self::Item<'a>;
}

// ---------------------------------------------------------------------
// Single-component queries: &A, &mut A
// ---------------------------------------------------------------------

impl<A: Component> Query for &A {
    type Item<'a> = &'a A;
    fn matches(archetype: &Archetype) -> bool {
        archetype.has::<A>()
    }
    fn fetch<'a>(archetype: &'a Archetype, index: usize) -> Self::Item<'a> {
        archetype
            .get::<A>(index as u32)
            .expect("query fetch: component A not present")
    }
}

impl<A: Component> Query for &mut A {
    type Item<'a> = &'a mut A;
    fn matches(archetype: &Archetype) -> bool {
        archetype.has::<A>()
    }
    fn fetch<'a>(archetype: &'a Archetype, index: usize) -> Self::Item<'a> {
        archetype
            .get_mut::<A>(index as u32)
            .expect("query fetch: component A not present")
    }
}

// ---------------------------------------------------------------------
// 1-tuples
// ---------------------------------------------------------------------

impl<A: Component> Query for (&A,) {
    type Item<'a> = (&'a A,);
    fn matches(archetype: &Archetype) -> bool {
        archetype.has::<A>()
    }
    fn fetch<'a>(archetype: &'a Archetype, index: usize) -> Self::Item<'a> {
        (archetype
            .get::<A>(index as u32)
            .expect("query fetch: component A not present"),)
    }
}

impl<A: Component> Query for (&mut A,) {
    type Item<'a> = (&'a mut A,);
    fn matches(archetype: &Archetype) -> bool {
        archetype.has::<A>()
    }
    fn fetch<'a>(archetype: &'a Archetype, index: usize) -> Self::Item<'a> {
        (archetype
            .get_mut::<A>(index as u32)
            .expect("query fetch: component A not present"),)
    }
}

// ---------------------------------------------------------------------
// 2-tuples: (read, read), (read, write), (write, read), (write, write)
// ---------------------------------------------------------------------

impl<A: Component, B: Component> Query for (&A, &B) {
    type Item<'a> = (&'a A, &'a B);
    fn matches(archetype: &Archetype) -> bool {
        archetype.has::<A>() && archetype.has::<B>()
    }
    fn fetch<'a>(archetype: &'a Archetype, index: usize) -> Self::Item<'a> {
        (
            archetype
                .get::<A>(index as u32)
                .expect("query fetch: component A not present"),
            archetype
                .get::<B>(index as u32)
                .expect("query fetch: component B not present"),
        )
    }
}

impl<A: Component, B: Component> Query for (&A, &mut B) {
    type Item<'a> = (&'a A, &'a mut B);
    fn matches(archetype: &Archetype) -> bool {
        archetype.has::<A>() && archetype.has::<B>()
    }
    fn fetch<'a>(archetype: &'a Archetype, index: usize) -> Self::Item<'a> {
        (
            archetype
                .get::<A>(index as u32)
                .expect("query fetch: component A not present"),
            archetype
                .get_mut::<B>(index as u32)
                .expect("query fetch: component B not present"),
        )
    }
}

impl<A: Component, B: Component> Query for (&mut A, &B) {
    type Item<'a> = (&'a mut A, &'a B);
    fn matches(archetype: &Archetype) -> bool {
        archetype.has::<A>() && archetype.has::<B>()
    }
    fn fetch<'a>(archetype: &'a Archetype, index: usize) -> Self::Item<'a> {
        (
            archetype
                .get_mut::<A>(index as u32)
                .expect("query fetch: component A not present"),
            archetype
                .get::<B>(index as u32)
                .expect("query fetch: component B not present"),
        )
    }
}

impl<A: Component, B: Component> Query for (&mut A, &mut B) {
    type Item<'a> = (&'a mut A, &'a mut B);
    fn matches(archetype: &Archetype) -> bool {
        archetype.has::<A>() && archetype.has::<B>()
    }
    fn fetch<'a>(archetype: &'a Archetype, index: usize) -> Self::Item<'a> {
        (
            archetype
                .get_mut::<A>(index as u32)
                .expect("query fetch: component A not present"),
            archetype
                .get_mut::<B>(index as u32)
                .expect("query fetch: component B not present"),
        )
    }
}

// ---------------------------------------------------------------------
// 3-tuples (read, read, read) and (write, write, write) for the common
// parallelizable cases.
// ---------------------------------------------------------------------

impl<A: Component, B: Component, C: Component> Query for (&A, &B, &C) {
    type Item<'a> = (&'a A, &'a B, &'a C);
    fn matches(archetype: &Archetype) -> bool {
        archetype.has::<A>() && archetype.has::<B>() && archetype.has::<C>()
    }
    fn fetch<'a>(archetype: &'a Archetype, index: usize) -> Self::Item<'a> {
        (
            archetype
                .get::<A>(index as u32)
                .expect("query fetch: component A not present"),
            archetype
                .get::<B>(index as u32)
                .expect("query fetch: component B not present"),
            archetype
                .get::<C>(index as u32)
                .expect("query fetch: component C not present"),
        )
    }
}

impl<A: Component, B: Component, C: Component> Query for (&mut A, &mut B, &mut C) {
    type Item<'a> = (&'a mut A, &'a mut B, &'a mut C);
    fn matches(archetype: &Archetype) -> bool {
        archetype.has::<A>() && archetype.has::<B>() && archetype.has::<C>()
    }
    fn fetch<'a>(archetype: &'a Archetype, index: usize) -> Self::Item<'a> {
        (
            archetype
                .get_mut::<A>(index as u32)
                .expect("query fetch: component A not present"),
            archetype
                .get_mut::<B>(index as u32)
                .expect("query fetch: component B not present"),
            archetype
                .get_mut::<C>(index as u32)
                .expect("query fetch: component C not present"),
        )
    }
}

// ---------------------------------------------------------------------
// 4-tuples (all read).
// ---------------------------------------------------------------------

impl<A: Component, B: Component, C: Component, D: Component> Query for (&A, &B, &C, &D) {
    type Item<'a> = (&'a A, &'a B, &'a C, &'a D);
    fn matches(archetype: &Archetype) -> bool {
        archetype.has::<A>()
            && archetype.has::<B>()
            && archetype.has::<C>()
            && archetype.has::<D>()
    }
    fn fetch<'a>(archetype: &'a Archetype, index: usize) -> Self::Item<'a> {
        (
            archetype
                .get::<A>(index as u32)
                .expect("query fetch: component A not present"),
            archetype
                .get::<B>(index as u32)
                .expect("query fetch: component B not present"),
            archetype
                .get::<C>(index as u32)
                .expect("query fetch: component C not present"),
            archetype
                .get::<D>(index as u32)
                .expect("query fetch: component D not present"),
        )
    }
}

// ---------------------------------------------------------------------
// Iterator
// ---------------------------------------------------------------------

/// Lazy iterator over the matching `(Entity, item)` pairs of a [`Query`].
pub struct QueryIter<'w, Q: Query> {
    archetypes: std::iter::Enumerate<std::slice::Iter<'w, Archetype>>,
    current: Option<&'w Archetype>,
    current_index: usize,
    _marker: PhantomData<Q>,
}

impl<'w, Q: Query> QueryIter<'w, Q> {
    pub(crate) fn new(world: &'w crate::World) -> Self {
        Self {
            archetypes: world.archetypes().iter().enumerate(),
            current: None,
            current_index: 0,
            _marker: PhantomData,
        }
    }
}

impl<'w, Q: Query> Iterator for QueryIter<'w, Q> {
    type Item = (Entity, Q::Item<'w>);

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if let Some(arch) = self.current {
                if self.current_index < arch.len() {
                    let entity = arch.entities()[self.current_index];
                    let item = Q::fetch(arch, self.current_index);
                    self.current_index += 1;
                    return Some((entity, item));
                }
            }
            // Advance to the next matching archetype.
            self.current_index = 0;
            self.current = None;
            for (_, arch) in self.archetypes.by_ref() {
                if Q::matches(arch) {
                    self.current = Some(arch);
                    break;
                }
            }
            if self.current.is_none() {
                return None;
            }
        }
    }
}
