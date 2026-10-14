use std::rc::Rc;

use crate::value::Value;

// The runtime environment: a persistent chain of immutable frames, one per
// lexical scope (see resolve.rs's scope table for exactly which scopes get a
// frame). A frame is fully built and only then wrapped in an Rc, so closure
// capture and multi-shot continuation resume stay safe by construction.
// `None` is the root (the builtin prelude is NOT stored in any frame --
// resolve::VarRef::Prelude indexes env::PRELUDE directly).

// What one frame holds. One/Two avoid a second allocation for the dominant
// shapes (a lambda param or let, a handler's payload+resume); Many is the
// group / multi-binder-pattern case. Private layout -- benchmark-tunable.
enum Slots {
    One(Value),
    Two(Value, Value),
    Many(Vec<Value>),
}

struct Frame {
    parent: Env,
    slots: Slots,
}

#[derive(Clone)]
pub struct Env(Option<Rc<Frame>>);

impl Env {
    pub fn root() -> Env {
        Env(None)
    }

    pub fn is_root(&self) -> bool {
        self.0.is_none()
    }

    pub fn push1(&self, v: Value) -> Env {
        self.push(Slots::One(v))
    }

    pub fn push2(&self, a: Value, b: Value) -> Env {
        self.push(Slots::Two(a, b))
    }

    fn push(&self, slots: Slots) -> Env {
        Env(Some(Rc::new(Frame { parent: self.clone(), slots })))
    }

    // Walks `hops` parents (by reference -- no Rc clones on the way) and
    // reads `slot`. A miss here means the resolver and the machine disagree
    // about a scope's layout, which is a bug, never a user error.
    pub fn get(&self, hops: u32, slot: u32) -> Value {
        let mut frame = self.0.as_deref().expect("variable resolved past the root frame");
        for _ in 0..hops {
            frame = frame.parent.0.as_deref().expect("variable resolved past the root frame");
        }
        match (&frame.slots, slot) {
            (Slots::One(v), 0) | (Slots::Two(v, _), 0) | (Slots::Two(_, v), 1) => v.clone(),
            (Slots::Many(vs), s) => vs[s as usize].clone(),
            _ => panic!("internal: slot {slot} out of range for its frame"),
        }
    }
}

// Same shape and reason as plist.rs's Drop for PList: the default drop
// recurses through `parent` one frame at a time, which is native stack
// proportional to the chain (a long sequence of lets builds exactly that).
// Unwind iteratively via Rc::try_unwrap, stopping the moment some other
// clone of a frame (a closure or continuation capturing it) still exists.
impl Drop for Env {
    fn drop(&mut self) {
        let mut cur = self.0.take();
        while let Some(rc) = cur {
            match Rc::try_unwrap(rc) {
                Ok(mut frame) => cur = frame.parent.0.take(),
                Err(_) => break,
            }
        }
    }
}

// The values one scope binds, collected in binder order (the order
// resolve::pattern_vars / the scope table define). Allocation-free up to two
// values, so a one-var arm or a handler costs exactly one Rc<Frame>.
#[derive(Default)]
pub struct Bindings(Option<Slots>);

impl Bindings {
    pub fn push(&mut self, v: Value) {
        self.0 = Some(match self.0.take() {
            None => Slots::One(v),
            Some(Slots::One(a)) => Slots::Two(a, v),
            Some(Slots::Two(a, b)) => Slots::Many(vec![a, b, v]),
            Some(Slots::Many(mut vs)) => {
                vs.push(v);
                Slots::Many(vs)
            }
        });
    }

    pub fn len(&self) -> usize {
        match &self.0 {
            None => 0,
            Some(Slots::One(_)) => 1,
            Some(Slots::Two(..)) => 2,
            Some(Slots::Many(vs)) => vs.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_none()
    }

    // The env a scope with these bindings runs in. A scope that bound
    // nothing has NO frame (matches resolve.rs: a zero-var match arm pushes
    // no scope), so `parent` itself is returned.
    pub fn extend(self, parent: &Env) -> Env {
        match self.0 {
            None => parent.clone(),
            Some(slots) => parent.push(slots),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::value::Value;

    fn int(n: i64) -> Value {
        Value::Int(n)
    }

    #[test]
    fn push1_and_hops() {
        let env = Env::root().push1(int(1)).push1(int(2));
        assert_eq!(env.get(0, 0).as_int(), 2);
        assert_eq!(env.get(1, 0).as_int(), 1);
    }

    #[test]
    fn push2_slots() {
        let env = Env::root().push2(int(10), int(20));
        assert_eq!(env.get(0, 0).as_int(), 10);
        assert_eq!(env.get(0, 1).as_int(), 20);
    }

    #[test]
    fn bindings_of_each_size_index_in_push_order() {
        for n in 1..=5usize {
            let mut b = Bindings::default();
            for i in 0..n {
                b.push(int(i as i64));
            }
            assert_eq!(b.len(), n);
            let env = b.extend(&Env::root());
            for i in 0..n {
                assert_eq!(env.get(0, i as u32).as_int(), i as i64, "n={n} slot={i}");
            }
        }
    }

    #[test]
    fn empty_bindings_add_no_frame() {
        let base = Env::root().push1(int(7));
        let b = Bindings::default();
        assert!(b.is_empty());
        let env = b.extend(&base);
        // Still exactly one frame deep: slot 0 of hop 0 is the base's value.
        assert_eq!(env.get(0, 0).as_int(), 7);
        assert!(Bindings::default().extend(&Env::root()).is_root());
    }

    #[test]
    fn root_is_root() {
        assert!(Env::root().is_root());
        assert!(!Env::root().push1(int(1)).is_root());
    }

    // The default test-thread stack (~2 MiB) would overflow on a recursive
    // Drop at this depth; the iterative Drop must not.
    #[test]
    fn deep_chain_drops_without_overflow() {
        let mut env = Env::root();
        for i in 0..300_000 {
            env = env.push1(int(i));
        }
        drop(env);
    }

    // A shared tail must survive: dropping one chain stops unwinding at a
    // frame something else still holds.
    #[test]
    fn shared_tail_survives_drop_of_a_longer_chain() {
        let tail = Env::root().push1(int(99));
        let longer = tail.push1(int(1)).push1(int(2));
        drop(longer);
        assert_eq!(tail.get(0, 0).as_int(), 99);
    }

    #[test]
    #[should_panic(expected = "past the root frame")]
    fn get_past_the_root_panics() {
        Env::root().push1(int(1)).get(1, 0);
    }
}
