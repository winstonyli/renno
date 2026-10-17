# Roadmap

Ordered milestones, each the question it answers. Tagging a milestone moves its line into the tag message (`experiment`: the milestone sentence). Deferred work is in `PARKED.md`.

- v0.2 - Is checking linear in scope depth? U8b level-based generalization (branch `u8b-levels`; q1 0.77 s -> 0.05 s at n=4000, cost on generalized-let chains still being gated).
- v0.3 - Can effect rows unify like types? U1, the one row sort with open tails; decide first whether row-polymorphic records are ever wanted (see PARKED.md).
- v0.4 - Are containers sound at Dyn crossings? U7, one lazy structural wrapper through List/Tuple/Record.
