# D01-W MJCF

| Scene | Model | Collision geometry |
| --- | --- | --- |
| `NTH-HA1-scene.xml` | `NTH-HA1.xml` | Visual meshes reused for collisions. |
| `NTH-HA1-scene-precise-collision.xml` | `NTH-HA1-precise-collision.xml` | Dedicated precise meshes and wheel cylinders. |

## Structure

Both models share visual meshes from `../../../common/visual/` and `../visual/`.
The precise model uses collision meshes from `../../../common/precise-collision/`
and `../precise-collision/`. Each scene includes its model and a ground plane.

## Collision rules

Visual geometry does not participate in collisions; separate collision geometry
provides contact. Contact masks and body-pair exclusions filter selected internal
and adjacent parts. The ordinary and precise models have different collision
geometry and filtering, so their contact behavior can differ.
