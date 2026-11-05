# north-d01-asset

Robot models and mesh assets for D01-W and D01-S.

## Structure

| Directory | Contents |
| --- | --- |
| `common/visual/` | Visual meshes shared by W and S. |
| `common/precise-collision/` | Shared precise collision meshes. |
| `d01-w/ha1.2/` | W URDF, MJCF and USD models, plus W-specific visual and collision meshes. |
| `d01-s/ha1.1/` | S URDF models and the shared foot mesh for its four supports. MJCF and USD are not provided yet. |

## Collision rules

Ordinary URDF and MJCF models reuse visual meshes for collisions. Precise models
use dedicated collision meshes; W wheels also use cylinders, while S supports
reuse the foot mesh. URDF defines collision geometry; MJCF and USD also encode
self-collision filters. USD contains its own geometry and materials.

## Get the assets

After cloning, install Git LFS and download the large assets:

```bash
git lfs install
git lfs pull
```
