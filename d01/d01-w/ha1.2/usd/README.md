# D01-W USD

Open `D01_NTH_HA1_SimReady.usdc` directly in Isaac Sim after downloading the assets
with Git LFS.

## Structure

The USDC contains the robot geometry, materials, joints and collision shapes.
Geometry and materials are embedded; external STL files are not required.

## Collision rules

The asset uses precise collision geometry and pairwise self-collision filters.
Selected internal and adjacent parts are excluded from contact; other eligible
robot parts and the environment can interact.
