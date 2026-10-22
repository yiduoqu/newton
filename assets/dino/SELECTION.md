# dino

The original selected export is copied in full and byte-for-byte. Its manifests and parameter arrays are unchanged.

For detailed static video rendering, use `high_detail_source/mesh_combined_textured.glb`. It embeds its textures. This source surface is more detailed than the tetrahedral simulation boundary and is not a replacement for the original visual-to-physics binding.

For the processed simulation appearance, use the original `appearance/visual_mesh.glb` and its matching binding when present. The Toyandona legacy export stores its original textured source beneath `asset_refinement/omnipart/`.

## Provenance

- Selected source location: local.
- Selected source folder: `outputs/dino_asset_20260911/dino_retry_20260910_02/`.
- Local complete source: `outputs/dino_asset_20260911/dino_retry_20260910_02/`.
- Additional high-detail source: `outputs/dino_mesh_retry_20260910/attempt_02/glb/mesh_combined_textured.glb`.
- High-detail SHA-256: `01d1fd2af5fbe015009fa009bf18494b3613937b984d9eb2a94ba61520c47886`.

## Selection and limitations

The only reviewed generation with a coherent four-legged dinosaur, intact head/neck/tail, and no severely fused or malformed limb arrangement.

The final visual has only 1,130 triangles and closes the small attachment-loop opening. The higher-detail OmniPart source retains more shape and texture detail. Uniform 5 MPa rubber approximation; diagnostics bypassed.
