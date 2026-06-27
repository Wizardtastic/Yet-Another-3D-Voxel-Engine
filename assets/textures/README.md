# Texture Overrides

Drop PNG files into this directory to provide block textures. Filenames
use the block/face name (stone.png, grass_top.png, etc.) rather than numbers,
so the system scales to hundreds of textures.

## How It Works

1. The config file `textures.toml` maps tile indices to PNG filenames
2. On startup, every tile is filled with a blue+black checkerboard error pattern
3. For each entry in `textures.toml`, the named PNG is loaded and placed at that tile
4. Tiles not listed (or whose PNG is missing) keep the error pattern

## Adding a New Texture

1. Add the PNG file to this directory (e.g. `granite.png`)
2. Add an entry to `textures.toml`:
   ```toml
   38 = "granite.png"
   ```
3. Update `crates/world/src/registry.rs` to use tile index 38 in the BlockDef

## Image Requirements

- Format: PNG (RGB or RGBA)
- Size: any size — images are automatically resized to 16x16 pixels
- Alpha channel is respected (transparent pixels work for foliage blocks)

## Error Texture

Missing textures show a blue+black checkerboard pattern so gaps are
immediately visible during development.

## Enabling

Set the textures directory in `config.toml`:

```toml
[graphics]
textures_dir = "assets/textures"
```

If the directory doesn't exist or has no `textures.toml`, all tiles show
the error pattern.
