"""Versioned Stage-1 FFT entry point; defaults to the frozen 117-episode recipe."""

from scripts.biyi import checkpoint_store as store
from scripts.biyi.train_pi05_100ep import main

if __name__ == "__main__":
    main(default_recipe=store.RECIPE_117)
