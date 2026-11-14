"""Routes from raid recordings, without screenshots: an optional part of eftjourney.

Every finished raid that has a recording gets one frame a second described by MegaLoc, a
place-recognition model. The frames of the other raids on the same map, placed by their
screenshots, are the places those descriptors are matched against, and a path model picks the
likeliest place for every second and says how sure it is (path.py).

Needs numpy, onnxruntime, FFmpeg and the MegaLoc model (pip install eftjourney[auto]); the rest
of the program runs without them and only reads the routes this writes (data/auto/<raid-id>.json).
Only raids that are over are analysed, from recordings that are no longer being written, and
never while the game runs.
"""
