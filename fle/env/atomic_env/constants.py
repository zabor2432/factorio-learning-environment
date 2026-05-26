ACTION_NOOP = 0
ACTION_PLACE_DRILL = 1
ACTION_PLACE_POLE = 2
N_ACTION_TYPES = 3

GRID_SIZE = 32
GRID_CENTER = GRID_SIZE // 2  # 16

# Grid channels
CH_COAL_RESOURCE = 0   # coal tiles (fixed per episode)
CH_DRILLS = 1          # placed electric mining drills
CH_POLES = 2           # placed electric poles
CH_PLAYER = 3          # player position (dynamic)
N_CHANNELS = 4

# Flat feature indices
FEAT_MINERS_PLACED = 0
FEAT_MINERS_ON_COAL = 1   # drills overlapping a coal tile (computed from grid)
FEAT_POLES_PLACED = 2
FEAT_COAL_TOTAL = 3        # cumulative coal drained this episode
N_FEATURES = 4

N_DIRECTIONS = 4
DIRECTION_MAP = {0: "UP", 1: "RIGHT", 2: "DOWN", 3: "LEFT"}

# Shaped reward bonus for placing a drill that overlaps a coal tile
SHAPED_DRILL_ON_COAL_REWARD = 1.0
