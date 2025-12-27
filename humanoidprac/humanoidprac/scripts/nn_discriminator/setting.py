
WHOLE_JOINT_NUM = 19   # 全関節数 (首の2つ抜き)
OBS_DIMENSION = 88    # 観測の次元数
SEQUENCE_LENGTH = 10  # シーケンス長
HIDDEN_SIZE = 128    # GRUの隠れ状態の次元数
NUM_LAYERS = 2       # GRUの層数
CHUNK_SIZE = 200     # 学習時のチャンクサイズ
MAX_GRAD_NORM = 1.0  # 勾配クリッピングの最大ノルム
BATCH_SIZE = 1024   # バッチサイズ