import torch

# MPS가 사용 가능한지 확인
print("MPS 지원 여부:", torch.backends.mps.is_available())

# MPS 관련 드라이버가 활성화되었는지 확인
print("MPS가 작동 가능한 상태인지:", torch.backends.mps.is_built())