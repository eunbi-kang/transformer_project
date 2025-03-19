import torch

# PyTorch 버전 확인
print(f"🔥 PyTorch 버전: {torch.__version__}")

# MPS 지원 확인
print("✅ MPS 지원 여부:", torch.backends.mps.is_available())

# MPS 빌드 여부 확인
print("✅ MPS가 PyTorch에 포함되었는가:", torch.backends.mps.is_built())

# MPS에서 텐서 연산이 실제로 되는지 확인
try:
    x = torch.ones(2, 2).to("mps")  # MPS 장치로 텐서 이동
    print("✅ MPS에서 텐서 연산 성공:", x)
except Exception as e:
    print("❌ MPS 연산 실패:", e)
