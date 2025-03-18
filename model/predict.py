import os
import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel
import joblib

# ✅ 모델 클래스 정의
class BERTClassifier(nn.Module):
    def __init__(self, num_classes, dropout=0.3):
        super(BERTClassifier, self).__init__()
        self.bert = BertModel.from_pretrained("bert-base-multilingual-cased")
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(self.bert.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        return self.fc(self.dropout(pooled_output))

# ✅ MPS(GPU) 사용 가능 여부 확인 후 자동 선택
if torch.backends.mps.is_available():
    device = torch.device("mps")  # ✅ MPS 사용
    print("✅ MPS(GPU) 사용 가능. MPS로 실행합니다.")
else:
    device = torch.device("cpu")  # ✅ CPU 사용
    print("❌ MPS(GPU) 사용 불가능. CPU에서 실행합니다.")

# ✅ 예측 함수
def predict_genre(synopsis):
    tokens = tokenizer(synopsis, padding="max_length", truncation=True, max_length=128, return_tensors="pt")
    input_ids = tokens["input_ids"].to(device)
    attention_mask = tokens["attention_mask"].to(device)

    with torch.no_grad():
        outputs = model(input_ids, attention_mask)
        predicted_label = torch.argmax(outputs, dim=1).item()

    predicted_genre = label_encoder.inverse_transform([predicted_label])[0]
    return predicted_genre

if __name__ == "__main__":
    # ✅ 파일 존재 여부 확인
    model_path = "../models/bert_webtoon_classifier.pth"
    label_encoder_path = "../models/label_encoder.pkl"

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"❌ 모델 파일이 없습니다: {model_path}")
    if not os.path.exists(label_encoder_path):
        raise FileNotFoundError(f"❌ 라벨 인코더 파일이 없습니다: {label_encoder_path}")

    # ✅ 모델 및 라벨 인코더 불러오기
    label_encoder = joblib.load(label_encoder_path)
    num_classes = len(label_encoder.classes_)

    model = BERTClassifier(num_classes=num_classes).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # ✅ 토크나이저 불러오기
    tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")

    # ✅ 테스트 예측 실행
    synopsis = "한 남자가 초능력을 사용하여 세상을 구한다."
    predicted_genre = predict_genre(synopsis)

    print(f"입력 줄거리: {synopsis}")
    print(f"예측된 장르: {predicted_genre}")