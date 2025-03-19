import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from transformers import BertTokenizer, BertModel
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset, DataLoader
import joblib
import json

# ✅ 실행 장치: 항상 GPU(MPS) 사용
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
print(f"✅ 실행 장치: {device}")

# ✅ Mixed Precision (반정밀도 연산) 활성화 → 속도 개선
scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "mps"))

# ✅ 1. BERT 분류 모델 정의
class BERTClassifier(nn.Module):
    def __init__(self, num_classes, dropout=0.3):
        super(BERTClassifier, self).__init__()
        self.bert = BertModel.from_pretrained("bert-base-multilingual-cased")
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(self.bert.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        with torch.cuda.amp.autocast(enabled=(device.type == "mps")):  # ✅ Mixed Precision 적용
            outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
            pooled_output = outputs.pooler_output
            return self.fc(self.dropout(pooled_output))


# ✅ 2. PyTorch Dataset 정의
class WebtoonDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        tokens = self.tokenizer(text, padding="max_length", truncation=True, max_length=self.max_len,
                                return_tensors="pt")

        input_ids = tokens["input_ids"].squeeze(0)  # (1, max_len) → (max_len)
        attention_mask = tokens["attention_mask"].squeeze(0)

        return input_ids, attention_mask, label


# ✅ 3. 추가 학습 가능하도록 구현
def continue_training(df, path, epochs_to_train=10):
    if os.path.exists(path):
        df_new = pd.read_excel(path, engine="openpyxl")
        df = pd.concat([df, df_new], ignore_index=True)  # ✅ 기존 데이터와 합치기
        print("✅ 새로운 데이터 추가 완료!")

    # ✅ 저장 경로 설정
    local_model_dir = os.path.expanduser("~/PycharmProjects/transformer/models")
    os.makedirs(local_model_dir, exist_ok=True)

    label_encoder_path = os.path.join(local_model_dir, "label_encoder.pkl")
    model_path = os.path.join(local_model_dir, "bert_webtoon_classifier.pth")

    tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")

    # ✅ 기존 라벨 인코더 로드 또는 새로 생성
    if os.path.exists(label_encoder_path):
        label_encoder = joblib.load(label_encoder_path)
        print(f"✅ 기존 라벨 인코더 로드 완료: {label_encoder_path}")
    else:
        label_encoder = LabelEncoder()
        print("🚀 새로운 라벨 인코더 생성")

    df["genre"] = df["genre"].apply(lambda x: eval(x)[0])
    df["genre_label"] = label_encoder.fit_transform(df["genre"])

    # ✅ 라벨 인코더 저장
    joblib.dump(label_encoder, label_encoder_path)
    print(f"✅ 라벨 인코더 저장 완료: {label_encoder_path}")

    # ✅ 데이터 분할
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        df["synopsis"].tolist(), df["genre_label"].tolist(), test_size=0.2, random_state=42
    )

    batch_size = 8  # ✅ 배치 크기 증가 (기존 4 → 8)
    accumulation_steps = 2  # ✅ Gradient Accumulation 조정 (기존 4 → 2)

    train_dataset = WebtoonDataset(train_texts, train_labels, tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    num_classes = len(label_encoder.classes_)

    # ✅ 모델 로드
    model = BERTClassifier(num_classes=num_classes).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs_to_train):
        model.train()
        total_loss = 0
        optimizer.zero_grad()

        for i, (input_ids, attention_mask, labels) in enumerate(train_loader):
            input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)

            with torch.cuda.amp.autocast(enabled=(device.type == "mps")):  # ✅ Mixed Precision 적용
                outputs = model(input_ids, attention_mask)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()  # ✅ Mixed Precision 스케일링

            if (i + 1) % accumulation_steps == 0:  # ✅ Gradient Accumulation 적용
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            total_loss += loss.item()

            # ✅ MPS 캐시 정리 (메모리 부족 방지) - 주기 줄임
            if device.type == "mps" and (i + 1) % 10 == 0:
                torch.mps.empty_cache()

        print(f"Epoch [{epoch + 1}/{epochs_to_train}], Loss: {total_loss / len(train_loader):.4f}")

        # ✅ 모델 저장
        torch.save(model.state_dict(), model_path)
        print(f"✅ 모델 저장 완료 (Epoch: {epoch + 1})")

    print("🎉 추가 학습 완료!")


# ✅ 실행
if __name__ == "__main__":
    text = "synopsis"
    label = "genre"
    fold = "model_data"

    df = pd.read_excel("../data/NAVER-Webtoon_OSMU.xlsx", engine="openpyxl")

    new_data_path = "../data/new_data.xlsx"
    continue_training(df, new_data_path, epochs_to_train=10)
