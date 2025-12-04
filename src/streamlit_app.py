import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt

# 페이지 설정
st.set_page_config(page_title="ASL 수화 분류기", layout="wide")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 스타일 설정
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
    }
    </style>
    """, unsafe_allow_html=True)

# 제목 및 설명
st.title("ASL(American Sign Language) 분류기")
st.markdown("학습된 PyTorch 모델을 사용하여 수화 알파벳 이미지를 분류합니다.")

# 1. 모델 클래스 정의 (Notebook과 동일하게 BatchNorm 포함)
class ASLClassifier(nn.Module):
    def __init__(self, input_size=784, num_classes=24):
        super(ASLClassifier, self).__init__()
        
        self.fc1 = nn.Linear(input_size, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.dropout1 = nn.Dropout(0.3)
        
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.dropout2 = nn.Dropout(0.3)
        
        self.fc3 = nn.Linear(256, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.dropout3 = nn.Dropout(0.2)
        
        self.fc4 = nn.Linear(128, num_classes)
        
        self.relu = nn.ReLU()

    def forward(self, x):
        # x shape: (batch, 784)
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout1(x)

        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.dropout2(x)

        x = self.fc3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.dropout3(x)

        x = self.fc4(x)
        return x

# 2. 모델 로드 함수
@st.cache_resource
def load_model():
    # notebook에서도 device = 'cuda' if ... 이런 식이라 그대로 맞춤
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ASLClassifier().to(device)
    model_path = os.path.join(BASE_DIR, "model", "best_nnLinear_model.pth")
    
    if os.path.exists(model_path):
        try:
            # 🔥 map_state ❌ → map_location ✅
            state = torch.load(model_path, map_location=device)
            
            # 학습 때와 구조가 동일하므로 strict=True로 로딩해도 됨
            missing, unexpected = model.load_state_dict(state, strict=False)
            # 디버깅 원하면 아래 로그 켜도 됨
            # print("missing keys:", missing)
            # print("unexpected keys:", unexpected)

            model.eval()
            return model, device
        except Exception as e:
            st.error(f"모델 로드 중 오류 발생: {e}")
            return None, device
    else:
        st.error(f"모델 파일을 찾을 수 없습니다: {model_path}")
        return None, device

model, device = load_model()

# 3. 이미지 전처리 함수 (Notebook의 test_transform과 동일하게 ToTensor만 사용)
def process_image(image):
    # Grayscale 변환
    image = image.convert('L')
    # 28x28 리사이즈
    image = image.resize((28, 28))
    
    # Notebook의 test_transform = transforms.ToTensor()
    transform = transforms.Compose([
        transforms.ToTensor(),
        # Normalize 안 썼으므로 그대로 두기
    ])
    
    image_tensor = transform(image)  # (1, 28, 28)
    # Flatten (1, 784)
    image_tensor = image_tensor.view(1, -1)
    
    return image_tensor, image

# 4. 예측 함수
def predict(model, image_tensor, device):
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        outputs = model(image_tensor)
        probs = torch.softmax(outputs, dim=1)
        
        # 상위 3개 예측
        top3_prob, top3_idx = torch.topk(probs, 3)
        
    return top3_prob.cpu().numpy()[0], top3_idx.cpu().numpy()[0]

# 레이블 매핑 (J, Z 제외)
# Notebook 기준: 0~23 → A~Y (J, Z 제외) 구조와 동일
label_to_letter = {i: chr(65+i) if i < 9 else chr(65+i+1) for i in range(24)}

# --- UI 구성 ---

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("1. 이미지 선택")
    option = st.radio("이미지 소스 선택:", ("샘플 이미지", "이미지 업로드"))
    
    input_image = None
    
    if option == "샘플 이미지":
        sample_dir = os.path.join(BASE_DIR, "data", "asl_image")
        if os.path.exists(sample_dir):
            sample_files = [f for f in os.listdir(sample_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            selected_sample = st.selectbox("샘플 이미지를 선택하세요:", sample_files)
            
            if selected_sample:
                image_path = os.path.join(sample_dir, selected_sample)
                input_image = Image.open(image_path)
                st.image(input_image, caption=f"선택된 샘플: {selected_sample}", width=300)
        else:
            st.warning("샘플 이미지 디렉토리를 찾을 수 없습니다.")
            
    else:  # 이미지 업로드
        uploaded_file = st.file_uploader("이미지 파일을 업로드하세요", type=['png', 'jpg', 'jpeg'])
        if uploaded_file is not None:
            input_image = Image.open(uploaded_file)
            st.image(input_image, caption="업로드된 이미지", width=300)

with col2:
    st.subheader("2. 분석 결과")
    
    if input_image is not None and model is not None:
        if st.button("분석 시작", type="primary"):
            with st.spinner('분석 중...'):
                # 전처리
                img_tensor, processed_img = process_image(input_image)
                
                # 예측
                top3_prob, top3_idx = predict(model, img_tensor, device)
                
                # 결과 표시
                top1_letter = label_to_letter.get(int(top3_idx[0]), '?')
                top1_conf = float(top3_prob[0] * 100)
                
                st.success(f"예측 결과: **{top1_letter}**")
                st.metric(label="신뢰도 (Confidence)", value=f"{top1_conf:.2f}%")
                
                # 전처리된 이미지 확인 (디버깅용)
                with st.expander("전처리된 입력 이미지 보기 (28x28 Grayscale)"):
                    st.image(processed_img, width=100)
                
                # 상위 3개 확률 시각화
                st.markdown("### 상위 3개 예측 확률")
                
                chart_data = pd.DataFrame({
                    'Alphabet': [label_to_letter.get(int(idx), '?') for idx in top3_idx],
                    'Probability': top3_prob * 100
                })
                
                # 막대 그래프
                st.bar_chart(chart_data.set_index('Alphabet'))
                
                # 상세 표
                st.table(chart_data.assign(Probability=lambda x: x['Probability'].map('{:.2f}%'.format)))
                
    elif model is None:
        st.error("모델이 로드되지 않았습니다.")
    else:
        st.info("이미지를 선택하면 분석을 시작할 수 있습니다.")

# 사이드바 정보
with st.sidebar:
    st.header("모델 정보")
    st.info("""
    - **모델 구조**: MLP (Linear + BatchNorm + Dropout)
    - **입력**: 28x28 Grayscale Image (Flattened to 784)
    - **출력**: 24 Classes (A-Y, excluding J, Z)
    - **학습 데이터**: Sign Language MNIST
    """)
    st.markdown("---")
    st.markdown("Created with Streamlit & PyTorch")
