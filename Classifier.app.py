import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

# --- MODEL DEFINITIE UIT JOUW NOTEBOOK ---
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3 * 224 * 224, 512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, 4)
        )

    def forward(self, x):
        return self.model(x)

model = MLP()

# Optioneel: als je een eerder getraind modelbestand wilt inladen:
# model.load_state_dict(torch.load("model.pth", map_location=torch.device('cpu')))

# Voor demo: random weights (vervang dit met echte weights als je hebt)
model.eval()

# --- TRANSFORMATIES ---
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# --- STREAMLIT UI ---
st.title("X-ray Classificatie App")
st.write("Upload een X-ray afbeelding. Het model classificeert in 4 klassen:")

classes = ['Too big', 'over exposed', 'Too small', 'Under exposed']
st.write(f"Klassen: {', '.join(classes)}")

uploaded_file = st.file_uploader("Kies een afbeelding", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Laad afbeelding
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Geüploade afbeelding", use_column_width=True)

    # Transformeer
    input_tensor = transform(img).unsqueeze(0)

    # Voorspellen
    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.sigmoid(output).squeeze().numpy()

    # Plot
    fig, ax = plt.subplots()
    bar_colors = ['red' if p > 0.5 else 'blue' for p in probs]
    ax.bar(classes, probs, color=bar_colors)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Waarschijnlijkheid")
    ax.set_title("Classificatie Resultaten")
    st.pyplot(fig)

    # Laat ruwe tensorwaarden zien
    st.code(f"Model output (ruwe tensor):\n{output}")
