import streamlit as st
import torch
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import time

# Define waste categories and bins
class_names = ["Biodegradable", "Hazardous" ,"Non-Biodegradable"]
bin_colors = {"Biodegradable": "🟢 Green Bin","Hazardous": "🟠 Orange Bin", "Non-Biodegradable": "🔴 Red Bin"}

# Load trained EfficientNet model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.efficientnet_b0(pretrained=False)
num_classes = len(class_names)
model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, num_classes)
model.load_state_dict(torch.load("waste_classifier.pth", map_location=device))
model.to(device)
model.eval()

# Define image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Streamlit UI
st.title("♻️ Waste Classification & Sorting System")
st.write("Upload an image, and the system will classify it and move it into the correct bin.")

# Upload waste image
uploaded_file = st.file_uploader("Upload a waste image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)

    # Preprocess and classify
    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image_tensor)
        predicted_class_idx = torch.argmax(output, 1).item()

    predicted_class = class_names[predicted_class_idx]
    bin_color = bin_colors[predicted_class]

    # Simulate moving waste to the correct bin
    st.write("Classifying waste...")
    time.sleep(1)  # Delay for effect

    # Display bins
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("🟢 Green Bin (Biodegradable)")
        if predicted_class == "Biodegradable":
            st.image(image, caption="Moved to Green Bin", use_container_width=True)

    with col2:
        st.subheader("🔴 Red Bin (Non-Biodegradable)")
        if predicted_class == "Non-Biodegradable":
            st.image(image, caption="Moved to Red Bin", use_container_width=True)

    with col3:
        st.subheader("🟠 Orange Bin (Hazardous)")
        if predicted_class == "Hazardous":
            st.image(image, caption="Moved to Orange Bin", use_container_width=True)

    st.success(f"✅ Waste classified as **{predicted_class}** and moved into the {bin_color}.")
