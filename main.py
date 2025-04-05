import cv2
import os
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.vgg16 import preprocess_input
from sklearn.model_selection import train_test_split
import joblib
import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

class FaceRecognitionModel:
    def __init__(self, data_path="images/originalimages_part1"):
        self.sift = cv2.SIFT_create(nfeatures=300)
        self.cnn_model = VGG16(weights='imagenet', include_top=False, pooling='avg')
        self.pca = PCA(n_components=0.95)
        self.scaler = StandardScaler()
        self.knn = KNeighborsClassifier(n_neighbors=5)
        self.data_path = data_path
        self.X_train, self.X_test, self.y_train, self.y_test = self.load_data()
        self.train_accuracy = None
        self.test_accuracy = None
        self.classification_rep = None
        self.confusion_mat = None

    def load_data(self):
        images = []
        labels = []
        
        for filename in sorted(os.listdir(self.data_path)):
            if filename.lower().endswith('.jpg'):
                person_id = filename.split('-')[0]
                img_path = os.path.join(self.data_path, filename)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    img = cv2.resize(img, (128, 128))
                    images.append(img)
                    labels.append(person_id)
        
        return train_test_split(
            np.array(images), np.array(labels),
            test_size=0.2,
            stratify=labels,
            random_state=42
        )

    def extract_features(self, images):
        features = []
        for img in images:
            # SIFT Features
            _, des = self.sift.detectAndCompute(img, None)
            sift_feat = des.mean(axis=0) if des is not None else np.zeros(128)
            
            # CNN Features
            img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            img_resized = cv2.resize(img_color, (224, 224))
            img_array = img_to_array(img_resized)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = preprocess_input(img_array)
            cnn_feat = self.cnn_model.predict(img_array, verbose=0).flatten()
            
            features.append(np.concatenate([sift_feat, cnn_feat]))
        
        return np.array(features)

    def train(self):
        # Feature extraction
        train_features = self.extract_features(self.X_train)
        test_features = self.extract_features(self.X_test)
        
        # Preprocessing
        self.scaler.fit(train_features)
        train_features_scaled = self.scaler.transform(train_features)
        test_features_scaled = self.scaler.transform(test_features)
        
        # Dimensionality reduction
        self.pca.fit(train_features_scaled)
        train_features_pca = self.pca.transform(train_features_scaled)
        test_features_pca = self.pca.transform(test_features_scaled)
        
        # Training
        self.knn.fit(train_features_pca, self.y_train)
        
        # Evaluation
        self.train_accuracy = self.knn.score(train_features_pca, self.y_train)
        self.test_accuracy = self.knn.score(test_features_pca, self.y_test)
        
        # Detailed metrics
        y_pred = self.knn.predict(test_features_pca)
        self.classification_rep = classification_report(self.y_test, y_pred, output_dict=True)
        self.confusion_mat = confusion_matrix(self.y_test, y_pred)
        
        return self.test_accuracy

    def predict_face(self, img_array):
        img_gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY) if len(img_array.shape) == 3 else img_array
        img_resized = cv2.resize(img_gray, (128, 128))
        
        features = self.extract_features([img_resized])
        features_scaled = self.scaler.transform(features)
        features_pca = self.pca.transform(features_scaled)
        
        label = self.knn.predict(features_pca)[0]
        confidence = np.max(self.knn.predict_proba(features_pca))
        return label, confidence

    def get_person_images(self, person_id):
        images = []
        for filename in sorted(os.listdir(self.data_path)):
            if filename.startswith(f"{person_id}-") and filename.lower().endswith('.jpg'):
                img_path = os.path.join(self.data_path, filename)
                images.append(img_path)
        return images[:14]

# Interface Streamlit
def main():
    st.set_page_config(layout="wide")
    st.title("🔍 Système de Reconnaissance Faciale - Dashboard Complet")
    
    # Initialisation du modèle
    if 'model' not in st.session_state:
        with st.spinner("Initialisation du modèle..."):
            model = FaceRecognitionModel()
            model.train()
            st.session_state.model = model
    
    # Sidebar avec les statistiques
    with st.sidebar:
        st.header("📊 Statistiques du Modèle")
        
        if st.session_state.model.train_accuracy:
            st.metric("Précision sur l'entraînement", 
                     f"{st.session_state.model.train_accuracy:.2%}")
            st.metric("Précision sur le test", 
                     f"{st.session_state.model.test_accuracy:.2%}")
            
            st.subheader("Rapport de Classification")
            class_report_df = pd.DataFrame(st.session_state.model.classification_rep).transpose()
            st.dataframe(class_report_df.style.highlight_max(axis=0))
            
            st.subheader("Matrice de Confusion")
            fig, ax = plt.subplots()
            sns.heatmap(st.session_state.model.confusion_mat, annot=True, fmt='d', ax=ax)
            st.pyplot(fig)
    
    # Section principale
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎯 Identification Faciale")
        uploaded_file = st.file_uploader("Téléversez une photo de visage", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        img_array = np.array(image)
        
        with col1:
            st.image(image, caption="Image téléversée", use_column_width=True)
        
        # Prédiction
        label, confidence = st.session_state.model.predict_face(img_array)
        
        with col2:
            st.success(f"**Personne identifiée : {label}**")
            st.info(f"**Confiance : {confidence:.2%}**")
            
            # Distribution des probabilités
            st.subheader("Distribution des Probabilités")
            img_resized = cv2.resize(cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY), (128, 128))
            features = st.session_state.model.extract_features([img_resized])
            features_scaled = st.session_state.model.scaler.transform(features)
            features_pca = st.session_state.model.pca.transform(features_scaled)
            
            probas = st.session_state.model.knn.predict_proba(features_pca)[0]
            proba_df = pd.DataFrame({
                "Personne": st.session_state.model.knn.classes_,
                "Probabilité": probas
            }).sort_values("Probabilité", ascending=False)
            
            fig, ax = plt.subplots()
            sns.barplot(x="Probabilité", y="Personne", data=proba_df.head(5), ax=ax)
            st.pyplot(fig)
        
        # Galerie des images
        st.subheader(f"📸 Galerie de référence - Personne {label}")
        person_images = st.session_state.model.get_person_images(label)
        
        cols = st.columns(7)
        for idx, img_path in enumerate(person_images):
            img = Image.open(img_path)
            cols[idx%7].image(img, use_column_width=True, caption=f"Image {idx+1}")

if __name__ == "__main__":
    main()