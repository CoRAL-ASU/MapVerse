import os
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import random

okabe_ito_12 = [
    "#0072B2",  # blue
    "#E69F00",  # orange
    "#009E73",  # green
    "#D55E00",  # red
    "#CC79A7",  # pink
    "#F0E442",  # yellow
    "#56B4E9",  # sky blue
    "#999999",  # gray
    "#7E57C2",  # purple (extension)
    "#26A69A",  # teal (extension)
    "#8D6E63",  # brown (extension)
    "#B3E5FC"   # pale blue (extension)
]


from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.manifold import TSNE
from umap import UMAP
from sentence_transformers import SentenceTransformer
from torchvision import transforms
from torchvision.models import resnet18
import torch
import pickle

# ========== SETTINGS ==========
CSV_PATH = 'data/typed_questions.csv'
IMAGE_ROOT = 'data/imgs/'
USE_UMAP = False
PLOTS_DIR = 'plots_umap' if USE_UMAP else 'plots_tsne'
os.makedirs(PLOTS_DIR, exist_ok=True)

TEXT_EMB_FILE = 'tmp/text_embeddings.npy'
DF_TEXT_FILE = 'tmp/df_text.pkl'
IMAGE_EMB_FILE = 'tmp/image_embeddings.npy'
DF_IMAGE_FILE = 'tmp/df_valid.pkl'

# ========== LOAD DATA ==========
df = pd.read_csv(CSV_PATH)
df["map_type"] = df["map_type"].apply(lambda x: x[1:])
df["geographic_level"] = df["geographic_level"].apply(lambda x: x[1:])
df_image = df.dropna(subset=['image_name']).reset_index()

# ========== TEXT EMBEDDING ==========
if os.path.exists(TEXT_EMB_FILE) and os.path.exists(DF_TEXT_FILE):
    print("Loading saved text embeddings...")
    text_embeddings = np.load(TEXT_EMB_FILE)
    df_text = pd.read_pickle(DF_TEXT_FILE)
else:
    print("Computing text embeddings...")
    df_text = df.reset_index()
    text_model = SentenceTransformer('all-MiniLM-L6-v2')
    text_embeddings = text_model.encode(df_text['question'].tolist(), show_progress_bar=True)
    os.makedirs(os.path.dirname(TEXT_EMB_FILE), exist_ok=True)
    np.save(TEXT_EMB_FILE, text_embeddings)
    df_text.to_pickle(DF_TEXT_FILE)
    print("Saved text embeddings.")

# ========== IMAGE EMBEDDING ==========
if os.path.exists(IMAGE_EMB_FILE) and os.path.exists(DF_IMAGE_FILE):
    print("Loading saved image embeddings...")
    image_embeddings = np.load(IMAGE_EMB_FILE)
    df_valid = pd.read_pickle(DF_IMAGE_FILE)
else:
    print("Computing image embeddings...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    resnet = resnet18(pretrained=True)
    resnet.fc = torch.nn.Identity()
    resnet = resnet.to(device).eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    image_embeddings = []
    valid_rows = []
    for _, row in tqdm(df_image.iterrows(), total=len(df_image)):
        img_path = None
        for root, _, files in os.walk(IMAGE_ROOT):
            if row['image_name'] in files:
                img_path = os.path.join(root, row['image_name'])
                break
        if img_path is None:
            continue

        try:
            img = Image.open(img_path).convert("RGB")
            img_tensor = transform(img).unsqueeze(0).to(device)
            with torch.no_grad():
                feat = resnet(img_tensor).cpu().numpy().squeeze()
            image_embeddings.append(feat)
            valid_rows.append(row)
        except:
            continue

    image_embeddings = np.vstack(image_embeddings)
    df_valid = pd.DataFrame(valid_rows)
    os.makedirs(os.path.dirname(IMAGE_EMB_FILE), exist_ok=True)
    np.save(IMAGE_EMB_FILE, image_embeddings)
    df_valid.to_pickle(DF_IMAGE_FILE)
    print("Saved image embeddings.")

# ========== PLOTTING FUNCTION ==========
def reduce_and_plot(embeddings, labels, category, filename_prefix, legend_order=None):
    labels = labels.dropna()
    mask = ~labels.isna()
    embeddings = embeddings[mask.values]
    labels = labels[mask]

    lda = LinearDiscriminantAnalysis(
        n_components=min(len(labels.unique()) - 1, embeddings.shape[1])
    )
    embeddings_lda = lda.fit_transform(embeddings, labels)
    
    # embeddings_lda = embeddings

    if USE_UMAP:
        SEED = random.randint(1,100000)
        MIN_DIST = random.randint(1,500)/1000
        N_NEIGHBORS = random.randint(2,100)
        reducer = UMAP(n_components=2, n_neighbors=N_NEIGHBORS, min_dist=MIN_DIST, metric='cosine', random_state=SEED)
        reduced = reducer.fit_transform(embeddings_lda)
        reducer_name = "umap"
        reducer_params = {"n_neighbors": N_NEIGHBORS, "min_dist": MIN_DIST, "random_state": SEED}
    else:
        SEED = random.randint(1,100000)
        PERPLEXITY = 800
        reducer = TSNE(n_components=2, perplexity=PERPLEXITY, learning_rate=200, max_iter=1000, random_state=SEED)
        reduced = reducer.fit_transform(embeddings_lda)
        reducer_name = "tsne"
        reducer_params = {"perplexity": PERPLEXITY, "learning_rate": 200, "max_iter": 1000, "random_state": SEED}

    plt.figure(figsize=(14, 10))
    unique_labels = sorted(pd.Series(labels).unique()) if legend_order is None else legend_order
    palette = sns.color_palette("bright")
    palette = okabe_ito_12
    markers = ['x', '*', 'o', 's', 'D', '^', 'v', '<', '>', 'P', 'H', '+']

    for i, label in enumerate(unique_labels):
        idx = labels == label
        marker = markers[i % len(markers)]
        if marker not in ['x', '+']:
            plt.scatter(reduced[idx, 0], -reduced[idx, 1], s=150, facecolors='none',
                        edgecolors=palette[i % len(palette)], marker=marker, label=str(label).capitalize(), alpha=0.75)
        else:
            plt.scatter(reduced[idx, 0], -reduced[idx, 1], s=150,
                        edgecolors=palette[i % len(palette)], color=palette[i % len(palette)], marker=marker, label=str(label).capitalize(), alpha=0.75)

    plt.xticks([])
    plt.yticks([])
    plt.legend(loc='lower left', fontsize=22, framealpha=0.7)
    # plt.title(f"Answer Type", fontsize=22)
    plt.tight_layout()

    param_str = "_".join([f"{k}{v}" for k, v in reducer_params.items()])
    plot_path = f"{PLOTS_DIR}/{filename_prefix}_{category}_{reducer_name}_lda_{param_str}.png"
    plt.savefig(plot_path, bbox_inches='tight')
    plt.close()
    print(f"Saved plot: {plot_path}")

# ========== PLOTTING LOOP ==========
for i in range(1):
    reduce_and_plot(
        text_embeddings, df_text['answer_type'], 'answer_type', 'question',
        ['Boolean', 'Single Entity', 'Counting', 'Listing', 'Ranking', 'Reasoning']
    )
    reduce_and_plot(image_embeddings, df_valid['map_type'], 'map_type', 'images')
    # Uncomment if needed
    reduce_and_plot(image_embeddings, df_valid['geographic_level'], 'geographic_level', 'images', 
        ["building", "campus", "neighborhood", "district", "city", "county", "state", "region", "country", "subcontinent", "continent", "world"]
    )
