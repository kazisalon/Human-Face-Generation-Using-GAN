import os
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from flask import Flask, render_template, request, send_from_directory
from werkzeug.utils import secure_filename
import base64
from io import BytesIO
from PIL import Image

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/generated'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload

# Create upload folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Define the Generator architecture
class Generator(nn.Module):
    def __init__(self, latent_dim=100, channels=3):
        super(Generator, self).__init__()
        self.latent_dim = latent_dim
        
        self.init_size = 8  # Initial size before upsampling
        self.l1 = nn.Sequential(nn.Linear(latent_dim, 128 * self.init_size ** 2))
        
        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),  # 8x8 -> 16x16
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),  # 16x16 -> 32x32
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),  # 32x32 -> 64x64
            nn.Conv2d(64, 32, 3, stride=1, padding=1),
            nn.BatchNorm2d(32, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, channels, 3, stride=1, padding=1),
            nn.Tanh()
        )

    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img

# Initialize the generator
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
latent_dim = 100
generator = Generator(latent_dim).to(device)

# Load pre-trained weights if available
model_path = "generator.pth"
if os.path.exists(model_path):
    generator.load_state_dict(torch.load(model_path, map_location=device))
    generator.eval()
    print("Generator model loaded successfully")
else:
    print("No pre-trained model found. The generator will produce random outputs.")

def generate_face(seed=None):
    """Generate a face using the GAN model"""
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
    
    # Generate random latent vector
    z = torch.randn(1, latent_dim).to(device)
    
    # Generate image
    with torch.no_grad():
        img = generator(z)
        img = img.cpu().detach().squeeze(0)
    
    # Convert to PIL Image
    img = 0.5 * img + 0.5  # Scale from [-1,1] to [0,1]
    img = img.permute(1, 2, 0).numpy()
    img = (img * 255).astype(np.uint8)
    pil_img = Image.fromarray(img)
    
    return pil_img

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate():
    # Get seed from form if provided
    seed = request.form.get('seed')
    if seed:
        try:
            seed = int(seed)
        except ValueError:
            seed = None
    
    # Generate the face
    img = generate_face(seed)
    
    # Save the image
    filename = f"face_{np.random.randint(10000)}.png"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    img.save(filepath)
    
    # Convert image to base64 for displaying
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
    
    return render_template('result.html', 
                          image_data=img_str, 
                          filename=filename,
                          seed=seed if seed else "Random")

@app.route('/static/generated/<filename>')
def display_image(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)