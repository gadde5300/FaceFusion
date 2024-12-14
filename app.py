import streamlit as st
import torch
from torch import nn
from diffusers import DDPMScheduler, UNet2DModel
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

# Reuse your existing model code
class ClassConditionedUnet(nn.Module):
    def __init__(self, num_classes=3, class_emb_size=12):
        super().__init__()
        self.class_emb = nn.Embedding(num_classes, class_emb_size)
        self.model = UNet2DModel(
            sample_size=64,
            in_channels=3 + class_emb_size,
            out_channels=3,
            layers_per_block=2,
            block_out_channels=(64, 128, 256, 512),
            down_block_types=(
                "DownBlock2D",
                "DownBlock2D",
                "AttnDownBlock2D",
                "AttnDownBlock2D",
            ),
            up_block_types=(
                "AttnUpBlock2D",
                "AttnUpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
            ),
        )

    def forward(self, x, t, class_labels):
        bs, ch, w, h = x.shape
        class_cond = self.class_emb(class_labels)
        class_cond = class_cond.view(bs, class_cond.shape[1], 1, 1).expand(bs, class_cond.shape[1], w, h)
        net_input = torch.cat((x, class_cond), 1)
        return self.model(net_input, t).sample

@st.cache_resource
def load_model(model_path):
    """Load the model with caching to avoid reloading"""
    device = 'cpu'  # For deployment, we'll use CPU
    net = ClassConditionedUnet().to(device)
    noise_scheduler = DDPMScheduler(num_train_timesteps=1000, beta_schedule='squaredcos_cap_v2')
    checkpoint = torch.load(model_path, map_location='cpu')
    net.load_state_dict(checkpoint['model_state_dict'])
    return net, noise_scheduler

def generate_mixed_faces(net, noise_scheduler, mix_weights, num_images=1):
    """Generate faces with mixed ethnic features"""
    device = next(net.parameters()).device
    net.eval()
    with torch.no_grad():
        x = torch.randn(num_images, 3, 64, 64).to(device)
        
        # Get embeddings for all classes
        emb_asian = net.class_emb(torch.zeros(num_images).long().to(device))
        emb_indian = net.class_emb(torch.ones(num_images).long().to(device))
        emb_european = net.class_emb(torch.full((num_images,), 2).to(device))
        
        progress_bar = st.progress(0)
        for idx, t in enumerate(noise_scheduler.timesteps):
            # Update progress bar
            progress_bar.progress(idx / len(noise_scheduler.timesteps))
            
            # Mix embeddings according to weights
            mixed_emb = (
                mix_weights[0] * emb_asian +
                mix_weights[1] * emb_indian +
                mix_weights[2] * emb_european
            )
            
            # Override embedding layer temporarily
            original_forward = net.class_emb.forward
            net.class_emb.forward = lambda _: mixed_emb
            
            residual = net(x, t, torch.zeros(num_images).long().to(device))
            x = noise_scheduler.step(residual, t, x).prev_sample
            
            # Restore original embedding layer
            net.class_emb.forward = original_forward
        
        progress_bar.progress(1.0)
    
    x = (x.clamp(-1, 1) + 1) / 2
    return x

def main():
    st.title("AI Face Generator with Ethnic Features Mixing")
    
    # Load model
    try:
        net, noise_scheduler = load_model('final_model/final_diffusion_model.pt')
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return
    
    # Create sliders for ethnicity percentages
    st.subheader("Adjust Ethnicity Mix")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        asian_pct = st.slider("Oriental Features %", 0, 100, 33, 1)
    with col2:
        indian_pct = st.slider("Indian Features %", 0, 100, 33, 1)
    with col3:
        european_pct = st.slider("European Features %", 0, 100, 34, 1)
    
    # Calculate total and normalize if needed
    total = 100
    if total == 0:
        st.warning("Total percentage cannot be 0%. Please adjust the sliders.")
        return
    
    # Normalize weights to sum to 1
    weights = [asian_pct/total, indian_pct/total, european_pct/total]
    
    # Display current mix
    st.write("Current mix (normalized):")
    st.write(f"Oriental: {weights[0]:.2%}, Indian: {weights[1]:.2%}, European: {weights[2]:.2%}")
    
    # Generate button
    if st.button("Generate Face"):
        try:
            with st.spinner("Generating face..."):
                # Generate the image
                generated_images = generate_mixed_faces(net, noise_scheduler, weights)
                
                # Convert to numpy and display
                img = generated_images[0].permute(1, 2, 0).cpu().numpy()
                st.image(img, caption="Generated Face", use_container_width =True)
                
        except Exception as e:
            st.error(f"Error generating image: {str(e)}")

if __name__ == "__main__":
    main()