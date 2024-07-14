# Intel-Unnati
Internship done in Intel Unnati Industrial training program about the problem statement detect pixelated image and correct it


import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from PIL import Image

# Define the Generator model
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=9, stride=1, padding=4),
            nn.PReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.PReLU(),
            nn.Conv2d(64, 3, kernel_size=9, stride=1, padding=4)
        )

    def forward(self, x):
        return self.main(x)

# Define the Discriminator model
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Flatten(),
            nn.Linear(64 * 16 * 16, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.main(x)


# Define loss functions
adversarial_loss = nn.BCELoss()
pixel_loss = nn.L1Loss()

# Initialize models
generator = Generator()
discriminator = Discriminator()

# Define optimizers
optimizer_G = optim.Adam(generator.parameters(), lr=0.0002)
optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0002)


# Preprocessing transformations
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Load datasets
train_dataset = datasets.ImageFolder(root='path_to_train_dataset', transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True)


num_epochs = 100

for epoch in range(num_epochs):
    for i, (imgs, _) in enumerate(train_loader):
        # Prepare inputs
        real_imgs = imgs
        low_res_imgs = transforms.Resize((16, 16))(imgs)
        low_res_imgs = transforms.Resize((64, 64))(low_res_imgs)

        # Adversarial ground truths
        valid = torch.ones(imgs.size(0), 1)
        fake = torch.zeros(imgs.size(0), 1)

        # ---------------------
        #  Train Discriminator
        # ---------------------
        optimizer_D.zero_grad()

        real_loss = adversarial_loss(discriminator(real_imgs), valid)
        fake_imgs = generator(low_res_imgs)
        fake_loss = adversarial_loss(discriminator(fake_imgs.detach()), fake)

        d_loss = (real_loss + fake_loss) / 2
        d_loss.backward()
        optimizer_D.step()

        # -----------------
        #  Train Generator
        # -----------------
        optimizer_G.zero_grad()

        g_loss = adversarial_loss(discriminator(fake_imgs), valid) + pixel_loss(fake_imgs, real_imgs)
        g_loss.backward()
        optimizer_G.step()

        print(f"[Epoch {epoch+1}/{num_epochs}] [Batch {i+1}/{len(train_loader)}] [D loss: {d_loss.item()}] [G loss: {g_loss.item()}]")

# Save models
torch.save(generator.state_dict(), 'generator.pth')
torch.save(discriminator.state_dict(), 'discriminator.pth')



# Load trained generator model
generator = Generator()
generator.load_state_dict(torch.load('generator.pth'))
generator.eval()

# Load and preprocess a pixelated image
def load_image(image_path):
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0)  # Add batch dimension
    return image

def save_image(tensor, path):
    image = tensor.squeeze(0).detach()
    image = transforms.ToPILImage()(image)
    image.save(path)

# Correct pixelated image
pixelated_image = load_image('path_to_pixelated_image')
with torch.no_grad():
    high_res_image = generator(pixelated_image)
save_image(high_res_image, 'path_to_save_high_res_image')
