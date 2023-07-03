import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import normalize
from torch.autograd import Variable
import torch.optim as optim
import numpy as np
from imageio import imread
from PIL import Image
from tokenizer import *
from dalle2_pytorch import DALLE2, DiffusionPriorNetwork, DiffusionPrior, Unet, Decoder, CLIP, DecoderTrainer, DiffusionPriorTrainer, OpenAIClipAdapter
from transformers import VisionEncoderDecoderModel, ViTFeatureExtractor, AutoTokenizer
from PIL import Image
import os
from transformers import BartForConditionalGeneration, BartTokenizer
import time

def time_elapsed(begin, end):
    elapsed = end - begin
    h, over = divmod(end - begin, 3600)
    m, s = divmod(over, 60)
    # print("{:0>2}h:{:0>2}m:{:05.2f}s".format(int(h), int(m), s))
    return (elapsed, "{:0>2}h:{:0>2}m:{:05.2f}s".format(int(h), int(m), s))

"""We will generate captions in order to train the DALLE-2 generator here"""
def predict_step(images):
    pixel_values = feature_extractor(images=images, return_tensors="pt").pixel_values
    output_ids = model.generate(pixel_values, **gen_kwargs)
    preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    preds = [pred.strip() for pred in preds]
    return preds

filenames = os.listdir('ab all/')
images = []
for i in range(len(filenames)):
    if filenames[i] != "ab all labels":
        image_path = 'ab all/' + filenames[i]
        i_image = Image.open(image_path)
        if i_image.mode != "RGB":
            i_image = i_image.convert(mode="RGB")
        images.append(i_image)

model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
feature_extractor = ViTFeatureExtractor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
max_length, num_beams = 32, 4
gen_kwargs = {"max_length": max_length, "num_beams": num_beams}

# captions = predict_step(images)
f = open("captions.txt", "r")
captions = []
for line in f:
   captions.append(line.strip())
f.close()


# Generator class for both GAN and CEM
class Dalle2_generator():
    def __init__(self):
        # Instantiate CLIP text encoder
        self.clip = OpenAIClipAdapter()

        # Instantiate diffusion prior transformer
        self.prior_network = DiffusionPriorNetwork(
            dim = 512,
            depth = 6,
            dim_head = 64,
            heads = 8
        )
        self.diffusion_prior = DiffusionPrior(
            net = self.prior_network,
            clip = self.clip,
            timesteps = 1000,
            sample_timesteps = 64,
            cond_drop_prob = 0.2
        )

        # Instantiate diffusion model using 2 UNETS
        self.unet1 = Unet(
            dim = 128,
            image_embed_dim = 512,
            text_embed_dim = 512,
            cond_dim = 128,
            channels = 3,
            dim_mults=(1, 2, 4, 8),
            cond_on_text_encodings = True    
        )
        self.unet2 = Unet(
            dim = 16,
            image_embed_dim = 512,
            cond_dim = 128,
            channels = 3,
            dim_mults = (1, 2, 4, 8, 16)
        )
        self.decoder = Decoder(
            unet = (self.unet1, self.unet2),
            image_sizes = (128, 256),
            clip = self.clip,
            timesteps = 100,
            image_cond_drop_prob = 0.1,
            text_cond_drop_prob = 0.5
        )

        # Instantiate the overall DALLE-2 architecture
        self.dalle2 = DALLE2(
            prior = self.diffusion_prior,
            decoder = self.decoder
        )

        # Trainer wrappers
        self.diffusion_prior_trainer = DiffusionPriorTrainer(
            self.diffusion_prior,
            lr = 1e-4,
            wd = 1e-2,
            ema_beta = 0.99,
            ema_update_after_step = 1000,
            ema_update_every = 10,
        )
        self.decoder_trainer = DecoderTrainer(
            self.decoder,
            lr = 2e-4, # previously 3e-4
            wd = 1e-2, 
            ema_beta = 0.99,
            ema_update_after_step = 1000,
            ema_update_every = 10,
        )

    def pretrain(self, text, images, num_iter_prior, num_iter_decoder):
        # Pretrain the diffusion prior
        print("Pretraining generator...")
        for i in range(num_iter_prior):
            for param in generator.prior_network.parameters():
                param.requires_grad = True

            loss = self.diffusion_prior_trainer(text, images, max_batch_size=4)
            print("{}: Prior loss: {}".format(i, loss))
            self.diffusion_prior_trainer.update() 
        print("Done. Pretraining decoder...")
        # Pretrain the diffusion decoder
        for i in range(num_iter_decoder):
            print("iteration:", i)
            for unet_number in (1, 2):
                if unet_number == 1:
                    for param in self.unet1.parameters():
                        param.requires_grad = True
                    for param in self.unet2.parameters():
                        param.requires_grad = False
                if unet_number == 2:
                    for param in self.unet1.parameters():
                        param.requires_grad = False
                    for param in self.unet2.parameters():
                        param.requires_grad = True

                loss = self.decoder_trainer(images, text=text, unet_number=unet_number, max_batch_size=4)
                print("UNet:", unet_number, " Decoder loss:", loss)
                self.decoder_trainer.update(unet_number)


class BART():
    def __init__(self):
        self.tokenizer = BartTokenizer.from_pretrained('eugenesiow/bart-paraphrase')
        self.model = BartForConditionalGeneration.from_pretrained('eugenesiow/bart-paraphrase')
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(device)

    def rephrase(self, captions):
        batch = self.tokenizer(captions, return_tensors='pt', padding=True)
        generated_ids = self.model.generate(batch['input_ids'].cuda())
        generated_sentence = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        return generated_sentence


"""**TRAINING LOOP BASED ON GAN OPTIMIZATION**"""
# Discriminator class for GAN
class CNN_discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=3, 
                      out_channels=16, 
                      kernel_size=(4, 4)),
            nn.MaxPool2d((4, 4)),
            nn.Conv2d(in_channels=16, 
                      out_channels=32, 
                      kernel_size=(4, 4)),
            nn.MaxPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(7200, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        ).cuda()
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-4)
        self.bceloss = nn.BCELoss()
    
    def init_weights(self):
        if type(self.model) == nn.Conv2d:
            torch.nn.init.kaiming_normal_(self.model.weight, mode='fan_in')
        elif type(self.model) == nn.Linear:
            torch.nn.init.kaiming_normal_(self.model.weight, mode='fan_in')
            self.model.bias.data.fill_(0.01)

    def forward(self, image):
        prediction = self.model(image)
        return prediction

    def pretrain(self, X, y, Xval, yval, epoch):
        self.init_weights()
        for i in range(epoch):
            batchtrainx = X
            batchtrainy = y
            batchvalx = Xval
            batchvaly = yval
            
            # Pass images through discriminator and backpropagate the loss
            self.optimizer.zero_grad()
            pred = self.model(batchtrainx)
            loss = self.bceloss(pred.float(), batchtrainy.float())
            loss.backward()
            self.optimizer.step()

            # Update training accuracy
            train_acc = torch.sum(torch.round(pred) == batchtrainy) / batchtrainx.shape[0]

            # Get validation loss and validation accuracy
            predval = self.model(batchvalx)
            valloss = self.bceloss(predval.float(), batchvaly.float())
            val_acc = torch.sum(torch.round(predval) == batchvaly) / batchvalx.shape[0]

            print("Train Loss: " + str(loss) + "   Train Accuracy: " + str(train_acc) + "-- Validation Loss: " + str(valloss) + "   Validation Accuracy: " + str(val_acc) + "   Epoch: " + str(i))


class GAN():
    def __init__(self, pretrained_generator, pretrained_discriminator):
        self.bart = BART()
        self.generator = pretrained_generator
        self.discriminator = pretrained_discriminator

    def discriminator_loss(self, real_output, fake_output):
        bceloss = nn.BCELoss()
        real_loss = bceloss(real_output, torch.ones_like(real_output))
        fake_loss = bceloss(fake_output, torch.zeros_like(fake_output))
        total_loss = real_loss + fake_loss
        return total_loss

    def generator_loss(self, fake_output):
        bceloss = nn.BCELoss()
        total_loss = bceloss(torch.ones_like(fake_output), fake_output)
        return total_loss

    def train_discriminator(self, real_output, fake_output):
        self.discriminator.optimizer.zero_grad()
        disc_loss = self.discriminator_loss(real_output, fake_output)
        print("Discriminator loss: ", disc_loss)
        disc_loss.backward()
        self.discriminator.optimizer.step()
    
    def train_generator(self, fake_output):  
        # DO NOT TRAIN DALLE-2, ONLY TRAIN BART MODEL AND FREEZE OTHER WEIGHTS
        for param in self.bart.model.parameters():
            param.requires_grad = True
        for param in self.generator.clip.parameters():
            param.requires_grad = False
        for param in self.generator.prior_network.parameters():
            param.requires_grad = False
        for param in self.generator.unet1.parameters():
            param.requires_grad = False
        for param in self.generator.unet2.parameters():
            param.requires_grad = False

        gen_loss = self.generator_loss(fake_output)
        print("Generator loss: ", gen_loss)
        gen_loss = Variable(gen_loss, requires_grad = True)
        gen_loss.backward()

        # How to optimize BART parameters?
        # self.generator_optimizer = optim.Adam(
        #     self.discriminator.parameters(),
        #     lr=0.00001,
        # )

    def train(self, captions, real_images, epoch=100):
        for i in range(epoch):
            print("Epoch number:", i)
            # Run captions through BART and RUN through generator
            # cond_scale is classifier free guidance strength (> 1 would strengthen the condition)
            captions = self.bart.rephrase(captions)
            generated_images = self.generator.dalle2(captions, cond_scale = 2.)     

            # Run real and generated images through through discriminator 
            real_output = self.discriminator(real_images)
            fake_output = self.discriminator(generated_images)

            # Train the discriminator and generator
            self.train_discriminator(real_output, fake_output)
            self.train_generator(fake_output)

        torch.save(self.bart.state_dict(), '/home/ubuntu/project/models/generator')


# Load dataset
tokenizer = SimpleTokenizer()       
tokenized_captions = tokenizer.tokenize(captions)
images, good_images, bad_images, good_captions, bad_captions, good_tokens, bad_tokens = [], [], [], [], [], [], []
targets = open('ab all labels').readlines()

count = 0
#print(len(targets))
for i in range(len(filenames)):
    if filenames[i] != "ab all labels":
        image = imread('ab all/' + filenames[i])
        targets[i-count] = int(targets[i-count].strip())
        if image.ndim == 3:
            images.append(np.asarray(Image.fromarray(image).resize((256, 256))))
        else:
            captions.pop(i- count)
            targets.pop(i - count)
            count += 1

print('lengths :', len(images), len(targets), len(captions))
print("Limiting dataset...")
images = images[-8:]
targets = targets[-8:]
captions = captions[-8:]
print('lengths :', len(images), len(targets), len(captions))
images = torch.tensor(np.transpose(np.array(images),[0,3,1,2]), dtype=torch.float)   
norm_images = normalize(images)

for i in range(len(norm_images)):
    if targets[i] == 1:
        good_images.append(norm_images[i].unsqueeze(0))
        good_captions.append(captions[i])
        good_tokens.append(tokenized_captions[i].unsqueeze(0))
    else:
        bad_images.append(norm_images[i].unsqueeze(0))
        bad_captions.append(captions[i])
        bad_tokens.append(tokenized_captions[i].unsqueeze(0))

good_images, bad_images = torch.cat(good_images), torch.cat(bad_images)
good_tokens, bad_tokens = torch.cat(good_tokens), torch.cat(bad_tokens)

# Create dataset
train_size = 4 * len(norm_images) // 5
images_train, images_val = norm_images[:train_size], norm_images[train_size:]
targets = torch.tensor(targets, dtype=torch.float)
targets_train, targets_val = targets[:train_size].unsqueeze(1), targets[train_size:].unsqueeze(1)



""" Pretrain DALLE-2 generator on good images and good captions only!
Pretrain CNN discriminator on both good and bad images! 
Train BART with GAN loop on bad images only! """

print("\nStart pretraining generator!")
generator = Dalle2_generator() 
generator.prior_network.training = True
generator.unet1.training = True
generator.unet2.training = True
gen_pre_start = time.time()
generator.pretrain(good_tokens.cuda(), good_images.cuda(), num_iter_prior=150, num_iter_decoder=250)
gen_pre_end = time.time()
gen_pre_time = time_elapsed(gen_pre_start, gen_pre_end)
print("Done pretraining generator! Time elapsed: ", gen_pre_time[1])

print("\nStart pretraining discriminator!")
discriminator = CNN_discriminator()
disc_pre_start = time.time()
discriminator.pretrain(images_train.cuda(), targets_train.cuda(), images_val.cuda(), targets_val.cuda(), epoch=100)
disc_pre_end = time.time()
disc_pre_time = time_elapsed(disc_pre_start, disc_pre_end)
print("Done pretraining discriminator! Time elapsed: ", disc_pre_time[1])

print("\nStart training GAN loop!")
gan_trainer = GAN(generator, discriminator)
gan_start = time.time()
gan_trainer.train(bad_captions, good_images.cuda(), epoch=100)
gan_end = time.time()
gan_time = time_elapsed(gan_start, gan_end)
print("Saving best models...")
torch.save(discriminator.state_dict(), '/home/ubuntu/project/models/discrim')
torch.save(generator.prior_network.state_dict(), '/home/ubuntu/project/models/prior_network')
torch.save(generator.unet1.state_dict(), '/home/ubuntu/project/models/unet1')
torch.save(generator.unet2.state_dict(), '/home/ubuntu/project/models/unet2')
print("Done training GAN loop! Time elapsed: ", gan_time[1])

total_time = gen_pre_time[0] + disc_pre_time[0] + gan_time[0]
total_elapsed = time_elapsed(0, total_time)
print("Times elapsed: \
        generator pretraining - {}, \
        discriminator pretraining - {}, \
        GAN training - {}, \
        total time - ".format(gen_pre_time[1], disc_pre_time[1], gan_time[1], total_elapsed[1]))