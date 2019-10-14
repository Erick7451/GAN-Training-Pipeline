import torch
import numpy as np
import pandas as pd


def train_GAN(netD,netG,optimizerD,optimizerG,criterion, num_epochs, dataloader):

    # Establish convention for real and fake labels during training
    real_label = 1
    fake_label = 0

    fixed_noise = torch.randn(64, nz, 1, 1, device=device)

    # Lists to keep track of progress
    img_list = []

    # Network losses
    D_x_losses = []
    D_G_z1_losses = []
    D_G_z2_losses = []
    total_D_losses = [] # equals sum of D_x_losses + D_G_z1_losses

    # Training Accuracy
    D_x_acc = []
    D_G_z1_acc = []
    D_G_z2_acc = []

    # Top-Bottom gradients
    D_x_bottom_top_grads = []
    D_G_z1_bottom_top_grads = []
    D_G_z2_bottom_top_grads = []

    iters = 0

    print("Starting Training Loop...")
    for epoch in range(num_epochs):
        # For each batch in the dataloader
        for i, data in enumerate(dataloader, 0):


            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            ## Train with all-real batch
            netD.zero_grad()
            # Format batch
            real_cpu = data[0].to(device).float() # changing this to fit MNIST upsampling
            b_size = real_cpu.shape[0]
            label = torch.full((b_size,), real_label, device=device)
            # Forward pass real batch through D
            output = netD(real_cpu).view(-1)
            # Calculate loss on all-real batch
            errD_real = criterion(output, label)
            # Calculate gradients for D in backward pass
            errD_real.backward()
            D_x_bottom_top_grad = bottom_top_grads(netD)
            D_x = output.mean().item()

            ## Train with all-fake batch
            # Generate batch of latent vectors
            noise = torch.randn(b_size, nz, 1, 1, device=device)
            # Generate fake image batch with G
            fake = netG(noise)
            label.fill_(fake_label)
            # Classify all fake batch with D
            output = netD(fake.detach())
            # Calculate D's loss on the all-fake batch
            errD_fake = criterion(output, label)
            # Calculate the gradients for this batch
            errD_fake.backward()
            D_G_z1_bottom_top_grad = bottom_top_grads(netD)
            D_G_z1 = output.mean().item()
            # Add the gradients from the all-real and all-fake batches
            errD = errD_real + errD_fake
            # Update D
            optimizerD.step()

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            netG.zero_grad()
            label.fill_(real_label)  # fake labels are real for generator cost
            # Since we just updated D, perform another forward pass of all-fake batch through D
            output = netD(fake).view(-1)
            # Calculate G's loss based on this output
            errG = criterion(output, label)
            # Calculate gradients for G
            errG.backward()
            D_G_z2_bottom_top_grad = bottom_top_grads(netG)
            D_G_z2 = output.mean().item()
            # Update G
            optimizerG.step()

            # Output training stats
            if i % 50 == 0:
                print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                      % (epoch, num_epochs, i, len(dataloader),
                         errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

            # Save Losses for plotting later
            D_x_losses.append(errD_real.item())
            D_G_z1_losses.append(errD_fake.item())
            D_G_z2_losses.append(errG.item())
            total_D_losses.append(errD.item())

            # Save bottom-top gradients for plotting later
            D_x_bottom_top_grads.append(D_x_bottom_top_grad)
            D_G_z1_bottom_top_grads.append(D_G_z1_bottom_top_grad)
            D_G_z2_bottom_top_grads.append(D_G_z2_bottom_top_grad)

            # Save Accuracies for plotting later
            D_x_acc.append(D_x) # Want to maximize to 1
            D_G_z1_acc.append(D_G_z1) # want to minimize to 0
            D_G_z2_acc.append(D_G_z2) # want to maximize to 1

            # Check how the generator is doing by saving G's output on fixed_noise
            if (iters % 500 == 0) or ((epoch == num_epochs-1) and (i == len(dataloader)-1)):
                with torch.no_grad():
                    fake = netG(fixed_noise).detach().cpu()
                img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

            iters += 1

    # Update model parameters
    netD_params = netD.state_dict()
    netG_params = netG.state_dict()

    # save parameters
    torch.save(netD_params,"netD_params.pt")
    torch.save(netG_params,"netG_params.pt")

    # save img_list for visualization purposes
    torch.save(img_list,"img_list.pt")


    D_x_losses = np.array(D_x_losses)
    D_G_z1_losses = np.array(D_G_z1_losses)
    D_G_z2_losses = np.array(D_G_z2_losses)
    total_D_losses = np.array(total_D_losses)

    all_losses = np.concatenate((D_x_losses,D_G_z1_losses,D_G_z2_losses,total_D_losses)).reshape(-1,4)



    D_x_acc = np.array(D_x_acc)
    D_G_z1_acc = np.array(D_G_z1_acc)
    D_G_z2_acc = np.array(D_G_z2_acc)

    all_accs = np.concatenate((D_x_acc, D_G_z1_acc,D_G_z2_acc)).reshape(-1,3)


    # dims = iterations X 2
    D_x_bottom_top_grads = np.array(D_x_bottom_top_grads)
    D_total_bottom_top_grads = np.array(D_G_z1_bottom_top_grads)
    D_G_z1_bottom_top_grads = D_total_bottom_top_grads - D_x_bottom_top_grads

    D_G_z2_bottom_top_grads = np.array(D_G_z2_bottom_top_grads)

    all_grads = np.concatenate((D_x_bottom_top_grads, D_G_z1_bottom_top_grads, D_total_bottom_top_grads,D_G_z2_bottom_top_grads), 1)

    # Concatenate all
    all = np.concatenate((all_losses, all_accs, all_grads),1)


    # Create DataFrame
    names = ['D_x_losses','D_G_z1_losses','D_G_z2_losses','total_D_losses', 'D_x_acc', 'D_G_z1_acc', 'D_G_z2_acc','D_x_bottom_grads','D_x_top_grads', 'D_G_z1_bottom_grads','D_G_z1_top_grads','D_total_bottom_grads','D_total_top_grads' ,'D_G_z2_bottom_grads','D_G_z2_top_grads']

    df = pd.DataFrame(all, columns = names)
    df.to_csv('df.pt')


