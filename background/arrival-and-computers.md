# BEFORE YOU ARRIVE!
-  **VIRS students check your email for the two-week arrival email and complete the tasks you can.**
- Be sure to get adaquete practice with PyTorch and read up on some tutorials. Ideally, train a simple MLP with MNIST for practice. There are tutorials online.
- Read Mariko's paper(s) **before and after** performing this practice, and look into the P3109 interim report. Both provide important background to the project and each file can be found in Boris/background/ (this directory).
- Familiarize your self with adding, multiplying, etc. floating point numbers by hand as outlined in the P3109 standard.
- (Optional, but reccomended) Look into MPTorch as well to get an idea of how it will work.

## Upon Arrival
You will need to gain access to the lab and building. **This can only be done after getting your UBC student card, so do this ASAP.** Until you have access, someone at the lab can let you in, but this is less than ideal. <br>

To get access, go to the Fred Kaiser (KAIS) Building and go to the fifth floor. Inform them that you are a VIRS student (if applicable) and let them know that you have arrived. You are requesting access for **KAIS 4025** as well as 24/7 access to both it and the building entrances. It is good practice to do this, however they will most likely hand you a card that simply states how to gain access. If you want to skip all of this, send an email to access@ece.ubc.ca with the following information:

- Your full name
- Student ID number
- Name of supervisor
- Role
- Building, room, lab numbers (KAIS 4025, access to main doors, 24/7)
- Expected end date

**Please ensure that all these tasks are completed ASAP. Ask group members for assistance if you have questions.**

# Gaining Access & Connecting

Your CWL will allow you to VPN into UBC while off-campus. 
The VPN account name is the same as your CWL. <br>

You will also need to make a local ECE account to connect to the ECE computers.
The website for IT and the PDF needed to make an account can be found [here.](https://eng-services.ece.ubc.ca/help/it-and-account-support/)
Or, if you just need the PDF, go [here.](https://help.ece.ubc.ca/mediawiki/images/5/5a/Account_application.pdf)

To connect to the computers from outside ECE, use **ssh-linux.ece.ubc.ca** as the initial destination. 
This is a front-end for several numbered ssh-linux machines in ECE. 
From here, you are inside the ECE firewall, so you can access any ECE computer using your ECE account.

You can also skip the ECE firewall using a VPN to ssh directly to the computers without needing ssh-linux.
To do this, you use a variation of your CWL account name (CWL.ece) with the same VPN process.

## Computers
There are two important ECE computers shared by several groups or about 100 users.

**hydra2 (hydra2.ece.ubc.ca) specs:<br>**
-4 x 1080 Ti GPUs (later to be moved to hydra and replaced by 3 x Titan V)<br>
-44 Cores<br>
-768 GB Ram<br>

**hyra (hydra.ece.ubc.ca) specs:<br>**
-Soon to have 4 x 1080 Ti GPUs<br>
-32 Cores<br>
-512 GB Ram<br>

There are also three other computers not considered ECE computers found inside the lab:<br> 
-**gomeisa** is the dell machine, currently with a Titan V.<br>
-**mxp** and **rvv** are two machines each with a RTX 4090 and 192 GB Ram.

One possibly important thing to note is that the 4090s and Titan Vs are running different drivers. Titan V machines are on Nvidia version 535 and cuda 12.2. 4090s are on 555 and 12.5, respectively.
This can further be verified by running nvidia-smi in the terminal.

### Important Notes on Some Errors<br>
Upon running MPTorch or PyTorch on our lab machines, you might run into several cuda errors or errors about an incompatible architecture. You will have to ensure the cuda directory is appended to $PATH and that CUDA_HOME is set properly. Furthermore, this must be added to ~/.bashrc.

For the **Titan V** machines:<br>
`echo "export PATH=$PATH:/usr/local/cuda-12.2" >> ~/.bashrc`<br>
`echo "export CUDA_HOME=/usr/local/cuda-12.2" >> ~/.bashrc`

For the **4090** machines:<br>
`echo "export PATH=$PATH:/usr/local/cuda-12.5" >> ~/.bashrc`<br>
`echo "export CUDA_HOME=/usr/local/cuda-12.5" >> ~/.bashrc`

You can verify this worked by checking ~/.bashrc or by using the `echo` command followed by `$PATH` or `$CUDA_HOME`, depending on what you check.
  

