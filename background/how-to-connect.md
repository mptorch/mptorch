## Gaining Access & Connecting

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

