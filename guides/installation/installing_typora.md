# Installing Typora in Ubuntu
```shell
$ sudo apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv-keys BA300B7755AFCFAE

Executing: /tmp/apt-key-gpghome.F32Mxu28oG/gpg.1.sh --keyserver hkp://keyserver.ubuntu.com:80 --recv-keys BA300B7755AFCFAE
gpg: key BA300B7755AFCFAE: "Abner Lee <abner@typora.io>" not changed
gpg: Total number processed: 1
gpg:              unchanged: 1

sudo add-apt-repository 'deb https://typora.io/linux ./'

sudo apt-get update

# install typora
sudo apt-get install typora
```