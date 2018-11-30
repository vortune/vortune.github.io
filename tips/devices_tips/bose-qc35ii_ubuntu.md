# Connecting Bose QC-35 II to Ubuntu Linux

The accepted answer did not work for me. [This blog entry worked: ](http://erikdubois.be/installing-bose-quietcomfort-35-linux-mint-18/)

1. Create `/etc/bluetooth/audio.conf`

   ```
   [General]
   Disable=Socket
   Disable=Headset
   Enable=Media,Source,Sink,Gateway
   AutoConnect=true
   load-module module-switch-on-connect
   ```
2. In `/etc/bluetooth/main.conf` set

   ```
   ControllerMode = bredr
   AutoEnable=true
   ```
3. Restart bluetooth

4. Connect your headphones **[NOTE] Push forward the power button on the right side of headset and hold 5 second**.

5. Choose `High Fidelity Playback (A2DP sink)`-mode in sound options