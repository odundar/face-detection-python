Copy `files/intel-openvino.sh` & `files/intel-openvino.conf` file as shown below, they set the environment variables for OpenVINO(TM) system wide.

```bash
sudo cp files/intel-openvino.sh /etc/profile.d/
sudo cp files/intel-openvino.conf /etc/ld.so.conf.d/ 
sudo reboot
```